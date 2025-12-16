import torch
import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class MiniPointNet(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.fc = nn.Linear(256, feature_dim)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0]
        x = self.fc(x)
        return x


class DifferentiableQPLayer(nn.Module):
    def __init__(self, n_vars=7, n_constraints=12):
        super().__init__()
        self.n_vars = n_vars
        self.n_constraints = n_constraints

        z = cp.Variable(n_vars)
        s = cp.Variable(n_constraints)
        p_target = cp.Parameter(n_vars)
        G_param = cp.Parameter((n_constraints, n_vars))
        h_param = cp.Parameter(n_constraints)
        obj = cp.Minimize(0.5 * cp.sum_squares(z - p_target) + 1000.0 * cp.sum_squares(s))
        cons = [G_param @ z <= h_param + s, s >= 0]
        problem = cp.Problem(obj, cons)

        self.cvxpylayer = CvxpyLayer(problem, parameters=[p_target, G_param, h_param], variables=[z, s])

    def forward(self, p, G, h):
        try:
            z_star, s_star = self.cvxpylayer(p, G, h, solver_args={'eps': 1e-4, 'max_iters': 10000})
            return z_star
        except:
            return p


class RNG_QP_Net(nn.Module):
    def __init__(self, n_constraints=12):
        super().__init__()
        self.feature_extractor = MiniPointNet(feature_dim=128)

        self.constraint_net = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_constraints * 4)
        )

        self.p_net = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 7)
        )

        with torch.no_grad():
            self.constraint_net[-1].bias[n_constraints * 3:] = 0.02
            self.p_net[-1].weight.data.normal_(0, 0.001)
            self.p_net[-1].bias.data.zero_()

        self.qp_layer = DifferentiableQPLayer(n_vars=7, n_constraints=n_constraints)
        self.n_cons = n_constraints

    def forward(self, normalized_pcd, rot_mats=None):
        feats = self.feature_extractor(normalized_pcd)
        raw_p = self.p_net(feats)
        p_trans = torch.tanh(raw_p[:, :3]) * 0.1
        p_rot = torch.tanh(raw_p[:, 3:6]) * 1.57
        p_width = torch.tanh(raw_p[:, 6:7]) * 0.05
        p_pred = torch.cat([p_trans, p_rot, p_width], dim=1)
        out_cons = self.constraint_net(feats)
        out_cons = out_cons.view(-1, self.n_cons, 4)
        pred_normals = out_cons[:, :, :3]
        pred_h = out_cons[:, :, 3]
        pred_normals = F.normalize(pred_normals, p=2, dim=2)
        pred_h = F.softplus(pred_h) + 0.005
        B = normalized_pcd.shape[0]
        zeros_rot = torch.zeros(B, self.n_cons, 4, device=normalized_pcd.device)
        G = torch.cat([pred_normals, zeros_rot], dim=2)
        delta_norm = self.qp_layer(p_pred, G, pred_h)

        return delta_norm, pred_normals, pred_h, p_pred