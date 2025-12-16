import torch
import torch.nn.functional as F


def apply_delta_pose(centers, quats, widths, delta):
    """
    Input:
        delta: (B, 7) [dx, dy, dz, rx, ry, rz, dw]
    """
    new_centers = centers + delta[:, 0:3]
    rot_vec = delta[:, 3:6]  # (B, 3)
    theta = torch.norm(rot_vec, dim=1, keepdim=True)
    mask_small = theta < 1e-8


    half_theta = theta / 2
    scale = torch.sin(half_theta) / (theta + 1e-8)
    scale[mask_small] = 0.5

    delta_q_xyz = rot_vec * scale
    delta_q_w = torch.cos(half_theta)

    d_quat = torch.cat([delta_q_xyz, delta_q_w], dim=1)  # (B, 4)

    x, y, z, w = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    dx, dy, dz, dw = d_quat[:, 0], d_quat[:, 1], d_quat[:, 2], d_quat[:, 3]
    nw = w * dw - x * dx - y * dy - z * dz
    nx = w * dx + x * dw + y * dz - z * dy
    ny = w * dy - x * dz + y * dw + z * dx
    nz = w * dz + x * dy - y * dx + z * dw

    new_quats = torch.stack([nx, ny, nz, nw], dim=1)
    new_quats = F.normalize(new_quats, p=2, dim=1)
    new_widths = widths + delta[:, 6:7]

    return new_centers, new_quats, new_widths


def get_gripper_points(centers, quats, widths, num_points=10):
    B = centers.shape[0]
    device = centers.device

    x, y, z, w = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.zeros(B, 3, 3, device=device)
    R[:, 0, 0] = 1 - 2 * (y2 + z2);
    R[:, 0, 1] = 2 * (xy - wz);
    R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz);
    R[:, 1, 1] = 1 - 2 * (x2 + z2);
    R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy);
    R[:, 2, 1] = 2 * (yz + wx);
    R[:, 2, 2] = 1 - 2 * (x2 + y2)

    finger_x = torch.linspace(-0.01, 0.04, num_points, device=device)

    pts_left = torch.zeros(B, num_points, 3, device=device)
    pts_left[:, :, 0] = finger_x;
    pts_left[:, :, 1] = -widths / 2

    pts_right = torch.zeros(B, num_points, 3, device=device)
    pts_right[:, :, 0] = finger_x;
    pts_right[:, :, 1] = widths / 2

    local_pts = torch.cat([pts_left, pts_right], dim=1)  # (B, 2N, 3)

    world_pts = torch.bmm(R, local_pts.transpose(1, 2)).transpose(1, 2) + centers.unsqueeze(1)
    return world_pts