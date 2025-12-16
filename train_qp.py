import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import glob
from tqdm import tqdm
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
import torch.nn as nn

# Import các module
from models.qpnet import RNG_QP_Net
from utils.normalization_utils import GraspNormalizer
from utils.gripper_utils import apply_delta_pose, get_gripper_points
from utils.sdf_utils import SignedDistanceField
from dataset.dataset_cached import get_cached_loader

CONFIG = {
    'cache_root': '/home/tanpx/PycharmProjects/pythonProject/data/cached_features_refine',
    'graspnet_root': '/home/tanpx/PycharmProjects/pythonProject/data',
    'sdf_root': '/home/tanpx/PycharmProjects/pythonProject/data/sdf_data',
    'batch_size': 4,
    'lr': 2e-4,   # Tăng nhẹ LR
    'epochs': 50,
    'device': 'cuda',
    'save_dir': 'checkpoints_rng_final_fix' # Đổi tên folder save
}


# ... (Class SceneManager giữ nguyên như cũ) ...
class SceneManager:
    def __init__(self, graspnet_root, sdf_root, device):
        self.graspnet_root = graspnet_root
        self.device = device
        self.sdf_cache = {}
        self.sdf_root = sdf_root
        self.scene_meta_cache = {}
        # Preload ít thôi cho nhanh
        print("Preloading SDFs...")
        sdf_files = glob.glob(os.path.join(sdf_root, "*", "textured.npy"))
        for path in tqdm(sdf_files[:10]):
            try:
                obj_id = int(os.path.basename(os.path.dirname(path)))
                self.sdf_cache[obj_id] = SignedDistanceField(path, device=device)
            except:
                pass

    def get_sdf(self, obj_id):
        # ... logic cũ
        if obj_id not in self.sdf_cache:
            path1 = os.path.join(self.sdf_root, f"{int(obj_id):03d}", "textured.npy")
            path2 = os.path.join(self.sdf_root, f"{int(obj_id):03d}", "nontextured.npy")
            target = path1 if os.path.exists(path1) else path2
            if os.path.exists(target):
                try:
                    self.sdf_cache[obj_id] = SignedDistanceField(target, device=self.device)
                except:
                    self.sdf_cache[obj_id] = None
            else:
                self.sdf_cache[obj_id] = None
        return self.sdf_cache[obj_id]

    def get_scene_pcd(self, scene_id, view_id):
        scene_name = f'scene_{int(scene_id):04d}'
        ply_path = os.path.join(self.graspnet_root, 'scenes', scene_name, 'realsense', 'cloud',
                                f'{int(view_id):04d}.ply')
        if not os.path.exists(ply_path):
            ply_path = os.path.join(self.graspnet_root, 'fusion_scenes', scene_name, 'realsense', 'points.npy')

        if os.path.exists(ply_path):
            try:
                if ply_path.endswith('.ply'):
                    import open3d as o3d
                    pcd = o3d.io.read_point_cloud(ply_path)
                    pcd = pcd.voxel_down_sample(voxel_size=0.005)
                    points = np.asarray(pcd.points)
                else:
                    data = np.load(ply_path, allow_pickle=True).item()
                    points = np.array(data['xyz'])
                    if points.shape[0] > 20000:
                        idx = np.random.choice(points.shape[0], 20000, replace=False)
                        points = points[idx]
                return torch.from_numpy(points).float().to(self.device)
            except:
                return None
        return None

    def get_scene_meta(self, scene_id, view_id):
        cache_key = f"{scene_id}_{view_id}"
        if cache_key in self.scene_meta_cache: return self.scene_meta_cache[cache_key]

        scene_name = f'scene_{int(scene_id):04d}'
        try:
            obj_list_path = os.path.join(self.graspnet_root, 'scenes', scene_name, 'object_id_list.txt')
            if not os.path.exists(obj_list_path): return None
            with open(obj_list_path, 'r') as f:
                obj_ids = [int(line.strip()) for line in f.readlines()]

            align_mat_path = os.path.join(self.graspnet_root, 'scenes', scene_name, 'realsense',
                                          'cam0_wrt_table.npy')
            if os.path.exists(align_mat_path):
                align_mat = np.load(align_mat_path)
            else:
                align_mat = np.eye(4)
            align_mat_t = torch.from_numpy(align_mat).float().to(self.device)

            meta_path = os.path.join(self.graspnet_root, 'scenes', scene_name, 'realsense', 'meta',
                                     f'{int(view_id):04d}.mat')
            if not os.path.exists(meta_path): return None

            meta_data = loadmat(meta_path)
            poses = meta_data['poses']

            inv_poses, world_poses = [], []

            def process_pose(p_np):
                p = np.eye(4);
                p[:3, :] = p_np
                t_pose_cam = torch.from_numpy(p).float().to(self.device)
                t_pose_aligned = torch.matmul(align_mat_t, t_pose_cam)
                world_poses.append(t_pose_aligned)

                R = t_pose_aligned[:3, :3]
                t = t_pose_aligned[:3, 3]
                inv_p = torch.eye(4, device=self.device)
                inv_p[:3, :3] = R.t()
                inv_p[:3, 3] = -torch.matmul(R.t(), t)
                inv_poses.append(inv_p)

            if poses.ndim == 3 and poses.shape[2] == len(obj_ids):
                for i in range(len(obj_ids)): process_pose(poses[:, :, i])
            elif poses.ndim == 3 and poses.shape[0] == len(obj_ids):
                for i in range(len(obj_ids)): process_pose(poses[i, :, :])
            else:
                return None

            data = {'obj_ids': obj_ids, 'inv_poses': inv_poses, 'world_poses': world_poses}
            self.scene_meta_cache[cache_key] = data
            return data
        except:
            return None

    def compute_collision_loss(self, refined_pose, scene_id, view_id):
        meta = self.get_scene_meta(scene_id, view_id)
        if meta is None: return torch.tensor(0.0, device=self.device, requires_grad=True)

        centers = refined_pose[:, 0:3]
        quats = refined_pose[:, 3:7]
        widths = refined_pose[:, 7:8]

        gripper_pts = get_gripper_points(centers, quats, widths, num_points=5)
        flat_pts = gripper_pts.reshape(-1, 3)
        ones = torch.ones(flat_pts.shape[0], 1, device=self.device)
        pts_homo = torch.cat([flat_pts, ones], dim=1)

        total_loss = torch.tensor(0.0, device=self.device)

        for k, oid in enumerate(meta['obj_ids']):
            sdf = self.get_sdf(oid)
            if sdf is None: continue

            pts_obj = torch.matmul(pts_homo, meta['inv_poses'][k].t())[:, :3]
            vals = sdf.get_sdf(pts_obj)

            # Phạt nếu đâm vào (vals < -0.002)
            total_loss = total_loss + torch.mean(F.softplus(-vals - 0.002))

        return total_loss

    def get_gt_info(self, centers, quats, widths, scene_id, view_id, num_check_points=12):
        meta = self.get_scene_meta(scene_id, view_id)
        if meta is None: return None, None

        B = centers.shape[0]

        # --- CẢI TIẾN: Chỉ dùng Center để lấy Normal đại diện ---
        # Điều này giúp Target Normal ổn định, không bị nhảy lung tung khi Gripper xoay
        flat_pts = centers  # (B, 3)
        ones = torch.ones(flat_pts.shape[0], 1, device=self.device)
        pts_homo = torch.cat([flat_pts, ones], dim=1)

        gt_normals_world = torch.zeros_like(flat_pts)
        gt_sdf_vals = torch.ones(flat_pts.shape[0], device=self.device)

        min_dists = torch.full((flat_pts.shape[0],), float('inf'), device=self.device)
        found_any = False

        for k, oid in enumerate(meta['obj_ids']):
            sdf = self.get_sdf(oid)
            if sdf is None: continue

            pts_obj = torch.matmul(pts_homo, meta['inv_poses'][k].t())[:, :3]
            vals, grads_obj = sdf.get_sdf_and_gradient(pts_obj)
            vals_abs = torch.abs(vals)

            # Lấy normal nếu khoảng cách < 10cm (nới rộng ra chút để bắt signal)
            closer_mask = (vals_abs < min_dists) & (vals_abs < 0.1)

            if closer_mask.any():
                found_any = True
                R_model2world = meta['world_poses'][k][:3, :3]
                grads_world = torch.matmul(grads_obj, R_model2world.t())

                min_dists[closer_mask] = vals_abs[closer_mask]
                gt_normals_world[closer_mask] = grads_world[closer_mask]
                gt_sdf_vals[closer_mask] = vals[closer_mask]

        if not found_any: return None, None

        # Transform Normal từ World sang Local để tính Loss dễ hơn (nếu cần)
        # Nhưng ở đây ta trả về World Normals, vì compute_alignment_loss dùng world quat

        # Reshape cho khớp output cũ (B, 1, 3)
        gt_normals_world = gt_normals_world.unsqueeze(1)
        gt_sdf_vals = gt_sdf_vals.unsqueeze(1)

        return gt_normals_world, gt_sdf_vals

def compute_alignment_loss(pred_quats, gt_normals):
    x, y, z, w = pred_quats[:, 0], pred_quats[:, 1], pred_quats[:, 2], pred_quats[:, 3]
    # Approach vector (X-axis)
    gripper_x_axis = torch.stack([
        1 - 2 * (y * y + z * z),
        2 * (x * y + w * z),
        2 * (x * z - w * y)
    ], dim=1)
    gripper_x_axis = F.normalize(gripper_x_axis, dim=1)

    target_normal = gt_normals.squeeze(1)
    target_normal = F.normalize(target_normal, dim=1)

    # Dot product
    dot_prod = torch.sum(gripper_x_axis * target_normal, dim=1)
    loss_align = torch.mean(1.0 + dot_prod)
    return loss_align


def train():
    if not os.path.exists(CONFIG['save_dir']): os.makedirs(CONFIG['save_dir'])

    scene_manager = SceneManager(CONFIG['graspnet_root'], CONFIG['sdf_root'], CONFIG['device'])
    normalizer = GraspNormalizer(device=CONFIG['device'])
    model = RNG_QP_Net(n_constraints=12).to(CONFIG['device'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])  # lr=2e-4
    dataloader = get_cached_loader(CONFIG['cache_root'], batch_size=CONFIG['batch_size'])
    mse_crit = nn.MSELoss()

    print("--- START TRAINING (CURRICULUM LEARNING) ---")

    for epoch in range(CONFIG['epochs']):
        model.train()
        pbar = tqdm(dataloader)
        epoch_loss = 0
        valid_batches = 0

        # --- CHIẾN THUẬT CURRICULUM ---
        # 5 Epoch đầu: Tắt Collision để mạng tập trung học xoay (Align)
        if epoch < 5:
            w_coll = 0.0
            w_align = 200.0  # Tăng cực mạnh
            w_p = 50.0
        else:
            w_coll = 20.0  # Bật lại nhẹ nhàng
            w_align = 100.0  # Vẫn ưu tiên Align
            w_p = 50.0

        for batch in pbar:
            if batch is None: continue
            bs = len(batch['coarse_pose'])

            for b_idx in range(bs):
                coarse_pose = batch['coarse_pose'][b_idx].to(CONFIG['device'])
                if coarse_pose.shape[0] == 0: continue

                if coarse_pose.shape[0] > 48:
                    perm = torch.randperm(coarse_pose.shape[0])[:48]
                    coarse_pose = coarse_pose[perm]

                scene_id, view_id = int(batch['scene_id'][b_idx]), int(batch['view_id'][b_idx])
                scene_pcd = scene_manager.get_scene_pcd(scene_id, view_id)
                if scene_pcd is None: continue

                centers = coarse_pose[:, :3]
                quats = coarse_pose[:, 3:7]
                widths = coarse_pose[:, 7:8]
                r = R.from_quat(quats.cpu().numpy())
                rot_mats = torch.from_numpy(r.as_matrix()).float().to(CONFIG['device'])

                patches, scales = normalizer.normalize_batch(scene_pcd.unsqueeze(0), centers, rot_mats, widths)
                patches = patches + torch.randn_like(patches) * 0.002

                optimizer.zero_grad()
                delta_norm, pred_normals, pred_h, p_pred = model(patches, rot_mats)

                delta_world = normalizer.denormalize_delta(delta_norm, rot_mats, scales)
                new_centers, new_quats, new_widths = apply_delta_pose(centers, quats, widths, delta_world)
                refined_pose = torch.cat([new_centers, new_quats, new_widths], dim=1)

                # Losses
                loss_coll = scene_manager.compute_collision_loss(refined_pose, scene_id, view_id)
                loss_reg = torch.mean(delta_norm ** 2)

                gt_normals, gt_sdf_vals = scene_manager.get_gt_info(centers, quats, widths, scene_id, view_id)

                if gt_normals is not None:
                    loss_align = compute_alignment_loss(new_quats, gt_normals)

                    min_sdf, min_idx = torch.min(gt_sdf_vals, dim=1)
                    batch_idx_range = torch.arange(gt_normals.shape[0], device=CONFIG['device'])
                    main_normal = gt_normals[batch_idx_range, min_idx, :]
                    penetration = F.relu(-min_sdf + 0.005).unsqueeze(1)
                    target_trans = main_normal * penetration

                    loss_p = mse_crit(p_pred[:, :3], target_trans)
                else:
                    loss_align = torch.tensor(0.0, device=CONFIG['device'])
                    loss_p = torch.tensor(0.0, device=CONFIG['device'])

                # Tổng hợp Loss theo Curriculum
                total_loss = w_coll * loss_coll + w_align * loss_align + w_p * loss_p + 1.0 * loss_reg

                if torch.isnan(total_loss): continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += total_loss.item()
                valid_batches += 1

                pbar.set_description(
                    f"Ep{epoch + 1}| L:{total_loss.item():.1f} C:{loss_coll.item():.3f} A:{loss_align.item():.3f}")

        if valid_batches > 0:
            print(f"Epoch {epoch + 1} Done. Avg Loss: {epoch_loss / valid_batches:.4f}")
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], f"rng_net_ep{epoch + 1}.pth"))


if __name__ == "__main__":
    train()