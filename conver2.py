import os
import sys
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from graspnetAPI import GraspGroup
from scipy.spatial.transform import Rotation as R

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet_sparseconv import GraspNet_MSCQ, pred_decode

try:
    from mink_dataset import GraspNetDataset_fusion, minkowski_collate_fn
except ImportError:
    from mink_dataset2 import GraspNetDataset_fusion, minkowski_collate_fn

from collision_detector import ModelFreeCollisionDetector
from loss_utils import batch_viewpoint_params_to_matrix
from models.qpnet import RNG_QP_Net
from normalization_utils import GraspNormalizer
from gripper_utils import apply_delta_pose

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default="/home/tanpx/PycharmProjects/pythonProject/data")
parser.add_argument('--checkpoint_path', default="/home/tanpx/Downloads/checkpoint.tar")
# Dùng checkpoint tốt nhất hiện tại
parser.add_argument('--refine_path', default="checkpoints_rng_final_fix/rng_net_ep32.pth")
parser.add_argument('--dump_dir', default="test_results_balanced")  # Folder mới
parser.add_argument('--camera', default="realsense")
parser.add_argument('--num_point', type=int, default=20000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--collision_thresh', type=float, default=0.01)
parser.add_argument('--voxel_size', type=float, default=0.01)
cfgs = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not os.path.exists(cfgs.dump_dir): os.makedirs(cfgs.dump_dir)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def convert_to_qp_input(end_points):
    features = end_points['fp2_features'].permute(0, 2, 1)
    centers = end_points['fp2_xyz']
    grasp_score = end_points['grasp_score_pred']
    best_depth_idx = torch.argmax(grasp_score, dim=2, keepdim=True)
    grasp_angle = torch.gather(end_points['grasp_angle_value_pred'], 2, best_depth_idx).squeeze(-1)
    grasp_width = torch.clamp(torch.gather(end_points['grasp_width_pred'], 2, best_depth_idx).squeeze(-1), 0,
                              0.1).unsqueeze(-1)
    approaching = -end_points['grasp_top_view_xyz']
    rot_mats = batch_viewpoint_params_to_matrix(approaching.reshape(-1, 3), grasp_angle.reshape(-1))

    rot_np = rot_mats.detach().cpu().numpy().reshape(-1, 3, 3)
    quats = torch.from_numpy(R.from_matrix(rot_np).as_quat()).to(centers.device).float().view(centers.shape[0], -1, 4)
    rot_mats_flat = rot_mats.view(centers.shape[0], -1, 9)
    return features, torch.cat([centers, quats, grasp_width], dim=-1), rot_mats_flat


def decode_smart_v2(refined_pose, original_end_points, top_k_indices, delta_norm):
    B, N, _ = refined_pose.shape
    device = refined_pose.device
    final_centers = refined_pose[..., :3]
    final_quats = refined_pose[..., 3:7]
    final_widths = refined_pose[..., 7:8]
    x, y, z, w = final_quats[..., 0], final_quats[..., 1], final_quats[..., 2], final_quats[..., 3]
    appr_x = 1 - 2 * (y ** 2 + z ** 2)
    appr_y = 2 * (x * y + w * z)
    appr_z = 2 * (x * z - w * y)
    approach_vec = torch.stack([appr_x, appr_y, appr_z], dim=-1)

    final_centers = final_centers - approach_vec * 0.005
    full_score = original_end_points['grasp_score_pred']
    best_depth_idx = torch.argmax(full_score, dim=2, keepdim=True)
    full_graspness = original_end_points['fp2_graspness']
    orig_scores_all = torch.gather(full_score, 2, best_depth_idx).squeeze(-1) * full_graspness
    idx_gather = top_k_indices if top_k_indices.dim() > 1 else top_k_indices.unsqueeze(0)
    final_scores = torch.gather(orig_scores_all, 1, idx_gather)

    final_quats = final_quats / (torch.norm(final_quats, dim=-1, keepdim=True) + 1e-8)
    r = R.from_quat(final_quats.reshape(-1, 4).detach().cpu().numpy())
    rot_mats_flat = torch.from_numpy(r.as_matrix().reshape(B, N, 9)).float().to(device)

    preds = torch.cat([
        final_scores.unsqueeze(-1), final_widths,
        torch.full((B, N, 1), 0.02, device=device), torch.full((B, N, 1), 0.02, device=device),
        rot_mats_flat, final_centers, torch.full((B, N, 1), -1.0, device=device)
    ], dim=-1)
    return preds


def run_test():
    setup_seed(822)
    print("Running Inference: Preservation Strategy...")

    net = GraspNet_MSCQ(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                        cylinder_radius=0.08, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04],
                        is_training=False).to(device)
    net.load_state_dict(torch.load(cfgs.checkpoint_path, map_location=device)['model_state_dict'])
    net.eval()

    rng_model = RNG_QP_Net(n_constraints=12).to(device)
    rng_model.load_state_dict(torch.load(cfgs.refine_path, map_location=device))
    rng_model.eval()

    normalizer = GraspNormalizer(device=device)

    TEST_DATASET = GraspNetDataset_fusion(cfgs.dataset_root, valid_obj_idxs=None, grasp_labels=None, split='test',
                                          camera=cfgs.camera, num_points=cfgs.num_point,
                                          remove_outlier=True, augment=False, load_label=False, use_fine=False)
    SCENE_LIST = TEST_DATASET.scene_list()

    def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=1, shuffle=False,
                                 num_workers=4, worker_init_fn=my_worker_init_fn,
                                 collate_fn=minkowski_collate_fn)

    for batch_idx, batch_data in enumerate(TEST_DATALOADER):
        for k in batch_data:
            if isinstance(batch_data[k], list): continue
            batch_data[k] = batch_data[k].to(device)

        with torch.no_grad():
            end_points = net(batch_data)
            grasp_preds_baseline, _ = pred_decode(end_points)
            preds_orig = grasp_preds_baseline[0]

            scores_raw = torch.max(end_points['grasp_score_pred'], dim=2)[0] * end_points['fp2_graspness']
            K_LIMIT = 2000
            top_k_vals, top_k_indices = torch.topk(scores_raw.view(-1), k=min(K_LIMIT, scores_raw.numel()))
            top_k_indices = top_k_indices.unsqueeze(0)

            _, coarse_pose_batch, coarse_rot_flat = convert_to_qp_input(end_points)

            idx_exp_8 = top_k_indices.unsqueeze(-1).expand(-1, -1, 8)
            coarse_pose_sub = torch.gather(coarse_pose_batch, 1, idx_exp_8).squeeze(0)
            idx_exp_9 = top_k_indices.unsqueeze(-1).expand(-1, -1, 9)
            rot_mats_sub_flat = torch.gather(coarse_rot_flat, 1, idx_exp_9).squeeze(0)

            centers_sub = coarse_pose_sub[:, :3]
            quats_sub = coarse_pose_sub[:, 3:7]
            widths_sub = coarse_pose_sub[:, 7:8]

            refined_list = []

            if isinstance(batch_data['point_clouds'], list):
                scene_pcd_batch = batch_data['point_clouds'][0].to(device)
            else:
                scene_pcd_batch = batch_data['point_clouds'][0]

            BATCH_SIZE = 64
            for i in range(0, len(centers_sub), BATCH_SIZE):
                curr_c = centers_sub[i:i + BATCH_SIZE]
                curr_r = rot_mats_sub_flat[i:i + BATCH_SIZE].view(-1, 3, 3)
                curr_w = widths_sub[i:i + BATCH_SIZE]
                curr_q = quats_sub[i:i + BATCH_SIZE]

                patches, scales = normalizer.normalize_batch(scene_pcd_batch.unsqueeze(0), curr_c, curr_r, curr_w)
                delta_norm, _, _, _ = rng_model(patches, curr_r)

                delta_world = normalizer.denormalize_delta(delta_norm, curr_r, scales)
                nc, nq, nw = apply_delta_pose(curr_c, curr_q, curr_w, delta_world)

                refined_list.append(torch.cat([nc, nq, nw], dim=1))

            refined_full = torch.cat(refined_list, dim=0).unsqueeze(0)
            preds_refined = decode_smart_v2(refined_full, end_points, top_k_indices, torch.zeros_like(refined_full))

        gg_orig = GraspGroup(preds_orig.detach().cpu().numpy())
        gg_refine = GraspGroup(preds_refined[0].detach().cpu().numpy())

        if cfgs.collision_thresh > 0:
            raw_cloud, _ = TEST_DATASET.get_data(batch_idx, return_raw_cloud=True)
            mfcdetector = ModelFreeCollisionDetector(raw_cloud, voxel_size=cfgs.voxel_size)

            mask_coll_orig = mfcdetector.detect(gg_orig, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
            mask_coll_refine = mfcdetector.detect(gg_refine, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)

            gg_safe_orig = gg_orig[~mask_coll_orig]

            top_k_indices_np = top_k_indices.view(-1).cpu().numpy()
            mask_coll_orig_subset = mask_coll_orig[top_k_indices_np]

            rescue_mask = mask_coll_orig_subset & (~mask_coll_refine)
            gg_rescue = gg_refine[rescue_mask]

            if len(gg_rescue) > 0:
                gg_rescue.scores = gg_rescue.scores * 0.75

                final_arr = np.concatenate([gg_safe_orig.grasp_group_array, gg_rescue.grasp_group_array], axis=0)
                gg_merged = GraspGroup(final_arr)

                gg_final = gg_merged.nms(translation_thresh=0.03, rotation_thresh=30.0 / 180.0 * np.pi)
            else:
                gg_final = gg_safe_orig
        else:
            gg_final = gg_orig

        gg_final = gg_final.sort_by_score()

        save_path = os.path.join(cfgs.dump_dir, SCENE_LIST[batch_idx], 'realsense')
        if not os.path.exists(save_path): os.makedirs(save_path)
        gg_final.save_npy(os.path.join(save_path, 'result.npy'))

        print(f"Scene {SCENE_LIST[batch_idx]}: Final={len(gg_final)} (Rescued {len(gg_rescue)})")


if __name__ == '__main__':
    run_test()