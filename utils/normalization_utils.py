import torch


class GraspNormalizer:
    def __init__(self, device='cuda'):
        self.device = device

    def normalize_batch(self, scene_points, grasp_centers, grasp_rots, grasp_widths, num_sample=512):
        if scene_points.dim() == 3:
            scene_points = scene_points.squeeze(0)

        B = grasp_centers.shape[0]
        normalized_patches = []
        scales = grasp_widths.clone()
        scales[scales < 0.001] = 0.001

        for i in range(B):
            P_world = scene_points
            Center = grasp_centers[i]
            Rot = grasp_rots[i]  # (3, 3) Local -> World
            Scale = scales[i]

            radius = max(0.05, 1.5 * Scale.item())
            dists = torch.norm(P_world - Center, dim=1)
            mask = dists < radius
            local_P = P_world[mask]

            if local_P.shape[0] < num_sample:
                if local_P.shape[0] == 0:
                    patch = torch.zeros(num_sample, 3, device=self.device)
                else:
                    repeat_times = (num_sample // local_P.shape[0]) + 1
                    patch = local_P.repeat(repeat_times, 1)[:num_sample]
            else:
                perm = torch.randperm(local_P.shape[0])[:num_sample]
                patch = local_P[perm]

            # --- SỬA LỖI QUAN TRỌNG NHẤT ---
            P_centered = patch - Center
            # Muốn chuyển World -> Local thì phải nhân với Inverse của Rot (tức là Rot.T)
            # P_local = (P_world - T) @ R.T
            P_rotated = torch.matmul(P_centered, Rot.t())

            patch_norm = P_rotated / Scale
            normalized_patches.append(patch_norm)

        return torch.stack(normalized_patches), scales

    def denormalize_delta(self, delta_norm, grasp_rots, scales):
        # Translation: Local -> World (Cần xoay)
        dt_norm = delta_norm[:, :3]
        dt_world = torch.bmm(grasp_rots, (dt_norm * scales).unsqueeze(-1)).squeeze(-1)

        # Rotation: Local -> Local (GIỮ NGUYÊN, KHÔNG XOAY)
        # Vì hàm apply_delta_pose sẽ áp dụng nó vào Local Frame
        dr_local = delta_norm[:, 3:6]

        # Width: Scale
        dw_norm = delta_norm[:, 6:7]
        dw_world = dw_norm * scales

        return torch.cat([dt_world, dr_local, dw_world], dim=1)