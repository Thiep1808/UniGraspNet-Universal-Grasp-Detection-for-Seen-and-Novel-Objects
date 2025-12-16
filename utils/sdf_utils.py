import torch
import torch.nn.functional as F
import numpy as np
import os


class SignedDistanceField:
    def __init__(self, sdf_path, device='cuda'):
        self.device = device
        if not os.path.exists(sdf_path):
            raise FileNotFoundError(f"SDF file not found: {sdf_path}")

        data = np.load(sdf_path, allow_pickle=True).item()
        self.sdf_grid_np = data['grid']
        self.origin = torch.from_numpy(data['info']['origin']).float().to(device)
        self.voxel_size = float(data['info']['voxel_size'])
        self.resolution = self.sdf_grid_np.shape[0]
        self.real_size = (self.resolution - 1) * self.voxel_size

        # Grid gốc (1, 1, D, H, W)
        self.sdf_tensor = torch.from_numpy(self.sdf_grid_np).float().to(device)
        self.sdf_tensor = self.sdf_tensor.unsqueeze(0).unsqueeze(0)

    def get_sdf(self, queries):
        input_shape = queries.shape
        queries_flat = queries.reshape(-1, 3)
        N_total = queries_flat.shape[0]

        coords_norm = (queries_flat - self.origin) / self.real_size
        coords_norm = coords_norm * 2.0 - 1.0

        max_vals = coords_norm.abs().max(dim=1).values
        out_of_bound_mask = max_vals > 1.0

        grid_sample_input = coords_norm.view(1, 1, 1, N_total, 3)

        sdf_sampled = F.grid_sample(
            self.sdf_tensor, grid_sample_input,
            mode='bilinear', padding_mode='border', align_corners=True
        )

        sdf_vals = sdf_sampled.view(N_total)
        sdf_vals[out_of_bound_mask] = 1.0

        return sdf_vals.view(input_shape[:-1])

    def get_sdf_and_gradient(self, queries):
        sdf = self.get_sdf(queries)
        epsilon = 1e-3
        gradients = []
        shifts = torch.eye(3, device=self.device) * epsilon

        for i in range(3):
            vec = shifts[i]
            sdf_plus = self.get_sdf(queries + vec)
            sdf_minus = self.get_sdf(queries - vec)
            grad_i = (sdf_plus - sdf_minus) / (2 * epsilon)
            gradients.append(grad_i)

        gradients = torch.stack(gradients, dim=-1)
        return sdf, gradients