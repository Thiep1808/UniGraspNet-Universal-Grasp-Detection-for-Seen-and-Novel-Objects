import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import re


class CachedGraspDataset(Dataset):
    def __init__(self, cache_dir):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        if len(self.files) == 0:
            raise ValueError(f"Không tìm thấy file .pt nào trong {cache_dir}")

        print(f"Cached Dataset: Tìm thấy {len(self.files)} mẫu.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data_path = self.files[idx]
        try:
            data = torch.load(data_path, map_location='cpu')

            features = data['features'].squeeze(0)
            coarse_pose = data['coarse_pose'].squeeze(0)
            scene_id = data['scene_id']
            if 'view_id' in data:
                view_id = data['view_id']
            else:
                basename = os.path.basename(data_path)
                nums = re.findall(r'\d+', basename)
                view_id = int(nums[-1]) if len(nums) > 0 else 0

            if 'scores' in data:
                scores = data['scores'].squeeze(0)
            else:
                scores = torch.ones(coarse_pose.shape[0], dtype=torch.float32)

            return {
                'features': features,
                'coarse_pose': coarse_pose,
                'scores': scores,
                'scene_id': int(scene_id),
                'view_id': int(view_id)
            }

        except Exception as e:
            print(f"Lỗi load file {data_path}: {e}")
            return None


def collate_fn_clean(batch):

    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    res = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            res[key] = torch.stack([b[key] for b in batch])
        else:
            res[key] = torch.tensor([b[key] for b in batch])
    return res


def get_cached_loader(cache_dir, batch_size=4, num_workers=4):
    dataset = CachedGraspDataset(cache_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_clean
    )