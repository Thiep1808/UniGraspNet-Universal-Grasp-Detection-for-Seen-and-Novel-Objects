import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image

import torch
import collections.abc as container_abcs
from torch.utils.data import Dataset
import MinkowskiEngine as ME
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, \
    get_workspace_mask, remove_invisible_grasp_points
from graspnetAPI.utils.utils import xmlReader, parse_posevector
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree


class GraspNetDataset_fusion(Dataset):
    def __init__(self, root, valid_obj_idxs, grasp_labels, camera='kinect', split='train', num_points=20000,
                 remove_outlier=False, remove_invisible=True, augment=False, load_label=True, voxel_size=0.005,
                 use_fine=False):
        assert (num_points <= 50000)
        self.root = root
        self.split = split
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible
        self.valid_obj_idxs = valid_obj_idxs
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.collision_labels = {}
        self.voxel_size = voxel_size

        if split == 'train':
            self.sceneIds = list(range(0, 100))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))

        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]

        self.pcdpath = []
        self.labelpath = []
        self.class_path = []
        self.sampath = []
        self.scenename = []
        self.inspath = []
        for x in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            if use_fine:
                self.pcdpath.append(os.path.join(root, 'fusion_scenes_fine', x, camera, 'points.npy'))
                self.labelpath.append(os.path.join(root, 'fusion_scenes_fine', x, camera, 'seg.npy'))
            else:
                self.pcdpath.append(os.path.join(root, 'fusion_scenes', x, camera, 'points.npy'))
                self.labelpath.append(os.path.join(root, 'fusion_scenes', x, camera, 'seg.npy'))
            self.scenename.append(x.strip())
            if self.load_label:
                collision_labels = np.load(os.path.join(root, 'collision_label', x.strip(), 'collision_labels.npz'))
                self.collision_labels[x.strip()] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.pcdpath)

    def augment_data(self, point_clouds, normals, object_poses_list):
        aug_trans = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        normals = transform_point_cloud(normals, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)
        aug_trans = np.dot(aug_trans, rot_mat.T)

        return point_clouds, normals, object_poses_list, aug_trans

    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
            return self.get_data(index)

    def get_data(self, index, return_raw_cloud=False):
        fusion_data = np.load(self.pcdpath[index], allow_pickle=True).item()
        point_cloud = np.array(fusion_data['xyz'])
        normal = np.array(fusion_data['normal'])
        color = np.array(fusion_data['color'])
        seg = np.array(np.load(self.labelpath[index]))
        scene = self.scenename[index]
        if return_raw_cloud:
            return point_cloud, seg

        if self.camera == "kinect":
            mask_x = ((point_cloud[:, 0] > -0.5) & (point_cloud[:, 0] < 0.5))
            mask_y = ((point_cloud[:, 1] > -0.5) & (point_cloud[:, 1] < 0.5))
            mask_z = ((point_cloud[:, 2] > -0.02) & (point_cloud[:, 2] < 0.2))
            workspace_mask = (mask_x & mask_y & mask_z)
            point_cloud = point_cloud[workspace_mask]
            normal = normal[workspace_mask]
            color = color[workspace_mask]
            seg = seg[workspace_mask]

        if len(point_cloud) >= self.num_points:
            idxs = np.random.choice(len(point_cloud), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(point_cloud))
            idxs2 = np.random.choice(len(point_cloud), self.num_points - len(point_cloud), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = point_cloud[idxs]
        seg_sampled = seg[idxs]
        normal_sampled = normal[idxs]
        color_sampled = color[idxs]
        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label > 1] = 1
        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['coors'] = cloud_sampled.astype(np.float32) / self.voxel_size
        ret_dict['feats'] = normal_sampled.astype(np.float32)
        ret_dict['pcd_color'] = np.concatenate([cloud_sampled.astype(np.float32), color_sampled.astype(np.float32)],
                                               axis=1)

        ret_dict['instance_mask'] = seg_sampled
        return ret_dict

    def compute_curvature(self, points, k=10):
        tree = cKDTree(points)
        curvatures = np.zeros(len(points))
        for i, point in enumerate(points):
            _, indices = tree.query(point, k=min(k + 1, len(points)))
            if len(indices) < 4:
                curvatures[i] = 0.0
                continue

            neighbors = points[indices[1:]]
            if len(neighbors) < 3:
                curvatures[i] = 0.0
                continue

            pca = PCA(n_components=3)
            pca.fit(neighbors)
            eigenvalues = pca.explained_variance_
            eigenvalues_sum = np.sum(eigenvalues)
            if eigenvalues_sum == 0 or np.isnan(eigenvalues_sum):
                curvatures[i] = 0.0  # Hoặc giá trị mặc định
            else:
                curvatures[i] = eigenvalues[-1] / eigenvalues_sum
                if np.isnan(curvatures[i]):
                    curvatures[i] = 0.0
        return curvatures

    def get_data_label(self, index):
        fusion_data = np.load(self.pcdpath[index], allow_pickle=True).item()
        point_cloud = np.array(fusion_data['xyz'])
        normal = np.array(fusion_data['normal'])
        color = np.array(fusion_data['color'])
        seg = np.array(np.load(self.labelpath[index]))

        scene = self.scenename[index]


        cloud_sampled = point_cloud
        seg_sampled = seg
        normal_sampled = normal
        color_sampled = color
        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label > 1] = 1

        align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
        scene_reader = xmlReader(
            os.path.join(self.root, 'scenes', scene, self.camera, 'annotations', '0000.xml'))
        posevectors = scene_reader.getposevectorlist()
        obj_list = []
        poses = []
        for posevector in posevectors:
            obj_idx, pose = parse_posevector(posevector)
            pose = np.matmul(align_mat, pose)
            poses.append(pose)
            obj_list.append(obj_idx + 1)
        poses = np.asarray(poses).astype(np.float32)

        object_poses_list = []
        grasp_points_list = []
        grasp_offsets_list = []
        grasp_scores_list = []
        grasp_tolerance_list = []
        ret_obj_list = []
        graspability_scores = np.zeros(len(cloud_sampled), dtype=np.float32)
        classes_map = np.zeros(len(cloud_sampled), dtype=np.float32)
        l = []
        for i, obj_idx in enumerate(obj_list):
            if obj_idx not in self.valid_obj_idxs:
                continue
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            obj_mask = (seg_sampled == obj_idx)
            obj_points = cloud_sampled[obj_mask]
            object_poses_list.append(poses[i, :3, :4])

            if obj_idx not in self.grasp_labels:
                self._load_grasp_label_for_object(obj_idx)

            points, offsets, scores, classes, tolerance = self.grasp_labels[obj_idx]
            collision = self.collision_labels[scene][i]  # (Np, V, A, D)

            l.extend(np.unique(classes))

            # remove invisible grasp points
            if self.remove_invisible:
                visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled == obj_idx], points,
                                                             poses[i, :3, :4], th=0.01)
                points = points[visible_mask]
                classes = classes[visible_mask]
                offsets = offsets[visible_mask]
                scores = scores[visible_mask]
                tolerance = tolerance[visible_mask]
                # print(collision.shape)
                # print(visible_mask.shape)
                collision = collision[visible_mask]

            idxs = np.random.choice(len(points), len(points), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_offsets_list.append(offsets[idxs])

            ret_obj_list.append(np.asarray([obj_idx - 1]).astype(np.int64))

            scores = scores[idxs].copy()
            collision = collision[idxs].copy()
            scores[collision] = 0
            classes = classes[idxs]
            grasp_scores_list.append(scores)
            tolerance = tolerance[idxs].copy()
            tolerance[collision] = 0
            grasp_tolerance_list.append(tolerance)

            grasp_points2 = transform_point_cloud(points, poses[i, :3, :4])[idxs]
            max_labels = np.max(scores, axis=(1, 2, 3))
            max_labels[max_labels == -1] = 0
            max_tolerance = np.max(tolerance, axis=(1, 2, 3))
            max_tolerance = max_tolerance / (np.max(max_tolerance) + 1e-8)  # Normalize
            collision_any = np.mean(collision, axis=(1, 2, 3)).astype(np.float32)
            tree = cKDTree(grasp_points2)
            distances, indices = tree.query(obj_points, k=1)
            valid_mask = distances <= 0.01
            obj_max_labels = np.zeros(len(obj_points), dtype=np.float32)
            obj_max_labels[valid_mask] = max_labels[indices[valid_mask]]
            curvatures = self.compute_curvature(obj_points, k=10)

            sigma = 1
            obj_graspability_scores = np.zeros(len(obj_points), dtype=np.float32)
            if np.any(valid_mask):
                obj_graspability_scores[valid_mask] = (
                        obj_max_labels[valid_mask] *
                        np.exp(-distances[valid_mask] / sigma))
            graspability_scores[obj_mask] = obj_graspability_scores  # Fixed indexing
            classes = classes.reshape(-1, )
            if np.any(valid_mask):
                obj_indices = np.where(obj_mask)[0]  # index thật trong cloud_sampled
                valid_obj_indices = obj_indices[valid_mask]

                classes_map[valid_obj_indices] = classes[indices[valid_mask]]
            else:
                classes_map[obj_mask] = 0

        sigma_heatmap = 0.01
        radius = 0.03
        cloud_masked = cloud_sampled
        tree = cKDTree(cloud_masked)
        heatmap_scores = np.zeros(len(cloud_masked), dtype=np.float32)
        for i in range(len(cloud_masked)):
            indices = tree.query_ball_point(cloud_masked[i], r=radius)
            if len(indices) > 0:
                distances = np.linalg.norm(cloud_masked[indices] - cloud_masked[i], axis=1)
                gaussian_weights = np.exp(-distances ** 2 / (2 * sigma_heatmap ** 2))
                total_weight = np.sum(gaussian_weights)
                heatmap_scores[i] = np.sum(graspability_scores[indices] * gaussian_weights) / total_weight

        if np.max(heatmap_scores) > 0:
            heatmap_scores = heatmap_scores / (np.max(heatmap_scores) + 1e-8)
        import open3d as o3d
        import matplotlib.pyplot as plt
        visual_heatmap = False
        if visual_heatmap:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud_masked)

            cmap = plt.get_cmap("viridis")
            colors = np.full((len(cloud_masked), 3), [0.5, 0.5, 0.5], dtype=np.float32)  # Xám mặc định
            non_zero_mask = heatmap_scores != 0
            colors = cmap(heatmap_scores)[:, :3]  # Áp dụng viridis
            pcd.colors = o3d.utility.Vector3dVector(colors)

            o3d.visualization.draw_geometries(
                [pcd],
                window_name="Cloud Masked: Viridis (Gaussian Heatmap Scores), Gray (Zero Scores)",
                width=800,
                height=600
            )

        ret_dict = {}
        if self.augment:
            cloud_sampled, normal_sampled, object_poses_list, aug_trans = self.augment_data(cloud_sampled,
                                                                                            normal_sampled,
                                                                                            object_poses_list)
        import matplotlib.pyplot as plt
        import open3d as o3d

        visual_heatmap = True
        visual_classes = True

        if visual_heatmap:
            pcd_heatmap = o3d.geometry.PointCloud()
            pcd_heatmap.points = o3d.utility.Vector3dVector(cloud_sampled)

            cmap = plt.get_cmap("viridis")
            heat_colors = cmap(heatmap_scores)[:, :3]  # Áp dụng viridis
            pcd_heatmap.colors = o3d.utility.Vector3dVector(heat_colors)

            o3d.visualization.draw_geometries(
                [pcd_heatmap],
                window_name="Grasp Heatmap",
                width=800,
                height=600
            )

        if visual_classes:
            pcd_classes = o3d.geometry.PointCloud()
            pcd_classes.points = o3d.utility.Vector3dVector(cloud_sampled)

            unique_classes = np.unique(classes_map)
            rng = np.random.default_rng(42)
            colors_map = {cls: rng.random(3) for cls in unique_classes}
            class_colors = np.array([colors_map[c] for c in classes_map], dtype=np.float32)
            pcd_classes.colors = o3d.utility.Vector3dVector(class_colors)

            o3d.visualization.draw_geometries(
                [pcd_classes],
                window_name="Classes Map",
                width=800,
                height=600
            )

        ret_dict['heatmap_scores'] = heatmap_scores.astype(np.float32)
        ret_dict['classes'] = classes_map.astype(np.int32)
        return ret_dict

    def _load_grasp_label_for_object(self, obj_idx):
        try:
            obj_name = obj_idx - 1  # Chuyển đổi index
            label_path = os.path.join(self.root, 'grasp_label', '{}_labels.npz'.format(str(obj_name).zfill(3)))
            tolerance_path = os.path.join(BASE_DIR, 'tolerance', '{}_tolerance.npy'.format(str(obj_name).zfill(3)))
            classes_path = os.path.join(self.root, 'output', '{}_classes.npy'.format(str(obj_name).zfill(3)))

            if os.path.exists(label_path) and os.path.exists(tolerance_path):
                label = np.load(label_path)
                tolerance = np.load(tolerance_path)
                classes = np.load(classes_path)

                self.grasp_labels[obj_idx] = (
                    label['points'].astype(np.float32),
                    label['offsets'].astype(np.float32),
                    label['scores'].astype(np.float32),
                    classes.astype(np.float32),
                    tolerance
                )
            else:
                print(f"Warning: Missing grasp label files for object {obj_idx}")
                self.grasp_labels[obj_idx] = (
                    np.empty((0, 3), dtype=np.float32),
                    np.empty((0, 3), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                    np.empty((0,), dtype=np.float32)
                )
        except Exception as e:
            print(f"Error loading grasp label for object {obj_idx}: {e}")
            self.grasp_labels[obj_idx] = (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32)
            )


def load_grasp_labels(root):

    obj_names = list(range(88))
    valid_obj_idxs = []

    grasp_labels = {}

    for i, obj_name in enumerate(tqdm(obj_names, desc='Valid object indices...')):
        if i == 18:
            continue
        valid_obj_idxs.append(i + 1)

    return valid_obj_idxs, grasp_labels



def minkowski_collate_fn(list_data):
    coordinates_batch, features_batch = ME.utils.sparse_collate([d["coors"] for d in list_data],
                                                                [d["feats"] for d in list_data])
    coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch.float(), features_batch.float(), return_index=True, return_inverse=True)
    res = {
        "coors": coordinates_batch,
        "feats": features_batch,
        "quantize2original": quantize2original
    }

    def collate_fn_(batch):
        if type(batch[0]).__module__ == 'numpy':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        elif isinstance(batch[0], container_abcs.Sequence):
            return [[torch.from_numpy(sample) for sample in b] for b in batch]
        elif isinstance(batch[0], container_abcs.Mapping):
            for key in batch[0]:
                if key == 'coors' or key == 'feats':
                    continue
                res[key] = collate_fn_([d[key] for d in batch])
            return res

    res = collate_fn_(list_data)
    return res


def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]

    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


if __name__ == "__main__":
    dataset_root = "/notebooks/data"
    output_dir = os.path.join(dataset_root, "heatmap")
    os.makedirs(output_dir, exist_ok=True)

    valid_obj_idxs, grasp_labels = load_grasp_labels(dataset_root)
    TRAIN_DATASET = GraspNetDataset_fusion(dataset_root, valid_obj_idxs, grasp_labels, camera='realsense',
                                           split='train',
                                           num_points=50000, remove_outlier=True, augment=True)

    # Iterate through the dataset and save ret_dict for each scene
    for idx in tqdm(range(len(TRAIN_DATASET))):
        ret_dict = TRAIN_DATASET[idx]
        scene_name = TRAIN_DATASET.scene_list()[idx]
        output_path = os.path.join(output_dir, f"{scene_name}_ret_dict.npy")
        np.save(output_path, ret_dict, allow_pickle=True)
        print(f"Saved ret_dict for {scene_name} to {output_path}")