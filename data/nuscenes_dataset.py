import json
import os

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from dataclasses import dataclass, field

from data.base_dataset import get_lidar_rays, BaseDataset


@dataclass
class nuScenesDataset(BaseDataset):
    device: str = "cpu"
    split: str = "train"  # train, val, test, (refine)
    root_path: str = "data/nuScenes"
    sequence_id: str = "450"
    preload: bool = True  # preload data into GPU
    scale: float = 1      # scale to bounding box
    offset: list = field(default_factory=list)  # offset
    fp16: bool = True     # if preload, load into fp16.
    patch_size_lidar: int = 1  # size of the image to extract from the Lidar.
    num_rays_lidar: int = 4096
    fov_lidar: list = field(default_factory=list)  # fov_up, fov [2.0, 26.9]

    def __post_init__(self):
        if self.sequence_id == "450":
            print("Using sequence 450-500")
            frame_start = 450
            frame_end = 500
        elif self.sequence_id == "1250":
            print("Using sequence 1250-1300")
            frame_start = 1250
            frame_end = 1300
        elif self.sequence_id == "1600":
            print("Using sequence 1600-1650")
            frame_start = 1600
            frame_end = 1650
        elif self.sequence_id == "2200":
            print("Using sequence 2200-2250")
            frame_start = 2200
            frame_end = 2250
        elif self.sequence_id == "3180":
            print("Using sequence 3180-3230")
            frame_start = 3180
            frame_end = 3230
        else:
            raise ValueError(f"Invalid sequence id: {self.sequence_id}")
        
        print(f"Using sequence {frame_start}-{frame_end}")
        self.frame_start = frame_start
        self.frame_end = frame_end

        self.training = self.split in ["train", "all", "trainval"]
        self.num_rays_lidar = self.num_rays_lidar if self.training else -1
        if self.split == 'refine':
            self.split = 'train'
            self.num_rays_lidar = -1

        # load nerf-compatible format data.
        with open(
            os.path.join(self.root_path, 
                         f"transforms_{self.sequence_id}_{self.split}.json"),
            "r",
        ) as f:
            transform = json.load(f)

        # load image size
        if "h" in transform and "w" in transform:
            self.H = int(transform["h"])
            self.W = int(transform["w"])
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None

        if "h_lidar" in transform and "w_lidar" in transform:
            self.H_lidar = int(transform["h_lidar"])
            self.W_lidar = int(transform["w_lidar"])

        # read images
        frames = transform["frames"]
        frames = sorted(frames, key=lambda d: d['lidar_file_path'])

        self.poses_lidar = []
        self.images_lidar = []
        self.times = []
        for f in tqdm.tqdm(frames, desc=f"Loading {self.split} data"):
            pose_lidar = np.array(f["lidar2world"], dtype=np.float32)  # [4, 4]

            f_lidar_path = os.path.join(self.root_path, f["lidar_file_path"])

            # channel1 None, channel2 intensity , channel3 depth
            pc = np.load(f_lidar_path)
            ray_drop = np.where(pc.reshape(-1, 3)[:, 2] == 0.0, 0.0, 1.0).reshape(
                self.H_lidar, self.W_lidar, 1
            )

            image_lidar = np.concatenate(
                [ray_drop, pc[:, :, 1, None], pc[:, :, 2, None] * self.scale],
                axis=-1,
            )

            time = np.asarray((f['frame_id']-(frame_start-frame_start))/((frame_end-frame_start)-(frame_start-frame_start)))
            
            self.poses_lidar.append(pose_lidar)
            self.images_lidar.append(image_lidar)
            self.times.append(time)

        self.poses_lidar = np.stack(self.poses_lidar, axis=0)
        self.poses_lidar[:, :3, -1] = (
            self.poses_lidar[:, :3, -1] - self.offset
        ) * self.scale
        self.poses_lidar = torch.from_numpy(self.poses_lidar)  # [N, 4, 4]

        self.images_lidar = torch.from_numpy(np.stack(self.images_lidar, axis=0)).float()  # [N, H, W, C]

        self.times = torch.from_numpy(np.asarray(self.times, dtype=np.float32)).view(-1, 1) # [N, 1]

        if self.preload:
            self.poses_lidar = self.poses_lidar.to(self.device)
            if self.fp16:
                dtype = torch.half
            else:
                dtype = torch.float
            self.images_lidar = self.images_lidar.to(dtype).to(self.device)
            self.times = self.times.to(self.device)

        self.intrinsics_lidar = self.fov_lidar

    def collate(self, index):
        B = len(index)  # a list of length 1

        results = {}

        poses_lidar = self.poses_lidar[index].to(self.device)  # [B, 4, 4]
        rays_lidar = get_lidar_rays(
            poses_lidar,
            self.intrinsics_lidar,
            self.H_lidar,
            self.W_lidar,
            self.num_rays_lidar,
            self.patch_size_lidar,
        )
        time_lidar = self.times[index].to(self.device) # [B, 1]

        images_lidar = self.images_lidar[index].to(self.device)  # [B, H, W, 3]
        if self.training:
            C = images_lidar.shape[-1]
            images_lidar = torch.gather(
                images_lidar.view(B, -1, C),
                1,
                torch.stack(C * [rays_lidar["inds"]], -1),
            )  # [B, N, 3]

        results.update(
            {
                "H_lidar": self.H_lidar,
                "W_lidar": self.W_lidar,
                "rays_o_lidar": rays_lidar["rays_o"],
                "rays_d_lidar": rays_lidar["rays_d"],
                "images_lidar": images_lidar,
                "time": time_lidar,
                "poses_lidar": poses_lidar,
            }
        )

        return results

    def dataloader(self):
        size = len(self.poses_lidar)
        loader = DataLoader(
            list(range(size)),
            batch_size=1,
            collate_fn=self.collate,
            shuffle=self.training,
            num_workers=0,
        )
        loader._data = self
        loader.has_gt = self.images_lidar is not None
        return loader

    def __len__(self):
        """
        Returns # of frames in this dataset.
        """
        num_frames = len(self.poses_lidar)
        return num_frames
