import torch
from torch.utils.data import Dataset
class ControlDataset(Dataset):
    def __init__(self, images, costmaps, odometries, cmd_vels):
        self.images = images
        self.costmaps = costmaps
        self.odometries = odometries
        self.cmd_vels = cmd_vels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        costmap = torch.tensor(self.costmaps[idx], dtype=torch.float32)
        odometry = torch.tensor(self.odometries[idx], dtype=torch.float32)
        cmd_vel = torch.tensor(self.cmd_vels[idx], dtype=torch.float32)
        return image, costmap, odometry, cmd_vel
