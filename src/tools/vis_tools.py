import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import torch
from tools.tools import tensor_to_np_img, np_img_to_tensor
import torchvision
import os


def plot_flow_circle(size=200, normalize=False, save_path=None):
    radius = size // 2

    grid_y, grid_x = torch.meshgrid([torch.arange(size), torch.arange(size)])

    grid = torch.stack([grid_x, grid_y]).float() - radius
    grid_norm_sq = grid[0] * grid[0] + grid[1] * grid[1]
    grid_mask = grid_norm_sq <= radius ** 2

    if normalize:
        grid = grid / grid_norm_sq.sqrt()[None]
    else:
        grid = grid / radius

    grid[~grid_mask.expand_as(grid)] = 0

    flow_vis = flow_to_vis(grid)
    flow_vis[~grid_mask.expand_as(flow_vis)] = 1

    plt.imshow(flow_vis.permute(1, 2, 0).detach().cpu().numpy())

    if save_path is not None:
        torchvision.utils.save_image(flow_vis, os.path.expanduser(save_path))


def plot_tensor_grid(tensors, titles=None, padding=10, nrow=8, figsize=None, save_path=None):
    from trainer.train_logger import TrainLogger

    grid = TrainLogger._make_img_grid(tensors, titles=titles, padding=padding, nrow=nrow)

    if figsize is not None:
        plt.figure(figsize=figsize)

    plt.imshow(grid.permute(1, 2, 0))

    if save_path is not None:
        torchvision.utils.save_image(grid, os.path.expanduser(save_path))


def flow_to_vis(flow):
    is_torch_tensor = isinstance(flow, torch.Tensor)

    if is_torch_tensor and len(flow.shape) == 4:
        return torch.stack([flow_to_vis(flow[i]) for i in range(flow.shape[0])])

    if is_torch_tensor:
        device = flow.device
        flow = tensor_to_np_img(flow)

    assert flow.shape[2] == 2

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if is_torch_tensor:
        rgb = np_img_to_tensor(rgb).float() / 0xff
        rgb = rgb.to(device)

    return rgb


def visualize_grid(tensor, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = tensor.shape
    grid_size = int(np.ceil(np.sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = tensor[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid


def visualize_tensors(tensors, titles=None, row_limit=None, random_order=False, column_order=False, cmap="viridis"):
    if column_order:
        tensors = list(list(zip(*tensors)))

    num_cols = len(tensors[0])
    num_rows = len(tensors)

    row_range = list(range(num_rows))

    if random_order:
        random.shuffle(row_range)

    if row_limit is None:
        row_limit = num_rows

    row_range = row_range[:row_limit]

    for row_i in range(row_limit):
        for col in range(num_cols):
            row = row_range[row_i]

            plt.subplot(row_limit, num_cols, col + row_i * num_cols + 1)
            tensor = tensors[row][col]
            num_channels = tensor.size(0)

            img = tensor.permute(1, 2, 0) if num_channels > 1 else tensor[0]
            img = img.detach().cpu()

            plt.imshow(img, cmap=None if num_channels > 1 else cmap)
            plt.axis('off')

            if titles is not None and row_i == 0:
                plt.title(titles[col])


