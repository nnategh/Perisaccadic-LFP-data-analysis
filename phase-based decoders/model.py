import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from skorch.callbacks import Callback

# Import from config
from config import LFP_SEGMENT_END_MS, LFP_SEGMENT_START_MS, SPIKE_SEGMENT_END_MS, SPIKE_SEGMENT_START_MS

# Determine the time dimension dynamically based on config
# Assuming LFP and Spike segments are made to be the same length for concatenation
TIME_DIM = LFP_SEGMENT_END_MS - LFP_SEGMENT_START_MS # e.g., 1300 - 1000 = 300
# Or if Spike data is the reference for time dimension after concatenation:
# TIME_DIM = SPIKE_SEGMENT_END_MS - SPIKE_SEGMENT_START_MS # e.g., 300 - 0 = 300


class BranchModule(nn.Module):
    """
    A single branch of the MultiBranchEEGNet, processing one channel's data.
    Input to this module is (batch, 1, 2, time_dim),
    where 2 represents (delta_phase, spike_data).
    """
    def __init__(self, time_dim=TIME_DIM): # Pass time_dim if it varies
        super(BranchModule, self).__init__()
        # Temporal convolution: kernel covers 64ms of input
        # Output length: time_dim - 64 + 1
        self.temporal_out_len = time_dim - 64 + 1
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=[1, 64], padding=0),
            nn.BatchNorm2d(32),
            nn.ELU(True),
            nn.Dropout(0.5)
        )
        # Depthwise convolution (acts on the '2' dimension from phase/spike)
        self.depthwise = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=[2, 1], padding=0),
            nn.BatchNorm2d(64),
            nn.ELU(True),
            nn.Dropout(0.5)
        )
        # Average pooling along the temporal dimension
        # Input to AvgPool2d: (batch, 64, 1, temporal_out_len)
        # Output of AvgPool2d: (batch, 64, 1, temporal_out_len // 2)
        self.avgPool_out_len = self.temporal_out_len // 2
        self.avgPool = nn.AvgPool2d([1, 2], stride=[1, 2], padding=0)

    def forward(self, x):  # x shape: (batch, 1, 2, time_dim)
        out = self.temporal(x)  # Expected out: (batch, 32, 2, temporal_out_len)
        out = self.depthwise(out)  # Expected out: (batch, 64, 1, temporal_out_len)
        out = self.avgPool(out)  # Expected out: (batch, 64, 1, avgPool_out_len)
        return out


class MultiBranchEEGNet(nn.Module):
    """
    Multi-branch network that processes data from multiple channels (branches),
    fuses them, and makes a final prediction.
    """
    def __init__(self, outputSize, n_branches=16, time_dim=TIME_DIM):
        super(MultiBranchEEGNet, self).__init__()
        self.n_branches = n_branches
        self.branches = nn.ModuleList([BranchModule(time_dim=time_dim) for _ in range(n_branches)])

        # The avgPool_out_len from BranchModule becomes the W_branch for fusion
        branch_output_temporal_dim = (time_dim - 64 + 1) // 2

        # Fusion convolution operates on the stacked branch outputs
        # Input to fusion_conv: (batch, C_branch=64, n_branches, W_branch)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=[n_branches, 1], padding=0), # Kernel spans all branches
            nn.BatchNorm2d(128),
            nn.ELU(True),
            nn.Dropout(0.5)
        )
        # AdaptiveAvgPool2d ensures a consistent size before the linear layer
        # It will pool the W_branch dimension to 2.
        # Input: (batch, 128, 1, W_branch)
        # Output: (batch, 128, 1, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 2))

        # Linear layer input features: 128 * 1 * 2 = 256
        self.linear = nn.Linear(128 * 1 * 2, outputSize)

    def forward(self, x):  # x shape: (batch, n_branches, 2, time_dim)
        branch_outs = []
        for i, branch_model in enumerate(self.branches):
            # Each branch processes (batch, 1, 2, time_dim)
            branch_input = x[:, i:i + 1, :, :]
            branch_output = branch_model(branch_input)  # Shape: (batch, 64, 1, W_branch)
            branch_outs.append(branch_output)

        # Concatenate branch outputs along a dimension that fusion_conv will treat as "height" (n_branches)
        # Current branch_output: (batch, C_branch=64, H_branch=1, W_branch)
        # We stack them to be (batch, C_branch=64, n_branches, W_branch) for Conv2d
        out = torch.cat(branch_outs, dim=2)  # Concatenate along H_branch, effectively stacking them

        out = self.fusion_conv(out)  # Expected: (batch, 128, 1, W_branch)
        out = self.global_avg_pool(out)  # Expected: (batch, 128, 1, 2)
        out = out.view(out.size(0), -1)  # Flatten: (batch, 256)
        out = self.linear(out)
        return out


class TensorBoardCallback(Callback):
    """
    Skorch callback to log training and validation losses to TensorBoard.
    """
    def __init__(self, log_dir="runs/skorch"):
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir # Store log_dir for reference

    def on_epoch_end(self, net, **kwargs):
        epoch = net.history[-1, 'epoch']
        train_loss = net.history[-1, 'train_loss']
        valid_loss = net.history[-1, 'valid_loss']
        self.writer.add_scalar("Loss/Train", train_loss, epoch)
        self.writer.add_scalar("Loss/Validation", valid_loss, epoch)
        # Log learning rate
        if 'lr' in net.history[-1]:
            lr = net.history[-1, 'lr']
            self.writer.add_scalar("LearningRate", lr, epoch)


    def on_train_end(self, net, **kwargs):
        print(f"TensorBoard logs saved to: {self.log_dir}")
        self.writer.close()