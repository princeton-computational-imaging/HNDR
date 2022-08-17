import argparse
import os
import time
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import BundleRefinementModel
from utils import utils
from utils.dataloader import BundleDataset


def train(args):
    
    print("Args: ", args)
    print("Starting training.")
    
    # Speedup if input is same size
    torch.backends.cudnn.benchmark = True
    
    # Tensorboard
    tensorboard_writer = SummaryWriter(args.checkpoint_path)
    
    bundle_dataset = BundleDataset(args)
    train_loader = DataLoader(dataset=bundle_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=False, drop_last=True)
    
    model = BundleRefinementModel(args, bundle_dataset).to(args.device)
    model.train()
    print("Number of trainable parameters:", utils.count_params(model, trainable=True))
        
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    
    criterion = nn.MSELoss()
    tensorboard_writer.add_text("args", str(args).replace(",","  \n"))
    
    for epoch in range(args.max_epochs):
        
        # Learning rate summary
        lr = optimizer.param_groups[0]["lr"]
        tensorboard_writer.add_scalar("learning_rate", lr, epoch)
        
        last_time = time.time()
        epoch_depth_losses = []
        epoch_photometric_losses = []
        for sample in train_loader:
            
            # Load sample
            qry_img = sample['img'].to(args.device)
            qry_depth = sample['depth'].to(args.device)
            qry_conf = sample['conf'].to(args.device)
            qry_intrinsics = sample['info']['intrinsics'].float().to(args.device)
            qry_idx = sample['frame_idx']
                        
            # Model forward call
            ref_rays, qry_rays, mlp_rays, qry_conf = model(qry_img, qry_depth, qry_conf, qry_intrinsics, qry_idx)
        
            loss  = criterion(ref_rays[:,4:], mlp_rays[:,4:]) # Photometric loss
            loss += args.lidar_weight * criterion(qry_rays[:,2], mlp_rays[:,2]) # LiDAR regularization
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                epoch_depth_losses.append(criterion(qry_rays[:,2], mlp_rays[:,2]).item())
                epoch_photometric_losses.append(criterion(ref_rays[:,4:], mlp_rays[:,4:]).item())
            optimizer.step()
                    
        model.confidence.data = utils.median_blur(model.confidence.data, (args.median_size,args.median_size))
    
        # Tensorboard summary
        if epoch % args.tensorboard_frequency == 0:
            qry_out, mlp_out = model.get_visualization(256,192)
            
            img_summary = []
            if epoch == 0:
                # Add to summary once, don't update
                img_summary.append(("train/depth_lidar", utils.colorize_tensor(model.ref_depth[0,0], vmin=0, cmap="Spectral", colorbar=True)))
                img_summary.append(("train/rgb_reference", model.ref_img[0,:3].permute(1,2,0)))
                
            img_summary.append(("train/rays_rgb_ref", visualize_rays(ref_rays)))
            img_summary.append(("train/rays_rgb_qry", visualize_rays(qry_rays)))
            img_summary.append(("train/rays_rgb_mlp", visualize_rays(mlp_rays)))
            img_summary.append(("train/rays_z_ref", visualize_rays_z(ref_rays, vmax=qry_out.max())))
            img_summary.append(("train/rays_z_qry", visualize_rays_z(qry_rays, vmax=qry_out.max())))
            img_summary.append(("train/rays_z_mlp", visualize_rays_z(mlp_rays, vmax=qry_out.max())))
            img_summary.append(("train/depth_mlp", utils.colorize_tensor(mlp_out, cmap="Spectral", vmin=0, vmax=qry_out.max(), colorbar=True)))
            img_summary.append(("train/depth_mlp_difference", utils.colorize_tensor(torch.abs(mlp_out - qry_out), cmap="hot", vmin=0, colorbar=True)))
            img_summary.append(("train/confidence", utils.colorize_tensor(F.relu(model.confidence[0,0]), cmap="magma", colorbar=True)))
            utils.save_images(tensorboard_writer, img_summary, epoch)
            
        if epoch % args.save_frequency == 0:
            torch.save(model, join(args.checkpoint_path, "model.pt"))
        
        print("Epoch: [{0:0>3d}/{1:0>3d}] Time per Epoch: {2:.2f}s Depth Loss: {3:.4f} Photo Loss: {4:.4f}".format(epoch, args.max_epochs, time.time() - last_time,
                                                                                                                   np.mean(epoch_depth_losses), np.mean(epoch_photometric_losses)))
        print("Ref mean:", qry_rays[:,2].mean(), "Qry mean:", qry_rays[:,2].mean(), "MLP mean:", mlp_rays[:,2].mean())
        tensorboard_writer.add_scalar("train/depth_loss", np.mean(epoch_depth_losses), epoch)
        tensorboard_writer.add_scalar("train/photo_loss", np.mean(epoch_photometric_losses), epoch)
        
        lr_scheduler.step()
        
    print("Training complete.")
    
def visualize_rays_z(rays, vmax):
    """ Collect ray depth points into an array for visualization """
    z = rays[0,2,:]
    N = int(np.sqrt(z.shape[-1]))
    z = z.reshape(N,N)
    return utils.colorize_tensor(z, cmap="Spectral", vmin=0, vmax=vmax, colorbar=True)
    
def visualize_rays(rays):
    """ Collect ray color points into an array for visualization """
    rgb = rays[0,4:7,:]
    N = int(np.sqrt(rgb.shape[-1]))
    rgb = rgb.reshape(3,N,N).permute(1,2,0)
    return rgb                            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data specific
    parser.add_argument('--bundle_path', type=str, required=True, help="Path to bundle.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to save checkpoints and final refined depth.")
    parser.add_argument('--height', type=int, default=1920, help="Image height.")
    parser.add_argument('--width', type=int, default=1440, help="Image width.")
    parser.add_argument('--reference_idx', type=int, default=0, help="Which index is the reference frame.")
                    
    # Training
    parser.add_argument('--tensorboard_frequency', type=int, default=1, help="How many epochs between tensorboard image summaries (1 = every epoch).")
    parser.add_argument('--save_frequency', type=int, default=10, help="How many epochs between saving the model (1 = every epoch).")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument('--max_epochs', type=int, default=200, help="Max training epochs.")
    parser.add_argument('--batch_size', type=int, default=1, help="Training batch size.")
    parser.add_argument('--device', type=str, default="cpu", help="Training device [cuda, cpu].")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate.")
    parser.add_argument('--gamma', type=float, default=0.985, help="Exponential decay rate.")

                    
    # Model / loss
    parser.add_argument('--num_rays', type=int, default=4096, help="Number of points/rays to sample per batch.") 
    parser.add_argument('--num_encoding_functions', type=int, default=6, help="Dimension of positional encoding.")
    parser.add_argument('--num_hidden_layers', type=int, default=4, help="Number of hidden layers in MLP.")
    parser.add_argument('--hidden_features', type=int, default=256, help="Feature size for MLP.")
    parser.add_argument('--lidar_weight', type=float, default=0.01, help="How much to weight lidar loss")
    parser.add_argument('--patch_size', type=int, default=11, help="What size RGB patch to sample for each ray.")
    parser.add_argument('--median_size', type=int, default=11, help="What size median filter to apply to confidence.")
    parser.add_argument('--coord_patch_size', type=int, default=11, help="What size of patch to use to generate rays.")
    parser.add_argument('--frames', type=int, default=None, nargs='+', help="Which specific frames to train on.")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    if not os.path.isdir(args.checkpoint_path):
        os.mkdir(args.checkpoint_path)

    train(args)
