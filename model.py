import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.neural_blocks as blocks
from utils.utils import *


class BundleRefinementModel(nn.Module):
    def __init__(self, args, bundle_dataset):
        super().__init__()
        
        self.args = args

        self.world_to_cameras = bundle_dataset.world_to_cameras.to(args.device)        
        self.ref_camera_to_world = bundle_dataset.ref_camera_to_world.to(args.device)
        self.ref_intrinsics = bundle_dataset.ref_intrinsics.to(args.device)
        self.ref_depth  = bundle_dataset.ref_depth.to(args.device)
        self.ref_img = bundle_dataset.ref_img.to(args.device)
        
        self.confidence = nn.Parameter(data=get_initial_conf(self.ref_img), requires_grad=True).float()
        self.gaussian_weights = torch.tensor(gkern(args.patch_size, args.patch_size/5).flatten(), device=args.device)[None,:,None] # [1,N,1]
        self.gaussian_rgb_weights = self.gaussian_weights.repeat(1,3,1) # [1,3N,1]
        self.net = ImplicitDepthNet(args)
        
        
    def get_visualization(self, y_samples=128, x_samples=96, coords=None):
        args = self.args
        
        if coords is None:
            y,x = torch.meshgrid(torch.linspace(0, args.height-1, y_samples),
                                 torch.linspace(0, args.width-1,  x_samples))
            coords = torch.stack((x.flatten(),y.flatten()), dim=0).unsqueeze(0).to(args.device) # [1,2,N]
        else:
            coords = coords.to(args.device) # Use custom coords
        
        ref_rays, qry_rays, mlp_rays, qry_conf = self.forward(self.ref_img, self.ref_depth, torch.ones_like(self.ref_depth), 
                                                              self.ref_intrinsics, [0], coords=coords, ref=True)
        qry_out = qry_rays[:,2].reshape(y_samples,x_samples)
        mlp_out = mlp_rays[:,2].reshape(y_samples,x_samples)
        return qry_out, mlp_out
        

    def forward(self, qry_img, qry_depth, qry_conf, qry_intrinsics, qry_idx, coords=None, ref=False):
        if coords is None:
            coords = get_random_coords(self.args)
        
        qry_world_to_camera = self.world_to_cameras[qry_idx]
        forward_transform = qry_world_to_camera @ self.ref_camera_to_world # ref xyz -> qry xyz
        backward_transform = torch.inverse(forward_transform) # qry xyz -> ref xyz
        
        if ref: # Reference view, identity transform
            forward_transform = backward_transform = eye_like(self.world_to_cameras[qry_idx])

        with torch.no_grad():
            qry_rays = to_rays(qry_depth, qry_img, coords, grid_sample=True, patch_size=self.args.patch_size)

        # Convert to meters
        qry_rays_m = convert_px_rays_to_m(qry_rays, qry_intrinsics)
        
        ref_from_qry_rays = transform_rays(qry_rays_m, backward_transform)
        ref_from_qry_rays = convert_m_rays_to_px(ref_from_qry_rays, self.ref_intrinsics)
        ref_from_qry_rays = resample_rgb(ref_from_qry_rays, self.ref_img, patch_size=self.args.patch_size) # Sample from ref image
                
        confidence = patch_grid_sampler(self.confidence, ref_from_qry_rays[:,:2], patch_size=self.args.patch_size)
        confidence = torch.mean(self.gaussian_weights * confidence, dim=1, keepdim=True)
        mlp_rays = self.net(ref_from_qry_rays, confidence) # Query mlp for depth information
        
        mlp_rays = convert_px_rays_to_m(mlp_rays, qry_intrinsics)
        mlp_rays = transform_rays(mlp_rays, forward_transform) # ref -> qry coordinate frame
        mlp_rays = convert_m_rays_to_px(mlp_rays, self.ref_intrinsics)
        mlp_rays = resample_rgb(mlp_rays, qry_img, patch_size=self.args.patch_size) # Sample from qry image
        
        return ref_from_qry_rays, qry_rays, mlp_rays, qry_conf

class ImplicitDepthNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        M = args.num_encoding_functions
        
        self.positional_encoder = blocks.PositionalEncoding(num_encoding_functions=M)
        self.FCNet = blocks.FCBlock(in_features=(2*M + 1)*3 + 3, out_features=1, num_hidden_layers=args.num_hidden_layers, 
                                    hidden_features=args.hidden_features, outermost_linear=True, nonlinearity='relu')
        self.dims = torch.tensor([args.height - 1, args.width - 1], device=args.device)[None,:,None] # [1,2,1]
        
    def forward(self, rays, confidence):
        yx = torch.clone(rays[:,:3])
        yx[:,:2] += 5e-2*torch.randn_like(yx[:,:2])
        yx[:,:2] = (yx[:,:2]/self.dims)  # Normalize to 0 to 1
                
        yx_encoded = self.positional_encoder(yx)
        
        M = self.args.patch_size ** 2
        N = (M - 1) // 2
        mlp_in = torch.cat((yx_encoded, rays[:,[4+N, 4+N+M, 4+N+2*M]]), dim=1) # Select center rgb values
        
        z_out = self.FCNet(mlp_in)
    
        mlp_rays = torch.clone(rays)
        z_out = mlp_rays[:,2:3] + F.relu(confidence) * z_out
        z_out[z_out < 0.05] = 0.05
        mlp_rays[:,2:3] = z_out
         
        return mlp_rays
    
def get_initial_conf(img):
    """ Use blurred image gradient to initialize confidence """
    edge = torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])[None,None].repeat(1,3,1,1).float().to(img.device)
    gaus = torch.tensor(gkern(25,15))[None,None].float().to(img.device)
    conf = torch.abs(F.conv2d(img, edge, padding=0))
    conf = F.max_pool2d(conf, 4)
    conf = F.conv2d(conf, gaus, padding=0)
    conf = F.interpolate(conf, img.shape[-2:])
    conf = 1e-5 + (conf/conf.max())*0.02
    
    return conf

def get_random_coords(args):
    """ Grab a patch of coordinates at a random center position """
    coord_patch_radius = (args.coord_patch_size - 1) // 2
    y = (args.height - coord_patch_radius * 2) * torch.rand(size=(args.batch_size, args.num_rays)) + coord_patch_radius
    x = (args.width -  coord_patch_radius * 2) * torch.rand(size=(args.batch_size, args.num_rays)) + coord_patch_radius
    
    return torch.stack((x,y),axis=1).to(args.device)
                
def eye_like(tensor):
    """ Like ones_like but for the identity matrix. """
    return torch.ones_like(tensor) * torch.eye(tensor.shape[-1]).to(tensor.device)
