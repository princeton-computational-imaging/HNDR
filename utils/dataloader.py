import copy

import numpy as np
import torch
from skimage.transform import resize
from torch.utils.data import Dataset


def eye_like(tensor):
    return torch.ones_like(tensor) * torch.eye(tensor.shape[-1]).to(tensor.device)

class BundleDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        # Calling dict() loads the whole npz into memory
        self.args = args
        self.bundle = dict(np.load(args.bundle_path, allow_pickle=True))
        
        if args.frames is None:
            self.num_frames = self.bundle['num_frames'].item()
            self.frames = np.arange(0,self.num_frames)
        else:
            self.num_frames = len(args.frames)
            self.frames = args.frames
        
        self.reference_idx = args.reference_idx
        self.sample_ref = dict()
        ref_info = self.bundle['info_{0}'.format(args.reference_idx)].item() # .npz returns dicts as array(dict)
        
        ref_img = self.bundle['img_{0}'.format(args.reference_idx)].astype(np.float32)/255
        ref_img = resize(ref_img, (args.height, args.width), order=1)
        ref_img = np.moveaxis(ref_img, 2, 0)
        
        ref_depth = self.bundle['depth_{0}'.format(args.reference_idx)].astype(np.float32)
        ref_depth = resize(ref_depth, (args.height, args.width), order=1)
        ref_depth = ref_depth[None,:,:] # Insert dummy channel
        
        ref_conf = self.bundle['conf_{0}'.format(args.reference_idx)].astype(np.float32)
        ref_conf = resize(ref_conf, (args.height, args.width), order=1)
        ref_conf = ref_conf[None,:,:] # Insert dummy channel
        
        # Update intrinsics to match new image size
        assert (1920 // args.height) == (1440 // args.width) # No messing with ratios
        self.ref_intrinsics = torch.from_numpy(ref_info['intrinsics'])[None].float().clone()
        self.ref_intrinsics[:,:2,:3] /= (1920.0 // args.height)
        
        # Convert reference data to tensors of appropriate size
        self.ref_world_to_camera = torch.from_numpy(ref_info['world_to_camera'])[None].float()
        self.ref_camera_to_world = torch.inverse(self.ref_world_to_camera).float()
        self.ref_depth = torch.from_numpy(ref_depth)[None] # To tensor with batch size = 1
        self.ref_img = torch.from_numpy(ref_img)[None]
        self.ref_conf = torch.from_numpy(ref_conf)[None]
        
        # Pre-compute forward transforms
        self.forward_transforms = []
        self.world_to_cameras = []
        
        for idx in range(self.bundle['num_frames'].item()): 
            world_to_camera = torch.from_numpy(self.bundle['info_{0}'.format(idx)].item()['world_to_camera'])[None].float()
            self.world_to_cameras.append(world_to_camera)
            
            if idx == args.reference_idx:
                self.forward_transforms.append(eye_like(self.ref_camera_to_world))
            else:
                qry_world_to_camera = torch.from_numpy(self.bundle['info_{0}'.format(idx)].item()['world_to_camera'])[None].float()
                forward_transform = qry_world_to_camera @ self.ref_camera_to_world
                self.forward_transforms.append(forward_transform)

        self.world_to_cameras = torch.cat(self.world_to_cameras, dim=0)
        self.forward_transforms = torch.cat(self.forward_transforms, dim=0)
        
    def __len__(self):
        return self.num_frames - 1
        
        # Repeat above for each query (qry) frame
    def __getitem__(self, idx):
        idx = idx + 1 # Never return reference frame, avoid division by zero
        idx = self.frames[idx] # Grab subset frame idx
        args = self.args
        
        sample = dict()
        sample['frame_idx'] = idx
        sample['info'] = copy.deepcopy(self.bundle['info_{0}'.format(idx)].item())  # Avoid mutation, .npz files are namespaces, tricky

        # Update intrinsics to match new image size
        sample['info']['intrinsics'] = sample['info']['intrinsics']
        sample['info']['intrinsics'][:2,:3] /= (1920.0 // args.height)
        
        img = self.bundle['img_{0}'.format(idx)].astype(np.float32)/255
        img = resize(img, (args.height, args.width), order=1)
        img = np.moveaxis(img, 2, 0)
        
        depth = self.bundle['depth_{0}'.format(idx)].astype(np.float32)
        depth = resize(depth, (args.height, args.width), order=1)
        depth = depth[None,:,:] # Insert dummy channel
        
        conf = self.bundle['conf_{0}'.format(idx)].astype(np.float32)
        conf = resize(conf, (args.height, args.width), order=1)
        conf = conf[None,:,:] # Insert dummy channel
        
        sample['img'] = img
        sample['depth'] = depth
        sample['conf'] = conf
        return sample
