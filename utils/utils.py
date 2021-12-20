import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils


def count_params(model, trainable=True):
    """ Count number of parameters in pytorch model (optional: only count trainable params)
    """
    if not trainable:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def colorize_tensor(value, vmin=None, vmax=None, cmap=None, colorbar=False, height=9.6, width=7.2):
    """ Convert tensor to 3 channel RGB array according to colors from cmap
        similar usage as plt.imshow
    """
    assert len(value.shape) == 2 # H x W
    
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(width,height)
    a = ax.imshow(value.detach().cpu(), vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_axis_off()
    if colorbar:
        cbar = plt.colorbar(a, fraction=0.05)
        cbar.ax.tick_params(labelsize=30)
    plt.tight_layout()
    plt.close()
    
    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0
    
    return img

def save_images(tensorboard_writer, img_summary, epoch):
    """ Call tensorboar_write.add_image a couple times for
        a set of 3 channel RGB arrays
    """
    for (tag, img) in img_summary:
        if not torch.is_tensor(img):
            img = torch.from_numpy(img)
            
        assert img.shape[-1] == 3 # 3 channel RGB
        assert len(img.shape) == 3 # H,W,C
        
        tensorboard_writer.add_image(tag, vutils.make_grid(img.permute(2,0,1)[None], padding=0, nrow=1, normalize=False, scale_each=False), epoch)

# concatenate pixel coordinates to depth tensor
def convert_px_rays_to_m(rays, intrinsics):
    """ iPhone poses are right-handed system, +x is right towards power button, +y is up towards front camera, +z is towards user's face
     images are opencv convention right-handed, x to the right, y down, and z into the world (away from face)
    """
    x = rays[:,0:1]
    y = rays[:,1:2]
    z = rays[:,2:3]
    
    # intrinsics are for landscape sensor, top row: y, middle row: x, bottom row: z
    fy, cy, fx, cx = intrinsics[:,0,0], intrinsics[:,0,2], intrinsics[:,1,1], intrinsics[:,1,2]
    fy, cy, fx, cx = fy[:,None,None], cy[:,None,None], fx[:,None,None], cx[:,None,None]
    
    x = (x - cx) * (z/fx)
    y = (y - cy) * (z/fy)

    # rotate around the camera's x-axis by 180 degrees
    # now point cloud is in y up and z towards face convention
    y = -y
    z = -z
                     
    # match pose convention (y,x,z)
    return torch.cat((x,y,z,rays[:,3:]), dim=1) # [B,7,N]

def convert_m_rays_to_px(rays, intrinsics):
    """ Get x,y coordinates in pixels from xyz rays in meters
    """    
    fy, cy, fx, cx = intrinsics[:,0,0], intrinsics[:,0,2], intrinsics[:,1,1], intrinsics[:,1,2]
    fy, cy, fx, cx = fy[:,None,None], cy[:,None,None], fx[:,None,None], cx[:,None,None]
    x, y, z = rays[:,0:1], rays[:,1:2], rays[:,2:3]
    
    # undo rotation from convert_px_rays_to_m
    y = -y
    z = -z
    
    x  = x * (fx/z) + cx
    y =  y * (fy/z) + cy
    
    return torch.cat((x,y,z,rays[:,3:]), dim=1) # [B,7,N]

def coord_sampler(img, coords):
    """ Sample img batch at integer (x,y) coords
        img: [B,C,H,W], coords: [B,2,N]
        returns: [B,C,N] points
    """
    B,C,H,W = img.shape
    N = coords.shape[2]
    
    batch_ref = torch.meshgrid(torch.arange(B), torch.arange(N))[0]
    out = img[batch_ref, :, coords[:,1,:], coords[:,0,:]]
    return out.permute(0,2,1)

def patch_grid_sampler(img, coords, patch_size=3, mode='bilinear'):
    """ Wrapper for grid_sample, uses pixel coordinates 
        Selects patch_size x patch_size set of points around each coordinate and 
        concatenates across channel dimension.
        img: [B,C,H,W], coords: [B,2,N]
        returns: [B,C,N] points
    """
    
    grid = coords.float().permute(0,2,1)[:,:,None,:] # channel last
    
    N = (patch_size - 1) // 2
    offsets_x, offsets_y = torch.meshgrid(torch.arange(-N,N+1), torch.arange(-N,N+1))
    offsets = torch.stack((offsets_y.flatten(), 
                           offsets_x.flatten()), dim=-1)[None,None].to(coords.device) # [1,1,patch_size**2,2]

    grid = grid + offsets # [B,N,patch_size**2,2]

    # grid_sample wants float locations from -1 to 1
    H, W = img.shape[-2:]
    grid[:,:,:,0] = 2*grid[:,:,:,0].float()/(W-1) - 1
    grid[:,:,:,1] = 2*grid[:,:,:,1].float()/(H-1) - 1
    
    img = F.grid_sample(img, grid, mode=mode, align_corners=True).permute(0,1,3,2)
    img = img.reshape(img.shape[0], -1, img.shape[-1])

    return img

def grid_sampler(img, coords, mode='bilinear'):
    """ Wrapper for grid_sample, uses pixel coordinates 
        img: [B,C,H,W], coords: [B,2,N]
        returns: [B,C,N] points
    """
    H, W = img.shape[-2:]
    ugrid, vgrid = coords.split([1,1], dim=1)
    
    # grid_sample wants float locations from -1 to 1
    ugrid = 2*ugrid.float()/(W-1) - 1
    vgrid = 2*vgrid.float()/(H-1) - 1

    grid = torch.cat([ugrid, vgrid], dim=1).permute(0,2,1)[:,:,None,:] # channel last
    img = F.grid_sample(img, grid, mode=mode, align_corners=True).squeeze(-1)

    return img

def to_rays(depth, img, coords, grid_sample=False, patch_size=None):
    """ Given depth + img construct 6D rays [Y,X,Z,1,R,G,B]
        depth: [B,1,H,W], img: [B,3,H,W], coords: [B,2,N]
        returns: [B,6,N] points
    """

    if grid_sample and patch_size is not None:
        z_pts = grid_sampler(depth, coords)
        rgb_pts = patch_grid_sampler(img, coords, patch_size)
        
    elif grid_sample: # no patch
        z_pts = grid_sampler(depth, coords)
        rgb_pts = grid_sampler(img, coords)
    
    else: # direct index
        z_pts = coord_sampler(depth, coords)
        rgb_pts = coord_sampler(img, coords)
    
    ones = torch.ones_like(z_pts) # homogenous coordinates
    
    return torch.cat((coords, z_pts, ones, rgb_pts), dim=1)
    
def transform_rays(rays, transform):
    """ Apply rotation + translation matrix to rays
        rays: [B,7,N], transform: [B,4,4]
        returns: rays [B,7,N]
    """
    
    return torch.cat((torch.bmm(transform, rays[:,:4]), rays[:,4:]), dim=1)
    
def resample_rgb(rays, img, mode='bilinear', patch_size=None):
    """ Use ray xy coordinates to sample into img and update rgb values
        rays: [B,7,N], img: [B,3,H,W]
        returns: rays [B,7,N]
    """
    coords = rays[:,:2]
    if patch_size is None:
        rgb = grid_sampler(img, coords, mode=mode)
    else:
        rgb = patch_grid_sampler(img, coords, mode=mode, patch_size=patch_size)
    
    return torch.cat((rays[:,:4], rgb), dim=1)

def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel


from torch.nn.modules.utils import _pair, _quadruple


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super().__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x
    
    
def get_binary_kernel2d(window_size) -> torch.Tensor:
    r"""Create a binary kernel to extract the patches. If the window size
    is HxW will create a (H*W)xHxW kernel.
    """
    window_range: int = window_size[0] * window_size[1]
    kernel: torch.Tensor = torch.zeros(window_range, window_range)
    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(window_range, 1, window_size[0], window_size[1])
    
def _compute_zero_padding(kernel_size):
    r"""Utility function that computes zero padding tuple."""
    computed: List[int] = [(k - 1) // 2 for k in kernel_size]
    return computed[0], computed[1]
    
def median_blur(input, kernel_size):
    r"""Blur an image using the median filter.

    .. image:: _static/img/median_blur.png

    Args:
        input: the input image with shape :math:`(B,C,H,W)`.
        kernel_size: the blurring kernel size.

    Returns:
        the blurred input tensor with shape :math:`(B,C,H,W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       filtering_operators.html>`__.

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> output = median_blur(input, (3, 3))
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    padding: Tuple[int, int] = _compute_zero_padding(kernel_size)

    # prepare kernel
    kernel: torch.Tensor = get_binary_kernel2d(kernel_size).to(input)
    b, c, h, w = input.shape

    # map the local window to single vector
    features: torch.Tensor = F.conv2d(input.reshape(b * c, 1, h, w), kernel, padding=padding, stride=1)
    features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

    # compute the median along the feature axis
    median: torch.Tensor = torch.median(features, dim=2)[0]

    return median
