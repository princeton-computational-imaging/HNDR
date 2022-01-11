import argparse
import numpy as np
import struct
from matplotlib import gridspec
import matplotlib.pyplot as plt
from glob import glob
import os
from os.path import join
from natsort import natsorted
from skimage.transform import resize
import re
from tqdm import tqdm

""" Code to process depth/image/pose binaries the ios DepthBundleRecorder app into more useable .npz files.
    Usage: python ConvertBinaries.py -d data_folder_with_binaries
    Output: a folder data_processed_folder_with_binaries containing the processed depth bundles
"""

def read_header(header):
    h = re.sub("\[|\]|\(|\)|\s|\'", "", str(header)) # Strip all delims but <> and commas
    h = h.split("<ENDHEADER>")[0] # Snip empty end of header

    timestamp = float(h.split("Time:")[1].split(",")[0])

    euler_angles = np.array(h.split("EulerAngles:SIMD3<Float>")[1].split(",")[0:3], dtype=np.float32)
    world_pose = np.array(h.split("WorldPose:simd_float4x4")[1].split(",")[0:16], dtype=np.float32).reshape((4,4))
    intrinsics = np.array(h.split("Intrinsics:Optionalsimd_float3x3")[1].split(",")[0:9], dtype=np.float32).reshape((3,3))
    world_to_camera = np.array(h.split("WorldToCamera:Optionalsimd_float4x4")[1].split(",")[0:16], dtype=np.float32).reshape((4,4))
    
    return {'timestamp' : timestamp, 
            'euler_angles' : euler_angles, 
            'world_pose' : world_pose.T,
            'intrinsics' : intrinsics.T,
            'world_to_camera' : world_to_camera.T}

def load_info(info_name): 
    with open(info_name, mode='rb') as file:
        file_content = file.read()
    header = file_content[:1024] # 1024 bit header
    return read_header(header)
 
def load_depth(depth_name):
    with open(depth_name, mode='rb') as file:
        file_content = file.read()
    header = file_content[:1024] # 1024 bit header
    file_content = file_content[1024:]
    file_content = struct.unpack('f'* ((len(file_content)) // 4), file_content)
    depth = np.reshape(file_content, (192,256))
    depth = np.flip(depth.T, 1).astype(np.float32)
    return depth, header

def load_conf(conf_name):
    with open(conf_name, mode='rb') as file:
        file_content = file.read()
    file_content = struct.unpack('B'* ((len(file_content))), file_content)
    conf = np.reshape(file_content, (192,256))
    conf = np.flip(conf.T, 1).astype(np.uint8)
    return conf

def load_img(img_name):
    with open(img_name, mode='rb') as file:
        file_content = file.read()
    
    Y = file_content[:1920*1440]
    Y = struct.unpack('B' * ((len(Y))), Y)
    Y = np.reshape(Y, (1440,1920))
    Y = np.flip(Y.T, 1)

    UV = file_content[1920*1440:]
    UV = struct.unpack('B' * ((len(UV))), UV)
    U,V = UV[0::2], UV[1::2]
    U,V = np.reshape(U, (720,960)), np.reshape(V, (720,960))
    U,V = np.flip(U.T, 1), np.flip(V.T, 1)
    # Re-Center U,V channels
    Y,U,V = Y.astype(np.float32), (U.astype(np.float32) - 128), (V.astype(np.float32) - 128)
    U,V = resize(U, (1920,1440), order=0), resize(V, (1920,1440), order=0)

    # Convert YUV 420 to RGB
    R = Y + (V*1/0.6350)
    B = Y + (U*1/0.5389)
    G = (Y - 0.2126*R - 0.0722*B)*(1/0.7152)

    img = np.stack((R,G,B), axis=-1)
    img[img<0] = 0
    img[img>255] = 255
    img = img.astype(np.uint8)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default=None, type=str, required=True, help='Data directory')
    args = parser.parse_args()
    
    bundle_names = natsorted(glob(join(args.d, "*")))
    
    for bundle_name in bundle_names:
        print("Processing {0}.".format(bundle_name.split("/")[-1]))
        
        if "-poses" not in bundle_name:
            # Process image + depth bundle
            
            depth_names = natsorted(glob(join(bundle_name, "depth*.bin")))
            img_names = natsorted(glob(join(bundle_name, "image*.bin")))
            conf_names = natsorted(glob(join(bundle_name, "conf*.bin")))

            save_path = bundle_name.replace("data", "data_processed")
            os.makedirs(save_path, exist_ok=True)

            npz_file = {}
            for i, (img_name, depth_name, conf_name) in tqdm(enumerate(zip(img_names, depth_names, conf_names))):
                img = load_img(img_name)
                depth, header = load_depth(depth_name)
                info = read_header(header)
                conf = load_conf(conf_name)

                if i == 0:
                    ref_time = info['timestamp']

                info['timestamp'] -= ref_time

                npz_file["img_{0}".format(i)] = img
                npz_file["depth_{0}".format(i)] = depth
                npz_file["conf_{0}".format(i)] = conf
                npz_file["info_{0}".format(i)] = info

            npz_file["num_frames"] = len(img_names)

            # Save first frame preview
            fig = plt.figure(figsize=(14, 30)) 
            gs = gridspec.GridSpec(1, 3, wspace=0.0, hspace=0.0, width_ratios=[1,1,1.12])
            ax1 = plt.subplot(gs[0,0])
            ax1.imshow(npz_file['img_0'])
            ax1.axis('off')
            ax1.set_title("Image")
            ax2 = plt.subplot(gs[0,1])
            ax2.imshow(npz_file['conf_0'], cmap="gray")
            ax2.axis('off')
            ax2.set_title("Confidence")
            ax3 = plt.subplot(gs[0,2])
            d = ax3.imshow(npz_file['depth_0'], cmap="Spectral", vmin=0, vmax=7)
            ax3.axis('off')
            ax3.set_title("Depth")
            fig.colorbar(d, fraction=0.055, label="Depth [m]")
            plt.savefig(join(save_path, "frame_first.png"), bbox_inches='tight', pad_inches=0.05, facecolor='white')
            plt.close()

            # Save last frame preview
            fig = plt.figure(figsize=(14, 30)) 
            gs = gridspec.GridSpec(1, 3, wspace=0.0, hspace=0.0, width_ratios=[1,1,1.12])
            ax1 = plt.subplot(gs[0,0])
            ax1.imshow(npz_file['img_{0}'.format(len(img_names) - 1)])
            ax1.axis('off')
            ax1.set_title("Image")
            ax2 = plt.subplot(gs[0,1])
            ax2.imshow(npz_file['conf_{0}'.format(len(img_names) - 1)], cmap="gray")
            ax2.axis('off')
            ax2.set_title("Confidence")
            ax3 = plt.subplot(gs[0,2])
            d = ax3.imshow(npz_file['depth_{0}'.format(len(img_names) - 1)], cmap="Spectral", vmin=0, vmax=7)
            ax3.axis('off')
            ax3.set_title("Depth")
            fig.colorbar(d, fraction=0.055, label="Depth [m]")
            plt.savefig(join(save_path, "frame_last.png"), bbox_inches='tight', pad_inches=0.05, facecolor='white')
            plt.close()

            # Save bundle
            np.savez(join(save_path, "frame_bundle"), **npz_file)
        
        else:
            # Process only poses + info bundle
            
            info_names = natsorted(glob(join(bundle_name, "info*.bin")))
            save_path = bundle_name.replace("data", "data_processed")
            os.makedirs(save_path, exist_ok=True)

            npz_file = {}
            for i, info_name in tqdm(enumerate(info_names)):
                info = load_info(info_name)

                if i == 0:
                    ref_time = info['timestamp']

                info['timestamp'] -= ref_time
                npz_file["info_{0}".format(i)] = info

            npz_file["num_frames"] = len(info_names)
            
            # Save bundle
            np.savez(join(save_path, "info_bundle"), **npz_file)
            
if __name__ == '__main__':
    main()