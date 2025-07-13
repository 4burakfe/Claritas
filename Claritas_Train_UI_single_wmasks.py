from datetime import datetime
from glob import glob
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from monai.losses import SSIMLoss
from monai.data import pad_list_data_collate
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, SpatialPadd,
    RandFlipd, RandRotate90d, ToTensord, Spacingd, SpatialCropd, RandSpatialCropd, GridPatchd, RandGridPatchd, CenterSpatialCropd, ScaleIntensityRanged, RandFlipd
)
from monai.data import Dataset, DataLoader
from monai.data import ITKReader  
from monai.networks.nets import SwinUNETR, UNet
from monai.inferers import sliding_window_inference
import numpy as np
import os 
import time  
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import threading
import SimpleITK as sitk
from monai import __version__ as monai_version
from packaging import version

# Hello, this code will help you to build your own models for PET denoising process.
# It takes input and target .nrrd files to train.
# The files of input and target datasets should be paired. The program will sort filenames in datasets.
# It is highly useful to have same filenames in different folders for the input and target images to prevent mismatch.
# The output will be logs in window, plots and pth file.
# I advise to save *.nrrd files as uncompressed to reduce burden on CPU and prevent throttling by decompression.
# Written by Burak Demir, MD, FEBNM


#####################################
#####                           #####
#####       LOSS Functions      #####
#####                           #####
#####################################


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.epsilon ** 2)
        return torch.mean(loss)

def asymmetric_l1_loss(pred, target, neg_weight=1.0):
    diff = pred - target
    weights = torch.ones_like(diff)
    weights[diff < 0] = neg_weight  # more penalty for underestimation
    loss = torch.abs(diff) * weights
    return loss.mean()

## For Edge Loss

def gradient(img):
    dz = img[:, :, 1:, :, :] - img[:, :, :-1, :, :]
    dy = img[:, :, :, 1:, :] - img[:, :, :, :-1, :]
    dx = img[:, :, :, :, 1:] - img[:, :, :, :, :-1]

    # Pad to match original size
    dz = F.pad(dz, (0,0,0,0,0,1))
    dy = F.pad(dy, (0,0,0,1,0,0))
    dx = F.pad(dx, (0,1,0,0,0,0))

    return torch.cat([dx, dy, dz], dim=1)

#####################################
#####                           #####
#####     Model Architectures   #####
#####                           #####
#####################################


class GCFN(nn.Module):
    def __init__(self, dim):
        super(GCFN, self).__init__()
        self.norm = nn.LayerNorm(dim)

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc0 = nn.Linear(dim, dim)

        self.conv1 = nn.Conv3d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.conv2 = nn.Conv3d(dim, dim, kernel_size=5, padding=2, groups=dim)

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        x_ = x.permute(0, 2, 3, 4, 1).contiguous().view(B * D * H * W, C)

        x1 = self.fc1(self.norm(x_))
        x2 = self.fc2(self.norm(x_))

        x1 = x1.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        x2 = x2.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        gate = F.gelu(self.conv1(x1)) * self.conv2(x2)

        gate = gate.permute(0, 2, 3, 4, 1).contiguous().view(B * D * H * W, C)
        out = self.fc0(gate).view(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        return out + x

class SwinDenoiser(nn.Module):
    def __init__(self,  in_channels=1, out_channels=1, feature_size=48,heads=(6,12,24,48),depths=(2,3,3,2),do_rate=0.0):
        super(SwinDenoiser, self).__init__()
        self.model = SwinUNETR(
            num_heads = heads,
            use_v2=True,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            depths = depths,
            dropout_path_rate=do_rate,
            **({"img_size": (64, 64, 64)} if version.parse(monai_version) < version.parse("1.5") else {}),
            use_checkpoint=True
        )

    def forward(self, x):
        return self.model(x)

class DenoiseUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1,  channels=(64, 128, 256,512,1024), num_res_units=2,strides=(2, 2, 2, 2),kernel_size=3,up_kernel_size=3):
        super(DenoiseUNet, self).__init__()
        
        # Use MONAI's 3D U-Net as the denoising backbone
        self.unet = UNet(
            strides=strides,
            num_res_units=num_res_units,
            kernel_size=kernel_size,
            up_kernel_size=up_kernel_size,           
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels
        )

    def forward(self, x):
        return self.unet(x)  # ✅ Predict noise only

#####################################
#####                           #####
#####        NPY Loader         #####
#####                           #####
#####################################


class NpyPatchDataset(Dataset):
    def __init__(self, input_folder, target_folder,mask_folder, transform=None):
        self.inputs = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.npy')])
        self.targets = sorted([os.path.join(target_folder, f) for f in os.listdir(target_folder) if f.endswith('.npy')])
        self.masks = sorted([os.path.join(mask_folder, f) for f in os.listdir(mask_folder) if f.endswith('.npy')])

        self.transform = transform
        assert len(self.inputs) == len(self.targets), "Mismatch between input and target files."

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        image = np.load(self.inputs[idx]).astype(np.float32)[np.newaxis, ...]
        target = np.load(self.targets[idx]).astype(np.float32)[np.newaxis, ...]
        mask=np.load(self.masks[idx]).astype(np.float32)[np.newaxis, ...]
        data = {"image": image, "target": target, "mask": mask}
        if self.transform:
            data = self.transform(data)
        return {"image": data["image"], "target": data["target"], "mask":data["mask"]}




################################
# User Interface - Using tkinter
################################


class PETDenoisingGUI:
# Training thread will be different to prevent freezing
    def start_training_thread(self):
        training_thread = threading.Thread(target=self.prepare)
        training_thread.daemon = True  # Automatically closes with main program
        training_thread.start()

    #######################################
    ######### Main Loss FUNCTİON ##########
    #######################################
    def adaptive_denoising_loss(self, output, target,mask=None):
        l1_weight = float(self.l1_weight.get())
        mse_weight = float(self.mse_weight.get())
        ssim_weight = float(self.ssim_weight.get())
        edge_weight = float(self.edge_weight.get())  # You can tune this
        charbonnier_weight = float(self.charbonnier_weight.get())

        mse = F.mse_loss(output, target)
        output_clamped = torch.clamp(output, 0, float(self.ssim_datarange.get()))
        target_clamped = torch.clamp(target, 0, float(self.ssim_datarange.get()))
        ssim = SSIMLoss(spatial_dims=3, data_range=float(self.ssim_datarange.get()))(output_clamped, target_clamped)
        edge = torch.nn.functional.l1_loss(gradient(output), gradient(target))

        l1 = F.l1_loss(output, target)
        charbonnier = CharbonnierLoss()(output, target)
        tumor_loss_float = 0.0
        total_loss = mse_weight * mse + ssim_weight * ssim + edge_weight * edge + l1_weight * l1 + charbonnier*charbonnier_weight
        if mask is not None:
            tumor_weight = float(self.tumor_weight.get())
            tumor_loss = asymmetric_l1_loss(output * mask, target * mask, neg_weight=float(self.tumor_asymweight.get()))*tumor_weight       
            total_loss +=  tumor_loss
            tumor_loss_float=tumor_loss.item()
            if tumor_weight == 0.0:
                tumor_loss_float=asymmetric_l1_loss(output * mask, target * mask).item()

        return total_loss, mse.item(), ssim.item(), edge.item(), l1.item(), charbonnier.item(), tumor_loss_float  



#####################################
#####                           #####
#####     MIP Viewer Logic      #####
#####                           #####
#####################################


    def show_mip_triplet(self, image_tensor, denoised_tensor, target_tensor, spacing, title=""):
        def prepare_image(img):
            img = img.squeeze().cpu().numpy()
            mip = np.max(img, axis=1)  # anterior view = max along axis 1
            mip = np.clip(mip, 0, 7) 
            mip = np.flipud(mip)  # 🔄 fix orientation
            return mip

        def prepare_image_lossmip(img):
            img = img.squeeze().cpu().numpy()
            mip = np.max(img, axis=1)  # anterior view = max along axis 1
            mip = np.clip(mip, 0, 7) 
            mip = np.flipud(mip)  # 🔄 fix orientation
            return mip

        def prepare_image_lossmin(img):
            img = img.squeeze().cpu().numpy()
            mip = np.min(img, axis=1)  # anterior view = max along axis 1
            mip = np.clip(mip, -7, 0) 
            mip = np.flipud(mip)  # 🔄 fix orientation
            return mip

        noisy_mip = prepare_image(image_tensor)
        denoised_mip = prepare_image(denoised_tensor)
        target_mip = prepare_image(target_tensor)
        loss_tensor = denoised_tensor - target_tensor
        loss_mip = prepare_image_lossmip(loss_tensor)
        loss_minip = prepare_image_lossmin(loss_tensor)

        self.mip_axes[0].imshow(noisy_mip, cmap='gray_r', aspect=spacing[2]/spacing[1], interpolation='bicubic', vmin=0, vmax=7)
        self.mip_axes[0].set_title("Noisy")

        self.mip_axes[1].imshow(denoised_mip, cmap='gray_r', aspect=spacing[2]/spacing[1], interpolation='bicubic', vmin=0, vmax=7)
        self.mip_axes[1].set_title("Denoised")

        self.mip_axes[2].imshow(target_mip, cmap='gray_r', aspect=spacing[2]/spacing[1], interpolation='bicubic', vmin=0, vmax=7)
        self.mip_axes[2].set_title("Target")

        self.mip_axes[3].imshow(loss_mip, cmap='seismic', aspect=spacing[2]/spacing[1], interpolation='bicubic', vmin=-7, vmax=7)
        self.mip_axes[3].set_title("L1 Positive Loss Map")

        self.mip_axes[4].imshow(loss_minip, cmap='seismic', aspect=spacing[2]/spacing[1], interpolation='bicubic', vmin=-7, vmax=7)
        self.mip_axes[4].set_title("L1 Negative Loss Map")

        for ax in self.mip_axes:
            ax.axis('off')
        self.mip_fig.suptitle(title)
        self.mip_canvas.draw()
        
        del noisy_mip
        del denoised_mip
        del target_mip
        del loss_tensor
        del loss_mip
        del loss_minip





#####################################
#####                           #####
#####        Contstruct         #####
#####        Main Window        #####
#####################################



    def __init__(self, root, architecture="swin"):
        self.root = root
        self.architecture = architecture

        root.title("Claritas - Denoising Trainer")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = f"trainSWIN_log_{timestamp}.txt"
        self.log_file = open(self.log_file_path, "w")
        # Folder selection
        self.train_dir_noisy = self.add_folder_selector("Training Noisy Image Folder (*.npy):")
        self.train_dir_ref = self.add_folder_selector("Training Reference Image Folder (*.npy):")
        self.train_mask_dir = self.add_folder_selector("Training Mask Image Folder (*.npy):")

        self.val_dir_noisy = self.add_folder_selector("Validation Noisy Image Folder (*.npy):")
        self.val_dir_ref = self.add_folder_selector("Validation Reference Image Folder (*.npy):")
        self.val_mask_dir = self.add_folder_selector("Validation Mask Image Folder (*.npy):")


        ## Only load necessary components
        if self.architecture == "swin":
            # SwinUNETR Model parameters
            self.feature_size = self.add_entry("Channels:", default="24")
            self.heads = self.add_entry("Number of Heads (comma separated):", default="3,6,12,24")
            self.depths = self.add_entry("Depths (comma separated):", default="2,2,2,2")
            self.do_rate = self.add_entry("Dropout Path Rate:", default="0.0")
 
        
        elif self.architecture == "unet":

            # UNET Model parameters
            self.feature_size = self.add_entry("Channels:", default="64,128,256,512,1024")
            self.res_units = self.add_entry("Res Units:", default="2")
            self.strides = self.add_entry("Strides (comma separated):", default="2,2,2,2")
            self.kernel = self.add_entry("Down Kernel (must be an odd number):", default="3")
            self.upkernel = self.add_entry("Up Kernel (must be an odd number):", default="3")


        # Loss weights
        self.mse_weight = self.add_entry("MSE Loss Weight:", default="0.0")
        self.ssim_weight = self.add_entry("SSIM Loss Weight:", default="0.0")
        self.ssim_datarange = self.add_entry("SSIM Data Range:", default="10")
        self.l1_weight = self.add_entry("L1 Loss Weight:", default="0.0")
        self.edge_weight = self.add_entry("Edge Loss Weight:", default="0.0")
        self.charbonnier_weight = self.add_entry("Charbonnier Loss Weight:", default="1.0")
        self.tumor_weight = self.add_entry("Tumor Specific Loss Weight (L1):", default="10.0")
        self.tumor_asymweight = self.add_entry("Tumor Negative Loss Multiplier:", default="1.0")

        self.learning_rate = self.add_entry("Learning Rate:", default="0.00001")
        self.epoch_num = self.add_entry("Epochs:", default="100")


        # UI for the transforms before validation
        self.add_label("Validation Transforms")
        self.val_padsize = self.add_entry("Validation SpatialPadd Size:", default="512,256,256")
        self.val_cropsize = self.add_entry("Val CenterSpatialCropd Size:", default="512,256,256")

        # UI for the loading previous pth file
        self.use_existing_model = tk.IntVar(value=0)
        tk.Checkbutton(root, text="Continue from existing pth?:", variable=self.use_existing_model).pack()
        self.previous_pth_path = self.add_file_selector("Previous pth file:")

        # Output filename. The output will be in the same folder as .py file...
        self.pth_filename = self.add_entry("Output .pth Filename:", default="pet_denoiser.pth")
        self.mip_case_entry = self.add_entry("Show MIP for Validation Case (nrrd filename):", default="0")
        self.mip_spacing = self.add_entry("Voxel spacing for MIP (H,W,D):", default="2,2,2")

        self.train_batchsize_entry = self.add_entry("Train Batch Size:", default="4")

        # Buttons
        tk.Button(root, text="Train", command=self.start_training_thread).pack(pady=10)

        # Output log
        self.log = scrolledtext.ScrolledText(root, height=20, width=100)
        self.log.pack()

        # Defining plotting window.
        self.plot_window = tk.Toplevel(self.root)
        self.plot_window.title("Training/Validation Loss Plot")

        self.fig, (self.mse_ax, self.ssim_ax) = plt.subplots(2, 1, figsize=(4, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_window)
        self.canvas.get_tk_widget().pack()


        # === MIP VIEWER SETUP ===
        self.mip_window = tk.Toplevel(self.root)
        self.mip_window.title("Anterior MIP Viewer")

        self.mip_fig, self.mip_axes = plt.subplots(1, 5, figsize=(15, 6))  # Noisy, Denoised, Target
        self.mip_canvas = FigureCanvasTkAgg(self.mip_fig, master=self.mip_window)
        self.mip_canvas.get_tk_widget().pack()


        self.mse_val_losses = []
        self.ssim_val_losses = []
        self.mse_losses = []
        self.ssim_losses = []        


        # Let me check if you have cuda.
        # Can still work with cpu but it will be painful...
        
        self.use_cuda = False
        if torch.cuda.is_available():
            try:
                total_vram = torch.cuda.get_device_properties(0).total_memory
                vram_gb = total_vram / (1024 ** 3)  # Convert bytes to GB
                if vram_gb >= 3.9:
                    self.use_cuda = True
                    self.print(f"CUDA available with {vram_gb:.1f} GB VRAM — using GPU.")
                else:
                    self.use_cuda = True
                    self.print(f"CUDA available but only {vram_gb:.1f} GB VRAM — using GPU.")
            except Exception as e:
                self.print(f"Could not check VRAM: {e}. Using CPU.")
        self.device = torch.device("cuda" if self.use_cuda else "cpu")


    # Functions to ease UI creation process

    def add_folder_selector(self, label):
        frame = tk.Frame(self.root)
        frame.pack(fill='x')
        tk.Label(frame, text=label, width=40).pack(side='left')
        entry = tk.Entry(frame, width=60)
        entry.pack(side='left', padx=5)
        tk.Button(frame, text="Browse", command=lambda: entry.insert(0, filedialog.askdirectory())).pack(side='left')
        return entry

    def add_file_selector(self, label):
        frame = tk.Frame(self.root)
        frame.pack(fill='x')
        tk.Label(frame, text=label, width=40).pack(side='left')
        entry = tk.Entry(frame, width=60)
        entry.pack(side='left', padx=5)
        tk.Button(frame, text="Browse", command=lambda: entry.insert(0, filedialog.askopenfilename())).pack(side='left')
        return entry


    def add_entry(self, label, default=""):
        frame = tk.Frame(self.root)
        frame.pack(fill='x')
        tk.Label(frame, text=label, width=40).pack(side='left')
        entry = tk.Entry(frame, width=60)
        entry.insert(0, default)
        entry.pack(side='left', padx=0)
        return entry

    def add_label(self, label):
        frame = tk.Frame(self.root)
        frame.pack(fill='x')
        tk.Label(frame, text=label, width=40).pack(side='left')
        return 0


    def print(self, message):
        self.log.insert(tk.END, message + "\n")
        self.log.see(tk.END)
        self.root.update()

        # Write to file
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.write(message + "\n")
            self.log_file.flush()

    def get_file_pairs(self,noisy_dir,ref_dir,mask_dir):
        target_files = sorted([os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith(".nrrd")])
        mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".nrrd")])
        k = 0
        datasett = []
        for x in sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir) if f.endswith(".nrrd")]):
            image_path = x
            target_path = target_files[k]
            mask_path=mask_files[k]
            datasett.append({"image": image_path, "target": target_path, "mask":mask_path})
            k = k+1
        return datasett

    # Preparing for training process.........
    def prepare(self):

 
 
        if self.architecture == "swin":
            # Getting input variables from UI
            xfeature_size = int(self.feature_size.get()) 
            pth_file = self.pth_filename.get()
            num_heads = tuple(int(s) for s in self.heads.get().split(","))
            do_rate = float(self.do_rate.get())

            # Defining model with inputs
            depths = tuple(int(s) for s in self.depths.get().split(","))

            model = SwinDenoiser(feature_size=xfeature_size,heads=num_heads,depths=depths,do_rate=do_rate).to(self.device)
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print(f"Trainable parameters: {num_params:,}")
            self.print(f"Feature size is: {xfeature_size}")
            self.print(f"Number of Heads is: {num_heads}")
            self.print(f"Depths are: {depths}")
            self.print(f"Dropout Path Rate: {do_rate}")

        elif self.architecture == "unet":
            pth_file = self.pth_filename.get()

            # Getting input variables from UI
            feature_size = tuple(int(s) for s in self.feature_size.get().split(","))
            res_units = int(self.res_units.get())
            strides = tuple(int(s) for s in self.strides.get().split(","))
            kernel_size = int(self.kernel.get())
            up_kernel_size = int(self.upkernel.get())
            model = DenoiseUNet(channels=feature_size, num_res_units=res_units,strides=strides,kernel_size=kernel_size,up_kernel_size=up_kernel_size).to(self.device)
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print(f"Trainable parameters: {num_params:,}")
            self.print(f"Channel sizes are: {feature_size}")
            self.print(f"Number of Res Units is: {res_units}")
            self.print(f"Strides are: {strides}")
            self.print(f"Down and Up Kernel sizes are: {kernel_size}/{up_kernel_size}")
            self.print(f"Model name to be saved is: {pth_file}")

        # The transforms to be applied before training. If you wish to add more variability, you are welcome to edit this section. Please refer to MONAI Transforms documentation. 

        train_transforms = Compose([
            # ✅ Add random flips
            RandFlipd(
                keys=["image", "target", "mask"],
                spatial_axis=[0],  # flip along z-axis
                prob=0.3
            ),
            RandFlipd(
                keys=["image", "target", "mask"],
                spatial_axis=[1],  # flip along y-axis
                prob=0.3
            ),
            RandFlipd(
                keys=["image", "target", "mask"],
                spatial_axis=[2],  # flip along x-axis
                prob=0.3
            ),

        ])

        # The transforms to be applied before validation. If you wish to add more variability, you are welcome to edit this section. Please refer to MONAI Transforms documentation.
        val_transforms = Compose([
            SpatialPadd(keys=["image", "target", "mask"], spatial_size=tuple(int(s) for s in self.val_padsize.get().split(","))),
            CenterSpatialCropd(keys=["image", "target","mask"],  roi_size=tuple(int(s) for s in self.val_cropsize.get().split(",")))  # Crop 3D patches
        ])

        # Get file names
        # Generate datasets
        train_ds = NpyPatchDataset(self.train_dir_noisy.get(), self.train_dir_ref.get(),self.train_mask_dir.get(), transform=train_transforms)
        val_ds = NpyPatchDataset(self.val_dir_noisy.get(), self.val_dir_ref.get(),self.val_mask_dir.get(), transform=val_transforms)

        # Generate dataloaders
        train_loader = DataLoader(train_ds, batch_size=int(self.train_batchsize_entry.get()), shuffle=True,collate_fn=pad_list_data_collate)
        self.val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        # Defining Adam as optimizer
        optimizer = optim.Adam(model.parameters(), lr=float(self.learning_rate.get()))

        # Load existing model if the checkbox is checked. Warning if the parameters is not exactly the same, it will throw error..
        
        if self.use_existing_model.get() == 1 and os.path.isfile(self.previous_pth_path.get()):
            try:
                model.load_state_dict(torch.load(self.previous_pth_path.get(), map_location=self.device))
                self.print("Loaded previous model successfully.")
            except Exception as e:
                self.print(f"Failed to load model: {e}")
        else:
            self.print("Skipping model load. Training from scratch.")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.print(f"Trainable parameters: {num_params:,}")

        self.print("\n===== Training Started =====")

        # TRAINING! :)
        self.train_model(model, train_loader, self.val_loader, optimizer, num_epochs=int(self.epoch_num.get()),pth_filename=pth_file)

        self.print(f"\nModel saved to {pth_file}")



    # Main Training Function
    def train_model(self,model, train_loader, val_loader, optimizer, num_epochs=50,pth_filename="best_pet_denoiser.pth"):
        model.train()
        best_val_loss = float("inf")

        # First will check the original MSE and SSIM loss values in comparison with input and target images.
        self.print("\nNow will compare the original noisy images with target images for reference. Be patient.")
        origval = self.evaluate_val_input_vs_target(self.val_loader,data_range=float(self.ssim_datarange.get()))
        origvalmse = []
        origvalssim = []


        # Training loop
        for epoch in range(num_epochs):
            epoch_start_time = time.time()  # ⏰ Start timer for epoch
            epoch_loss = 0
            total_mse = 0
            total_ssim = 0
            total_edge = 0
            total_l1 = 0
            total_charbonnier = 0
            total_tumor_loss = 0
            num_patches_this_epoch = 0  # 🔍 New counter

            for batch in train_loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                targets = batch["target"].to(self.device)
                
                num_patches_this_epoch += images.shape[0]  # Add number of patches in this batch

                optimizer.zero_grad()

                predicted_noise = model(images)
                denoised_output = torch.clamp(images - predicted_noise, min=0) 
                loss, mse, ssim, edge, l1, charbonnier, tumor_loss =self.adaptive_denoising_loss(denoised_output, targets,mask=masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                total_mse += mse
                total_ssim += ssim
                total_edge += edge
                total_l1 += l1
                total_charbonnier += charbonnier
                total_tumor_loss+=tumor_loss


            self.print(f"\nEpoch {epoch+1}: Processed {num_patches_this_epoch} patches in training.")
            avg_train_loss = epoch_loss / len(train_loader)
            avg_mse = total_mse / len(train_loader)
            avg_ssim = total_ssim / len(train_loader)
            avg_edge = total_edge / len(train_loader)
            avg_l1 = total_l1 / len(train_loader)
            avg_charbonnier = total_charbonnier/len(train_loader)
            avg_tumor_loss = total_tumor_loss/len(train_loader)
            self.print(f"Epoch {epoch + 1}/{num_epochs}")
            self.print(f"Train: Loss={avg_train_loss:.6f}, MSE_Loss={avg_mse:.6f}, SSIM_Loss={avg_ssim:.6f},Edge_Loss={avg_edge:.6f},L1_Loss={avg_l1:.6f}, Charbonnier Loss = {avg_charbonnier:.6f}")
            self.print(f"Tumor specific Train Loss is = {avg_tumor_loss:.6f}")

            # Validate
            val_start_time = time.time()  # ⏰ Start timer for validation
            val_metrics = self.validate_model(model, self.val_loader)
            val_end_time = time.time()    # ⏰ End timer for validation
            val_elapsed = val_end_time - val_start_time

            epoch_end_time = time.time()   # ⏰ End timer for epoch
            epoch_elapsed = epoch_end_time - epoch_start_time
            self.mse_losses.append(avg_mse)
            self.ssim_losses.append(avg_ssim)
            self.mse_val_losses.append(val_metrics['mse'])
            self.ssim_val_losses.append(val_metrics['ssim'])
            origvalmse.append(origval["mse"])
            origvalssim.append(origval["ssim"])



            # ===== MSE Plot =====
            self.mse_ax.clear()
            self.mse_ax.set_yscale("log")

            mse_all = self.mse_losses + self.mse_val_losses
            if mse_all:
                ymin = 10 ** math.floor(math.log10(max(min(mse_all), 1e-6)))
                ymax = 10 ** math.ceil(math.log10(max(mse_all)))
                self.mse_ax.set_ylim(ymin, ymax)

            self.mse_ax.plot(range(1, len(self.mse_losses) + 1), self.mse_losses, label="Train MSE", color="blue")
            self.mse_ax.plot(range(1, len(self.mse_val_losses) + 1), self.mse_val_losses, label="Val MSE", color="cyan")
            self.mse_ax.plot(range(1, len(self.mse_val_losses) + 1), origvalmse, label="Original Validation MSE", color="green")
            self.mse_ax.set_title("MSE Loss over Epochs")
            self.mse_ax.set_xlabel("Epoch")
            self.mse_ax.set_ylabel("Loss (log scale)")
            self.mse_ax.legend()
            self.mse_ax.grid(True)

            # ===== SSIM Plot =====
            self.ssim_ax.clear()
            self.ssim_ax.set_yscale("log")

            ssim_all = self.ssim_losses + self.ssim_val_losses
            if ssim_all:
                ymin = 10 ** math.floor(math.log10(max(min(ssim_all), 1e-6)))
                ymax = 10 ** math.ceil(math.log10(max(ssim_all)))
                self.ssim_ax.set_ylim(ymin, ymax)

            self.ssim_ax.plot(range(1, len(self.ssim_losses) + 1), self.ssim_losses, label="Train SSIM", color="red")
            self.ssim_ax.plot(range(1, len(self.ssim_val_losses) + 1), self.ssim_val_losses, label="Val SSIM", color="orange")
            self.ssim_ax.plot(range(1, len(self.ssim_val_losses) + 1), origvalssim, label="Original Validation SSIM", color="blue")
            self.ssim_ax.set_title("SSIM Loss over Epochs")
            self.ssim_ax.set_xlabel("Epoch")
            self.ssim_ax.set_ylabel("Loss (log scale)")
            self.ssim_ax.legend()
            self.ssim_ax.grid(True)

            # Redraw
            self.canvas.draw()

            self.print(f"Val:   Loss={val_metrics['loss']:.6f}, MSE_Loss={val_metrics['mse']:.6f}, SSIM_Loss={val_metrics['ssim']:.6f},Edge_Loss={val_metrics['edge']:.6f},L1_Loss={val_metrics['l1']:.6f},Charbonnier_Loss={val_metrics['charbonnier']:.6f},Tumor_Loss={val_metrics['tloss']:.6f}")
            self.print(f"Train+Val time: {epoch_elapsed:.2f} sec (Validation: {val_elapsed:.2f} sec)")

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save(model.state_dict(),pth_filename)
                self.print("Model saved based on Val_Loss.")
#            torch.save(model.state_dict(),pth_filename)
#            self.print("Model saved...")


    # ================================
    # Validate & Save Model Output
    # ================================
    def validate_model(self,model, val_loader):
        model.eval()
        val_loss = 0
        total_mse = 0
        total_ssim = 0
        total_edge = 0
        total_l1 = 0
        total_charbonnier = 0
        total_tloss = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                images = batch["image"].to(self.device)
                targets = batch["target"].to(self.device)
                masks = batch["mask"].to(self.device)
                predicted_noise = sliding_window_inference(
                    inputs=images,
                    roi_size=(64, 64, 64),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.25,
                    mode="gaussian"
                )

                denoised_output = torch.clamp(images - predicted_noise, min=0) 
                # ==== MIP Viewer Logic ====
                mip_index = int(self.mip_case_entry.get())
                if i == mip_index:  # i from enumerate(val_loader)
                    spacing = (2, 2,2)  # z, y, x in mm
                    self.show_mip_triplet(images, denoised_output, targets, spacing, title=f"Val sample {i}")


                loss, mse, ssim, edge, l1, charbonnier, tloss = self.adaptive_denoising_loss(denoised_output, targets,mask=masks)
                val_loss += loss.item()
                total_mse += mse
                total_ssim += ssim
                total_edge += edge
                total_l1 += l1
                total_charbonnier += charbonnier
                total_tloss +=tloss
                self.print(f"Individual Validation Loss:  Loss={loss.item():.6f}, MSE_Loss={mse:.6f}, SSIM_Loss={ssim:.6f},Edge_Loss={edge:.6f},L1_Loss={l1:.6f},Charbonnier Loss={charbonnier:.6f}, Tumor_Loss = {tloss:.6f}")

                del loss
                del images
                del targets
                del masks
                del predicted_noise
                del denoised_output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                import gc
                gc.collect()



        avg_loss = val_loss / len(val_loader)
        avg_mse = total_mse / len(val_loader)
        avg_ssim = total_ssim / len(val_loader)
        avg_edge = total_edge / len(val_loader)
        avg_l1 = total_l1 / len(val_loader)
        avg_charbonnier = total_charbonnier/ len(val_loader)
        avg_tloss = total_tloss/ len(val_loader)
        return{
            "loss": avg_loss,
            "mse": avg_mse,
            "ssim": avg_ssim,
            "edge": avg_edge,
            "l1":avg_l1,
            "charbonnier":avg_charbonnier,
            "tloss":avg_tloss
        }

    def evaluate_val_input_vs_target(self, val_loader=None, data_range=50.0):
        l1_total = 0.0
        mse_total = 0.0
        ssim_total = 0.0
        edge_total = 0.0
        charbonnier_total= 0.0
        count = 0
        total_loss = 0
        tloss_total =0
        for batch in val_loader:
            img = batch["image"].to(self.device)
            target = batch["target"].to(self.device)
            masks = batch["mask"].to(self.device)
            self.print(f"Val input shape: {img.shape}")
            spacing = tuple(int(s) for s in self.mip_spacing.get().split(","))  #h,w,d
            self.show_mip_triplet(img, img, target, spacing, title=f"Val sample")
            # Use your own loss function
            loss, mse, ssim_loss, edge,l1,charbonnier, tloss = self.adaptive_denoising_loss(img, target,mask=masks)
            total_loss += loss.item()
            self.print(f"Original Individual Validation Loss:  MSE_Loss={mse:.6f}, SSIM_Loss={ssim_loss:.6f},Edge_Loss={edge:.6f},L1_Loss={l1:.6f},Charbonnier={charbonnier:.6f},Tumor_Loss = {tloss:.6f}")
            mse_total += mse
            ssim_total += ssim_loss
            edge_total += edge
            l1_total += l1
            tloss_total +=tloss
            charbonnier_total += charbonnier
            count += 1
            del img, target, masks, loss, batch




        avg_mse = mse_total / count
        avg_ssim = ssim_total / count
        avg_edge = edge_total / count
        avg_l1 = l1_total/count
        avg_charbonnier = charbonnier_total/count
        avg_loss = total_loss/count
        avg_tloss = tloss_total/count
        self.print(f"Avg Loss (input vs. target): {avg_loss:.6f}")
        self.print(f"Avg MSE (input vs. target): {avg_mse:.6f}")
        self.print(f"Avg SSIM Loss (input vs. target): {avg_ssim:.6f}")
        self.print(f"Avg Edge Loss (input vs. target): {avg_edge:.6f}")
        self.print(f"Avg L1 Loss (input vs. target): {avg_l1:.6f}")
        self.print(f"Avg Charbonnier Loss (input vs. target): {avg_charbonnier:.6f}")
        self.print(f"Avg Tumor Loss (input vs. target): {avg_tloss:.6f}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        import gc
        gc.collect()
        return{
            "mse": mse_total / count,
            "ssim": ssim_total / count,
        }


if __name__ == "__main__":



    root = tk.Tk()
    root.withdraw()  # Hide main window until model is selected

    def launch_gui(arch_choice):
        root.architecture = arch_choice
        root.deiconify()  # Show main window now
        app = PETDenoisingGUI(root, architecture=arch_choice)


    # Launch pop-up
    selector = tk.Toplevel()
    selector.title("Select Model Architecture")

    tk.Label(selector, text="Choose model architecture to use:").pack(padx=20, pady=10)

    tk.Button(selector, text="SwinUNETR", width=20, command=lambda: (selector.destroy(), launch_gui("swin"))).pack(pady=5)
    tk.Button(selector, text="UNet", width=20, command=lambda: (selector.destroy(), launch_gui("unet"))).pack(pady=5)

    selector.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()