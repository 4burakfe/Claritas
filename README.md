# Claritas

**Claritas** is a collection of standalone Python tools for denoising PET images using SwinUNETR-based neural networks. 
It includes scripts for preprocessing `.nrrd` PET data, training models with or without tumor masks, and evaluating performance. 
The repository is designed for research purposes only.
The developed models can be directly implemented in SlicerPETDenoise Extension which is available from here: https://github.com/4burakfe/SlicerPETDenoise

---
## 🛠️ Script Descriptions

### `Claritas_Train_UI_single_wmasks.py`

GUI application for training PET denoising models **with tumor masks**.

**Features:**
- Supports SwinUNETR and UNet
- Fully customizable loss functions (MSE, MAE, SSIM, tumor-weighted L1)
- Real-time training plots and logging
- Uses .npy files for training and validation
- The input files shold have be in pairs in seperate folders with exact same names

### `Claritas_Train_UI_single_nomask.py`
Same as above but for training without tumor segmentation masks.

### `batch_nrrd_to_npy_fullvol.py`
Resamples .nrrd PET files to 2×2×2 mm
Crops central 256×256 area
Saves full volumes as .npy

### `batch_nrrd_to_npy_gridpatch.py`
Extracts non-overlapping patches of size 64×64×64
Saves each patch as a separate .npy file


## 🧠 Model Descriptions

Models are designed to use with SlicerPETDenoise Extension which is available from here: https://github.com/4burakfe/SlicerPETDenoise
| Filename                   | Description |
|---------------------------|-------------|
| `pet_denoiser_std_char.pth` | Trained with standard Charbonnier loss for general denoising. |
| `pet_denoiser_x19w1_5.pth` | Tumor-weighted loss model (1.5x penalty for underestimation). |
| `pet_denoiser_x14w3.pth`   | Strong tumor-focused model (3x weight for underestimation). |

---


📦 Requirements
Python 3.9+
PyTorch
MONAI
SimpleITK
numpy
matplotlib
einops

For installation of torch please visit https://pytorch.org
For other dependencies you can use command: "pip install monai einops SimpleITK matplotlib numpy"

Disclaimer: This software is for research purposes only and is not certified for clinical use.

