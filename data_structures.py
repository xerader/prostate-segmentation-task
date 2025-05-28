import numpy as np 
import os
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

class MedicalImage:
    def __init__(self, image_vol, mask_vol, image_id=None, spacing=[1.0, 1.0, 1.0]):
        """
        image_vol: 3D numpy array, shape [D, H, W]
        mask_vol:  3D numpy array, shape [D, H, W]
        image_id:  identifier string/number for this case (optional)
        spacing:   voxel spacing for the image
        """
        self.image = image_vol
        self.mask = mask_vol
        self.id = image_id
        self.generated_mask = None
        self.spacing = spacing
    
    def __len__(self):
        """Return number of slices in the image"""
        return self.image.shape[0]
    
    def get_slice(self, idx):
        """Return one slice (image and mask)"""
        return self.image[idx], self.mask[idx]
        
    # When saving the prediction to NIfTI
    def save_to_nifti(self, output_dir='nifti_results', threshold=0.5):
        """Save image, mask, and prediction (if available) as NIfTI files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # First, read the original image to get its metadata
        original_image_path = os.path.join('./data/UCL/processed', f'{self.id}.nii')
        if os.path.exists(original_image_path):
            original_sitk = sitk.ReadImage(original_image_path)
            # Get metadata from original
            origin = original_sitk.GetOrigin()
            direction = original_sitk.GetDirection()
            spacing = original_sitk.GetSpacing()
        else:
            # Use default values if original not found
            origin = (0.0, 0.0, 0.0)
            direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            spacing = self.spacing
        
        # Convert to SimpleITK images
        image_sitk = sitk.GetImageFromArray(self.image)
        mask_sitk = sitk.GetImageFromArray(self.mask)
        
        # Set metadata
        for img in [image_sitk, mask_sitk]:
            img.SetOrigin(origin)
            img.SetDirection(direction)
            img.SetSpacing(spacing)
        
        # Save the images
        sitk.WriteImage(image_sitk, os.path.join(output_dir, f'{self.id}_input.nii'))
        sitk.WriteImage(mask_sitk, os.path.join(output_dir, f'{self.id}_ground_truth.nii'))
        
        # Save prediction if available
        if self.generated_mask is not None:
            # Make prediction binary using threshold
            binary_pred = (self.generated_mask > threshold).astype(np.float32)
            pred_sitk = sitk.GetImageFromArray(binary_pred)
            
            # Set the same metadata
            pred_sitk.SetOrigin(origin)
            pred_sitk.SetDirection(direction)
            pred_sitk.SetSpacing(spacing)
            
            sitk.WriteImage(pred_sitk, os.path.join(output_dir, f'{self.id}_prediction.nii'))
            print(f"Saved NIfTI files for case {self.id}")
        else:
            print(f"Warning: No prediction available for case {self.id}")
    
    def calculate_dice(self, threshold=0.5):
        """Calculate Dice coefficient between mask and generated_mask"""
        if self.generated_mask is None:
            return None
        
        pred_binary = (self.generated_mask > threshold).astype(np.float32)
        true_binary = self.mask.astype(np.float32)
        
        # Calculate Dice
        smooth = 1e-6
        y_true_f = true_binary.flatten()
        y_pred_f = pred_binary.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
        
        return dice

class SliceDataset(Dataset):
    def __init__(self, case_list, transform=None):
        self.slices = []
        self.transform = transform
        # Flatten all slices from all cases into a list
        for case in case_list:
            for idx in range(len(case)):
                self.slices.append((case, idx))
    
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        case, slice_idx = self.slices[idx]
        img, msk = case.get_slice(slice_idx)
        # [1, H, W]
        img = np.expand_dims(img, 0).astype(np.float32)
        msk = np.expand_dims(msk, 0).astype(np.float32)
        
        if self.transform:
            img = self.transform(img)
            msk = self.transform(msk)
            
        img = torch.from_numpy(img)
        msk = torch.from_numpy(msk)
        return img, msk