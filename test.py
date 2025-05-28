import nibabel as nib
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk

from models import UNet
from data_structures import MedicalImage, SliceDataset

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def resample_image(image, target_size=(256, 256), is_mask=False):
    """
    Function to resample an image to a target size.
    Args:
        image (SimpleITK.Image): Input image to resample.
        target_size (tuple): Desired output size (width, height).
        is_mask (bool): If True, use nearest neighbor interpolation (for masks).
    """
    # Get current image size
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    
    # Calculate new spacing to fit target size
    new_size = [target_size[0], target_size[1], original_size[2]]
    new_spacing = [
        original_spacing[0] * original_size[0] / new_size[0],
        original_spacing[1] * original_size[1] / new_size[1],
        original_spacing[2]
    ]

    # Set interpolator
    interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    
    resample = sitk.ResampleImageFilter()
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetInterpolator(interpolator)
    
    return resample.Execute(image)

def normalize_image(img_slice):
    """Function to normalize a single image slice to [0, 1] range.
    Normalization is done by subtracting the minimum value and dividing by the range.
    """
    # Normalize to [0,1], avoid divide by zero
    min_val = np.min(img_slice)
    max_val = np.max(img_slice)
    if max_val - min_val > 1e-5:
        return (img_slice - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(img_slice)

def load_and_preprocess_test_data(data_dir='./data', target_size=(256, 256)):
    test_cases = []
    folder = os.path.join(data_dir, 'UCL/processed')
    
    print(f"Loading test data from {folder}...", flush=True)
    if not os.path.isdir(folder):
        print(f"Directory not found: {folder}")
        return test_cases
    
    img_files = glob.glob(os.path.join(folder, 'Case*.nii'))
    for img_file in img_files:
        case_id = os.path.basename(img_file).split('.')[0]
        seg_file = os.path.join(folder, f"{case_id}_segmentation.nii")
        
        if os.path.exists(seg_file):
            print(f"Processing {case_id}")
            
            # Read images
            sitk_img = sitk.ReadImage(img_file)
            sitk_seg = sitk.ReadImage(seg_file)
            original_spacing = sitk_img.GetSpacing()
            
            # Resample if needed
            if target_size is not None:
                sitk_img = resample_image(sitk_img, target_size=target_size, is_mask=False)
                sitk_seg = resample_image(sitk_seg, target_size=target_size, is_mask=True)
            
            # Convert to numpy arrays
            img_arr = sitk.GetArrayFromImage(sitk_img)  # [slices, H, W]
            seg_arr = sitk.GetArrayFromImage(sitk_seg)
            
            # Normalize image and binarize mask
            img_arr = np.stack([normalize_image(slc) for slc in img_arr])
            seg_arr = (seg_arr > 0).astype(np.float32)
            
            # Create MedicalImage object
            test_cases.append(MedicalImage(
                img_arr, seg_arr, image_id=case_id, spacing=original_spacing
            ))
    
    return test_cases

def run_inference(model, case, batch_size=16):
    """Run inference on a single case and return the case with predictions"""
    model.eval()
    test_dataset = SliceDataset([case])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create empty array for predictions with the EXACT same shape as the input
    predictions = np.zeros_like(case.image, dtype=np.float32)
    
    # Track which slices have been processed
    processed_slices = set()
    
    # Process all slices
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            # Get batch indices
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(test_dataset))
            
            # Run model
            images = images.to(device)
            outputs = model(images)
            
            # Store predictions
            for j, output in enumerate(outputs):
                if start_idx + j < len(test_dataset):
                    # Get the exact slice index from the dataset
                    case_obj, slice_idx = test_dataset.slices[start_idx + j]
                    # Ensure we're using the correct case
                    assert case_obj.id == case.id, "Case mismatch in dataset"
                    # Store prediction at the exact same slice index
                    predictions[slice_idx] = output[0].cpu().numpy()
                    processed_slices.add(slice_idx)
    
    # Verify all slices were processed
    all_slices = set(range(len(case)))
    if processed_slices != all_slices:
        missing = all_slices - processed_slices
        print(f"Warning: Slices {missing} were not processed for case {case.id}")
    
    # Store predictions in the case object
    case.generated_mask = predictions
    return case

def evaluate_results(cases):
    """Evaluate Dice scores for all cases"""
    dice_scores = []
    
    for case in cases:
        dice = case.calculate_dice()
        if dice is not None:
            dice_scores.append(dice)
            print(f"Case {case.id} Dice: {dice:.4f}")
    
    # Print overall statistics
    if dice_scores:
        print(f"\nOverall Dice (mean): {np.mean(dice_scores):.4f}")
        print(f"Overall Dice (median): {np.median(dice_scores):.4f}")
        print(f"Dice range: {np.min(dice_scores):.4f} - {np.max(dice_scores):.4f}")
    
    return dice_scores

def main():
    # Load model
    model = UNet().to(device)
    model.load_state_dict(torch.load('unet_model_fold1_best.pth', map_location=device))
    
    # Load test data
    test_cases = load_and_preprocess_test_data()
    print(f"Loaded {len(test_cases)} test cases.")
    
    # Run inference on each case
    for case in test_cases:
        print(f"Processing case {case.id}...")
        case = run_inference(model, case)
        
        # Save results
        case.save_to_nifti()
    
    # Evaluate results
    evaluate_results(test_cases)

if __name__ == "__main__":
    main()