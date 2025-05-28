import nibabel as nib
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch.nn.functional as F
import SimpleITK as sitk

from models import UNet  # Assuming UNet is defined in models.py
from data_structures import MedicalImage, SliceDataset

# set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# check if gpu is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

def resample_image(image, target_size=(256, 256), is_mask=False):
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
    # Normalize to [0,1], avoid divide by zero
    min_val = np.min(img_slice)
    max_val = np.max(img_slice)
    if max_val - min_val > 1e-5:
        return (img_slice - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(img_slice)

def load_and_preprocess_data_volumetric(data_dir='./data', target_size=(256, 256)):
    cases = []
    for folder in glob.glob(os.path.join(data_dir, '*')):
        if not os.path.isdir(folder) or not os.path.exists(os.path.join(folder, 'processed')) or folder.endswith('UCL'):
            print("UCL folder found, skipping...")
            continue
        img_files = glob.glob(os.path.join(folder, 'processed', 'Case*.nii'))
        for img_file in img_files:
            if '_segmentation' in img_file:
                continue
            case_id = os.path.basename(img_file).split('.')[0]
            seg_file = os.path.join(folder, 'processed', f"{case_id}_segmentation.nii")
            if os.path.exists(seg_file):
                print(f"Processing {img_file} and {seg_file}")
                # Read using SimpleITK
                sitk_img = sitk.ReadImage(img_file)
                sitk_seg = sitk.ReadImage(seg_file)

                # Resample 
                if target_size is not None:
                    sitk_img = resample_image(sitk_img, target_size=target_size, is_mask=False)
                    sitk_seg = resample_image(sitk_seg, target_size=target_size, is_mask=True)

                img_arr = sitk.GetArrayFromImage(sitk_img)  # [slices, H, W]
                seg_arr = sitk.GetArrayFromImage(sitk_seg)

                # Normalize and binarize
                img_arr = np.stack([normalize_image(slc) for slc in img_arr])
                seg_arr = (seg_arr > 0).astype(np.float32)
                cases.append(MedicalImage(img_arr, seg_arr, image_id=case_id))
    return cases

##########################################
#
#      TRAINING WITH CROSS-VALIDATION
#
#########################################

from sklearn.model_selection import KFold

# Load all the cases
cases = load_and_preprocess_data_volumetric()
ids = [c.id for c in cases]

print(f"Total number of cases loaded: {len(cases)}")

# Implement 5-fold cross validation
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Store fold results
all_train_losses = []
all_val_losses = []
all_best_epochs = []
all_best_val_losses = []

# Define loss function
criterion = nn.BCELoss()

def train_model(model, train_loader, val_loader, criterion, optimizer, fold_num, num_epochs=50, patience=15):
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    
    # Save initial model state to restore the best model later
    torch.save(model.state_dict(), f'unet_model_fold{fold_num}_init.pth')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                val_loss = criterion(outputs, masks)
                
                running_val_loss += val_loss.item() * images.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f'Fold {fold_num}, Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}', flush=True)
        
        # Check if this is the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'unet_model_fold{fold_num}_best.pth')
            print(f'Model saved at epoch {epoch+1}', flush=True)
        else:
            epochs_no_improve += 1
            
        # Early stopping with patience
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}! No improvement for {patience} epochs.', flush=True)
            break
    
    # Load the best model for this fold
    model.load_state_dict(torch.load(f'unet_model_fold{fold_num}_best.pth'))
    
    return train_losses, val_losses, best_epoch, best_val_loss

# Train models with cross validation
fold = 0
for train_idx, val_idx in kf.split(cases):
    fold += 1
    print(f"\n{'='*50}")
    print(f"Starting Fold {fold}/{n_folds}")
    print(f"{'='*50}")
    
    # Get training and validation cases for this fold
    train_cases_fold = [cases[i] for i in train_idx]
    val_cases_fold = [cases[i] for i in val_idx]
    
    print(f"Number of training cases in fold {fold}: {len(train_cases_fold)}")
    print(f"Number of validation cases in fold {fold}: {len(val_cases_fold)}")
    
    # Create datasets for this fold
    train_dataset = SliceDataset(train_cases_fold)
    val_dataset = SliceDataset(val_cases_fold)
    
    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize a fresh model for this fold
    model = UNet().to(device)
    
    # Define optimizer for this fold
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Train the model for this fold with early stopping
    train_losses, val_losses, best_epoch, best_val_loss = train_model(
        model, train_loader, val_loader, criterion, optimizer, fold, num_epochs=30, patience= 5
    )
    
    # Store results for this fold
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    all_best_epochs.append(best_epoch)
    all_best_val_losses.append(best_val_loss)
    
    # plot training and validation losses for this fold
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Epoch')
    plt.title(f'Fold {fold} Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(f'fold_{fold}_losses.png')
    


# Summarize results across all folds
print("\nCross-validation Results:")
for fold in range(n_folds):
    print(f"Fold {fold+1}: Best epoch = {all_best_epochs[fold]+1}, Best validation loss = {all_best_val_losses[fold]:.4f}")

print(f"\nAverage best validation loss: {np.mean(all_best_val_losses):.4f}")
print(f"Standard deviation of best validation loss: {np.std(all_best_val_losses):.4f}")

# Identify best fold
best_fold = np.argmin(all_best_val_losses) + 1
print(f"\nBest performing fold: {best_fold} with validation loss: {min(all_best_val_losses):.4f}")

# Plot training curves for all folds
plt.figure(figsize=(12, 8))

# Plot training loss
for fold in range(n_folds):
    plt.subplot(2, 1, 1)
    plt.plot(all_train_losses[fold], label=f'Fold {fold+1}')
    plt.axvline(x=all_best_epochs[fold], color=f'C{fold}', linestyle='--')

plt.title('Training Loss by Fold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot validation loss
for fold in range(n_folds):
    plt.subplot(2, 1, 2)
    plt.plot(all_val_losses[fold], label=f'Fold {fold+1}')
    plt.axvline(x=all_best_epochs[fold], color=f'C{fold}', linestyle='--')

plt.title('Validation Loss by Fold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('cross_validation_results.png')
plt.close()

# Final model using best fold
print("\nTraining final model using best fold configuration...")

# Load the best model from the best fold
best_model = UNet().to(device)
best_model.load_state_dict(torch.load(f'unet_model_fold{best_fold}_best.pth'))

# Save as the final model
torch.save(best_model.state_dict(), 'unet_model_best.pth')
print(f"Final model saved as 'unet_model_best.pth' (from fold {best_fold})")

