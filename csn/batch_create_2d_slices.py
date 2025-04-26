import os
from pathlib import Path
from typing import Optional
import SimpleITK as sitk
import pandas as pd
from config import RAW_DATA_PATH, DEFAULT_OUTPUT_DIR, CSV_OUTPUT_DIR

def resize_2d_image(itk_image: sitk.Image, new_size: tuple, method=sitk.sitkLinear) -> sitk.Image:
    """Resize 2D image while preserving spatial information."""
    resampler = sitk.ResampleImageFilter()
    original_size = itk_image.GetSize()
    original_spacing = itk_image.GetSpacing()
    
    new_spacing = [
        original_spacing[0] * original_size[0] / new_size[0],
        original_spacing[1] * original_size[1] / new_size[1]
    ]
    
    resampler.SetReferenceImage(itk_image)
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(method)
    
    return resampler.Execute(itk_image)

def process_slice(image: sitk.Image, origin: list, direction: list, size: tuple) -> sitk.Image:
    """Process individual 2D slice with proper spatial attributes."""
    processed = resize_2d_image(image, size, sitk.sitkNearestNeighbor)
    processed.SetOrigin([origin[0], origin[2]])
    processed.SetSpacing([1, 1])
    processed.SetDirection([direction[0], direction[2], direction[3], direction[4]])
    return processed

def create_matched_hq_lq_slices(data_type: str = 'train', output_dir: Optional[Path] = None) -> None:
    """Batch create matched HQ and LQ coronal CT 2D slice NII files."""
    input_dir = RAW_DATA_PATH / data_type
    output_dir = output_dir or DEFAULT_OUTPUT_DIR / data_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    new_size = (512, 512)
    
    for subject in sorted(input_dir.iterdir()):
        pid = subject.stem
        lq_path = subject / 'LQ' / 'image.nii.gz'
        hq_path = subject / 'HQ' / 'image.nii.gz'
        
        lq_img = sitk.ReadImage(str(lq_path))
        hq_img = sitk.ReadImage(str(hq_path))
        
        lq_data = sitk.GetArrayFromImage(lq_img)
        hq_data = sitk.GetArrayFromImage(hq_img)
        
        if lq_data.shape[1:] != hq_data.shape[1:]:
            continue
            
        for slice_idx in range(lq_data.shape[1]):
            # Process LQ slice
            lq_slice = process_slice(
                sitk.GetImageFromArray(lq_data[:, slice_idx, :]),
                lq_img.GetOrigin(),
                lq_img.GetDirection(),
                new_size
            )
            lq_output_path = output_dir / 'LQ' / f'{pid}_cor_{slice_idx}.nii.gz'
            lq_output_path.parent.mkdir(exist_ok=True)
            sitk.WriteImage(lq_slice, str(lq_output_path))
            
            # Process HQ slice
            hq_slice = process_slice(
                sitk.GetImageFromArray(hq_data[:, slice_idx, :]),
                lq_img.GetOrigin(),  # Use same spatial ref as LQ
                lq_img.GetDirection(),
                new_size
            )
            hq_output_path = output_dir / 'HQ' / f'{pid}_cor_{slice_idx}.nii.gz'
            hq_output_path.parent.mkdir(exist_ok=True)
            sitk.WriteImage(hq_slice, str(hq_output_path))
            
            records.append({
                'pid': pid,
                'lq_nii_path': str(lq_output_path),
                'hq_nii_path': str(hq_output_path)
            })
    
    # Save records to CSV
    pd.DataFrame(records).to_csv(
        CSV_OUTPUT_DIR / f'lq_hq_pid_nii_cor_slices_path_data_{data_type}.csv',
        index=False
    )

if __name__ == '__main__':
    create_matched_hq_lq_slices()
