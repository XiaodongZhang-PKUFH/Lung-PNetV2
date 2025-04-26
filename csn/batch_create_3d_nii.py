from pathlib import Path
from typing import Optional
import SimpleITK as sitk
from config import HOLDOUT_DATA_DIR

def resize_3d_image(
    image: sitk.Image, 
    new_size: tuple, 
    method: int = sitk.sitkLinear
) -> sitk.Image:
    """Resample 3D image to target size while preserving spatial information."""
    resampler = sitk.ResampleImageFilter()
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    
    new_spacing = [
        original_spacing[0] * original_size[0] / new_size[0],
        original_spacing[1] * original_size[1] / new_size[1],
        original_spacing[2] * original_size[2] / new_size[2]
    ]
    
    resampler.SetReferenceImage(image)
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(method)
    
    return resampler.Execute(image)

def process_nii_file(
    input_path: Path,
    output_path: Path,
    reference_image: sitk.Image,
    new_size: tuple = (512, 512, 512)
) -> None:
    """Process and save NIfTI file with consistent spatial attributes."""
    image = sitk.ReadImage(str(input_path))
    resampled = resize_3d_image(image, new_size, sitk.sitkNearestNeighbor)
    resampled.SetOrigin(reference_image.GetOrigin())
    resampled.SetSpacing((1, 1, 1))
    resampled.SetDirection(reference_image.GetDirection())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(resampled, str(output_path))

def create_matched_hq_lq_nii(
    data_type: str = 'pkufh4', 
    output_dir: Optional[Path] = None
) -> None:
    """Batch create matched HQ/LQ 3D NIfTI files with consistent processing."""
    input_dir = HOLDOUT_DATA_DIR / f'{data_type}/nii_checked'
    output_dir = output_dir or HOLDOUT_DATA_DIR / f'{data_type}/nii_checked_lq'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for subject in sorted(input_dir.iterdir()):
        pid = subject.stem
        print(f"Processing: {pid}")
        
        lq_path = subject / 'image.nii.gz'
        hq_path = subject / 'label.nii.gz'
        lq_img = sitk.ReadImage(str(lq_path))
        
        # Process LQ and HQ images
        process_nii_file(
            lq_path,
            output_dir / pid / 'image.nii.gz',
            lq_img
        )
        process_nii_file(
            hq_path,
            output_dir / pid / 'label.nii.gz',
            lq_img
        )

if __name__ == '__main__':
    create_matched_hq_lq_nii()
