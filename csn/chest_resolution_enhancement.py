from typing import Optional, List
import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *
from torchvision.models import vgg16_bn
import nibabel as nib
import pandas as pd
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import config

def open_nifti2d_image(fn: Path) -> Image:
    """Load and normalize 2D NIfTI image with intensity clipping."""
    nii_img = nib.load(str(fn))
    image = np.asanyarray(nii_img.dataobj)
    image = np.clip(image, -1024, 3072)  # CT intensity range
    image_norm = (image + 1024) / 4096   # Normalize to [0,1]
    return Image(pil2tensor(image_norm, np.float32))

class Nifti2dImageList(ImageList):
    """Custom ImageList for processing 2D NIfTI images."""
    def open(self, fn: Path) -> Image:
        return open_nifti2d_image(fn)

class FeatureLoss(nn.Module):
    """Perceptual loss combining VGG16 features and Gram matrices."""
    def __init__(self, layer_wgts: List[float] = [5, 15, 2]):
        super().__init__()
        self.m_feat = vgg16_bn(True).features.cuda().eval()
        requires_grad(self.m_feat, False)
        blocks = [i-1 for i,o in enumerate(children(self.m_feat)) 
                 if isinstance(o, nn.MaxPool2d)]
        self.loss_features = [self.m_feat[i] for i in blocks[2:5]]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.base_loss = F.l1_loss

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute perceptual loss between input and target images."""
        out_feat = self._extract_features(target, clone=True)
        in_feat = self._extract_features(input)
        losses = [self.base_loss(input, target)]
        
        for f_in, f_out, w in zip(in_feat, out_feat, self.wgts):
            losses.append(self.base_loss(f_in, f_out) * w)
            losses.append(self._gram_loss(f_in, f_out) * w**2 * 5e3)
            
        return sum(losses)

    def _extract_features(self, x: Tensor, clone: bool = False) -> List[Tensor]:
        """Extract VGG16 features from input tensor."""
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def _gram_loss(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute Gram matrix loss between feature maps."""
        return self.base_loss(self._gram_matrix(x), self._gram_matrix(y))
    
    def _gram_matrix(self, x: Tensor) -> Tensor:
        """Compute Gram matrix for feature maps."""
        n, c, h, w = x.size()
        return (x.view(n, c, -1) @ x.view(n, c, -1).transpose(1,2))/(c*h*w)

def enhance_ct_resolution(
    data_path: Optional[Path] = None,
    model_path: Optional[Path] = None
) -> List[Image]:
    """Enhance CT resolution using deep learning."""
    data_path = data_path or config.DATA_PATH
    df_train = pd.read_csv(data_path/config.TRAIN_CSV)
    df_valid = pd.read_csv(data_path/config.VALID_CSV)
    df_test = pd.read_csv(data_path/config.TEST_CSV)
    
    # Prepare data
    src = (Nifti2dImageList.from_df(df_train, data_path, cols=['LQ'])
           .split_from_df(col='Cat')
           .label_from_func(lambda x: str(x).replace('/LQ/','/HQ/')))
    
    data = (src.transform(get_transforms(), size=256)
            .databunch(bs=8)
            .normalize(imagenet_stats))
    
    # Train model
    learn = unet_learner(
        data, models.resnet34,
        wd=config.wd,
        loss_func=FeatureLoss(),
        callback_fns=LossMetrics,
        blur=True,
        self_attention=True
    )
    
    if model_path:
        learn.load(model_path)
    
    learn.fit_one_cycle(10, config.lr)
    
    # Make predictions
    return [learn.predict(open_nifti2d_image(data_path/p))[0] 
            for p in df_test['LQ']]

if __name__ == '__main__':
    enhance_ct_resolution()
