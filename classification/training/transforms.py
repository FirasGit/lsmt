from torchvision import transforms as T
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transform(cfg=None, shape=None):
    mean = cfg.model.transforms.get('mean', None)
    std = cfg.model.transforms.get('std', None)
    size = cfg.model.transforms.get('img_size', None)
    preprocessing_transform_ = [
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, value=(0,0,0)),
    ]
    if mean and std:
        preprocessing_transform_.append(A.Normalize(mean=mean, std=std))
    if not cfg.meta.transforms:
        preprocessing_transform_.append(ToTensorV2())
    preprocessing_transform = A.Compose(
        preprocessing_transform_, additional_targets={"image1": 'image'})
    train_transform = A.Compose([
        A.Rotate(limit=45, p=0.5, border_mode=0, value=(-1,-1,-1)),
        A.Flip(p=0.5),
        ToTensorV2()
    ], additional_targets={'image1': 'image'})
    val_transform = A.Compose([
        ToTensorV2()
    ], additional_targets={'image1': 'image'})
    return preprocessing_transform, train_transform, val_transform
