import cv2 as cv
from albumentations.core.transforms_interface import ImageOnlyTransform


class MinMaxBrightness(ImageOnlyTransform):
    def apply(self, img_rgb, **params):
        img_hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
        img_hsv[:, :, 2] = self.min_max_scaler(
            img_hsv[:, :, 2], min=0, max=255)
        img_rgb_scaled = cv.cvtColor(img_hsv, cv.COLOR_HSV2RGB)
        return img_rgb_scaled

    def min_max_scaler(self, img, min=0, max=255):
        img_std = (img - img.min()) / (img.max() - img.min())
        img_scaled = img_std * (max - min) + min
        return img_scaled