"""
salience_trans.py (extension of trans.py)

Bottom-up, category-agnostic saliency pipeline. Fixation points are chosen by
local orientation-energy variance computed from a bank of Gabor filters
(a model of V1 simple-cell receptive fields). This is NOT face-specific: it
finds regions of high local structure/contrast and therefore generalises to
faces, houses and objects alike.

SaliencePipeline:
    rotates, foveates and log-polar transforms around each salient point (LP),
    and also returns the plain foveated crop (CNN).
"""

import cv2
import torch
import torchvision.transforms.functional as TF
import numpy as np
from trans import LogPolar, Rotate, Foveate


class SaliencePipeline(torch.nn.Module):
    def __init__(self, type='train', device='cpu', logpolar=True, crop_size=180,
                 output_shape=(180, 180), num_salient_points=4):
        """
        Args:
            type (str): 'train'  -> random rotation augmentation
                        'test'   -> inverted (180 deg) presentation
                        other    -> identity (upright presentation, e.g. 'valid')
            device (str): torch device
            logpolar (bool): kept for API compatibility; pipeline always returns both LP and CNN
            crop_size (int): crop size for both LP and CNN
            output_shape (tuple): output shape for the log-polar transform
            num_salient_points (int): number of fixations per base image
        """
        super().__init__()
        self.num_salient_points = num_salient_points
        self.device = device
        self.type = type
        self.crop_size = crop_size

        if type == 'train':
            self.rotate = Rotate()
        elif type == 'test':
            self.rotate = Rotate(invert=True)
        else:
            self.rotate = torch.nn.Identity()

        self.foveate = Foveate(crop_size=crop_size)
        self.logpolar = LogPolar(input_shape=(crop_size, crop_size),
                                 output_shape=output_shape, device=device)
        self.gaborlogpolar = LogPolar(input_shape=(224, 224),
                                      output_shape=(224, 224), device=device)

        self.kernels = self.get_kernels().to(device)

    # Bank of Gabor filters at multiple orientations and scales (V1-like).
    def get_kernels(self):
        kernels = []
        size = (31, 31)
        lambd = [4.0, 8.0]
        sigma = [0.5 * l for l in lambd]
        psi = [0.0, np.pi / 2]  # real + imaginary (quadrature) pair
        theta = [0.0, 1 * np.pi / 4, 2 * np.pi / 4, 3 * np.pi / 4]
        gamma = 0.5

        for p in range(len(psi)):
            for l in range(len(lambd)):
                for t in range(len(theta)):
                    kernels.append(cv2.getGaborKernel(size, sigma[l], theta[t], lambd[l], gamma, psi[p], ktype=cv2.CV_32F))

        self.num_kernels = len(kernels)
        stacked_filters = torch.from_numpy(np.stack(kernels)).unsqueeze(1)

        filters = torch.nn.Conv2d(in_channels=1, out_channels=self.num_kernels,
                                  kernel_size=31, padding='same', bias=False)
        filters.weight.data = stacked_filters
        filters.weight.requires_grad_(False)

        return filters

    def sample_salience_points(self, img, center=None):
        img = img.to(self.device)
        B, _, H, W = img.shape

        img = TF.rgb_to_grayscale(img, num_output_channels=1)

        # Gaussian attention prior centered on the image.
        if center is None:
            center_x = W / 2 - 0.5
            center_y = H / 2 - 0.5
        else:
            center_x, center_y = center

        x = torch.arange(0, W, device=self.device)
        y = torch.arange(0, H, device=self.device)
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')

        # -----------------------------------------------------
        # Hyperparameters of the Gaussian attention prior
        # -----------------------------------------------------
        alpha = 2       # sharpness of the edge drop-off
        sigma = W / 4   # spatial range of attention

        gaussian_mask = torch.exp(
            -((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / (2 * sigma ** 2)
        ) ** alpha

        gaussian_mask = gaussian_mask.repeat(B, 1, 1).unsqueeze(1)
        weighted_img = gaussian_mask * img

        weighted_img, xMap, yMap = self.gaborlogpolar.forwardReturnMapping(weighted_img)

        # normalize
        fmean = torch.mean(weighted_img, dim=(2, 3), keepdim=True)
        fstd = torch.std(weighted_img, dim=(2, 3), keepdim=True)
        weighted_img = (weighted_img - fmean) / (fstd + 1e-9)

        # apply gabor filters
        with torch.no_grad():
            filtered = self.kernels(weighted_img)
        _, _, _, W = filtered.shape

        # mask out boundaries
        filtered[..., :10, :] = 0
        filtered[..., -10:, :] = 0
        filtered[..., :, :10] = 0
        filtered[..., :, -10:] = 0

        # magnitude = sqrt(real**2 + imaginary**2)
        num_pairs = filtered.shape[1] // 2
        filtered = torch.sqrt(filtered[:, :num_pairs] ** 2 + filtered[:, num_pairs:] ** 2 + 1e-9)

        # variance across orientation channels -> saliency
        variance = torch.var(filtered, dim=1)

        # sample top salient points proportional to variance
        coords = torch.zeros(B, self.num_salient_points, 2, device=self.device, dtype=torch.long)

        for b in range(B):
            flat = variance[b].flatten()
            idx = torch.multinomial(flat, self.num_salient_points, replacement=False)
            ys = idx // W
            xs = idx % W
            y_actual = yMap[ys, xs]
            x_actual = xMap[ys, xs]
            coords[b] = torch.stack([x_actual, y_actual], dim=-1)  # [num_points, 2]
        return coords  # [B, num_points, 2]

    def forward(self, img):
        assert isinstance(img, torch.Tensor), f"Expected Tensor, got {type(img)}."

        img = img.to(self.device)
        B, C, H, W = img.shape
        transformed_imgs = torch.zeros((B, self.num_salient_points, C, self.crop_size, self.crop_size), device=self.device)

        salient_points = self.sample_salience_points(img)

        # crop, centered on each fixation point
        for b in range(B):
            for salient_idx, center in enumerate(salient_points[b]):
                transformed_imgs[b, salient_idx] = TF.crop(
                    img[b],
                    top=center[1] - self.crop_size // 2,
                    left=center[0] - self.crop_size // 2,
                    height=self.crop_size, width=self.crop_size)

        transformed_imgs = transformed_imgs.flatten(0, 1)  # (B*N, C, H, W)
        transformed_imgs = self.rotate(transformed_imgs)

        transformed_imgs = self.foveate(transformed_imgs)
        transformed_imgs_cnn = transformed_imgs.clone()
        transformed_imgs = self.logpolar(transformed_imgs)

        transformed_imgs = transformed_imgs.unflatten(0, (B, self.num_salient_points))
        transformed_imgs_cnn = transformed_imgs_cnn.unflatten(0, (B, self.num_salient_points))

        return transformed_imgs, transformed_imgs_cnn  # lp, cnn


class ImageNetAugment(torch.nn.Module):
    """Standard ImageNet-recipe augmentations — random resized crop (scale
    jitter), color jitter, random erasing — implemented batched with
    per-sample parameters so they run on the GPU inside OnTheFlyTransform.

    Applied to the raw fixation crop BEFORE rotation/foveation/log-polar, so
    the same augmentations serve both the lp and cnn variants. Expects float
    crops in [0, 1], shape (B, C, H, W). hflip lives in OnTheFlyTransform.
    """

    def __init__(self, scale=(0.6, 1.0), brightness=0.4, contrast=0.4,
                 saturation=0.4, erase_p=0.25, erase_scale=(0.02, 0.15),
                 erase_ratio=(0.3, 3.3)):
        super().__init__()
        self.scale = scale
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.erase_p = erase_p
        self.erase_scale = erase_scale
        self.erase_ratio = erase_ratio

    def _random_resized_crop(self, x):
        # per-sample zoom + shift via a batched affine grid (the fixation
        # crop is already a "crop", so the scale range is gentler than the
        # torchvision default of (0.08, 1.0))
        B = x.size(0)
        dev = x.device
        side = torch.empty(B, device=dev).uniform_(*self.scale).sqrt()
        tx = (1 - side) * (torch.rand(B, device=dev) * 2 - 1)
        ty = (1 - side) * (torch.rand(B, device=dev) * 2 - 1)
        theta = torch.zeros(B, 2, 3, device=dev, dtype=x.dtype)
        theta[:, 0, 0] = side
        theta[:, 1, 1] = side
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        grid = torch.nn.functional.affine_grid(theta, list(x.shape), align_corners=False)
        return torch.nn.functional.grid_sample(x, grid, mode='bilinear',
                                               padding_mode='reflection',
                                               align_corners=False)

    def _color_jitter(self, x):
        shape = (x.size(0), 1, 1, 1)
        dev = x.device
        if self.brightness > 0:
            f = 1 + (torch.rand(shape, device=dev) * 2 - 1) * self.brightness
            x = x * f
        if self.contrast > 0:
            f = 1 + (torch.rand(shape, device=dev) * 2 - 1) * self.contrast
            mean = TF.rgb_to_grayscale(x).mean(dim=(2, 3), keepdim=True)
            x = (x - mean) * f + mean
        if self.saturation > 0:
            f = 1 + (torch.rand(shape, device=dev) * 2 - 1) * self.saturation
            gray = TF.rgb_to_grayscale(x)
            x = (x - gray) * f + gray
        return x.clamp_(0, 1)

    def _random_erase(self, x):
        B, C, H, W = x.shape
        hit = torch.rand(B, device=x.device) < self.erase_p
        for b in hit.nonzero(as_tuple=True)[0].tolist():
            area = np.random.uniform(*self.erase_scale) * H * W
            log_ratio = np.random.uniform(np.log(self.erase_ratio[0]),
                                          np.log(self.erase_ratio[1]))
            ratio = np.exp(log_ratio)
            eh = int(round(np.sqrt(area * ratio)))
            ew = int(round(np.sqrt(area / ratio)))
            if 0 < eh < H and 0 < ew < W:
                top = np.random.randint(0, H - eh)
                left = np.random.randint(0, W - ew)
                x[b, :, top:top + eh, left:left + ew] = torch.rand(
                    C, eh, ew, device=x.device)
        return x

    def forward(self, x):
        x = self._random_resized_crop(x)
        x = self._color_jitter(x)
        x = self._random_erase(x)
        return x


class OnTheFlyTransform(torch.nn.Module):
    """The tail of SaliencePipeline (rotate -> foveate -> [logpolar]), applied
    batched on the GPU at train time to raw 180x180 fixation crops produced by
    datasets.make_packed_datasets. Replaces reading pre-rendered crop PNGs.

    Args:
        type (str): 'train' -> random-rotation augmentation (fresh every epoch,
                    unlike the baked-in rotation of preprocess.py)
                    'test' / 'inverted' -> 180-degree rotation
                    anything else -> upright
        variant (str): 'lp' applies the log-polar transform, 'cnn' skips it
    """

    def __init__(self, type='train', variant='lp', device='cpu',
                 crop_size=180, output_shape=(180, 180), hflip_p=0.5,
                 imagenet_aug=False):
        super().__init__()
        self.variant = variant
        self.augment = ImageNetAugment() if (type == 'train' and imagenet_aug) else None
        # horizontal (left-right) flip only, and only at train time: faces/objects
        # are ~bilaterally symmetric so this is a free augmentation, but a
        # vertical flip would fake the inversion manipulation the whole
        # pipeline exists to test, so it is never applied here.
        self.hflip_p = hflip_p if type == 'train' else 0.0
        if type == 'train':
            self.rotate = Rotate()
        elif type in ('test', 'inverted'):
            self.rotate = Rotate(invert=True)
        else:
            self.rotate = torch.nn.Identity()
        self.foveate = Foveate(crop_size=crop_size)
        self.logpolar = LogPolar(input_shape=(crop_size, crop_size),
                                 output_shape=output_shape, device=device)

    def forward(self, crops):
        """crops: (B, C, H, W) uint8 [0,255] or float [0,1] -> float (B, C, H, W)."""
        if crops.dtype == torch.uint8:
            crops = crops.float().div_(255.0)
        if self.hflip_p > 0:
            flip_mask = torch.rand(crops.size(0), device=crops.device) < self.hflip_p
            if flip_mask.any():
                crops = crops.clone()
                crops[flip_mask] = crops[flip_mask].flip(-1)
        if self.augment is not None:
            crops = self.augment(crops)
        crops = self.rotate(crops)
        crops = self.foveate(crops)
        if self.variant == 'lp':
            crops = self.logpolar(crops)
        return crops
