import torch
import numpy as np

def preprocessing_image(image):
    np_image = np.array(image) / 255
    clipped_image = np.clip(np_image, 0, 1)
    srgb_image = np.where(clipped_image <= 0.04045, clipped_image / 12.92, ((clipped_image + 0.055) / 1.055) ** 2.4)
    h, w, c = srgb_image.shape
    linear_image = srgb_image.reshape(h * w, c)
    for pixel in linear_image:
        if pixel[3] == 0.0:
            pixel[0:3] = 0.0
    reshaped_image = linear_image.transpose().reshape(c, h, w)
    torch_image = torch.from_numpy(reshaped_image).float() * 2.0 - 1
    return torch_image


def postprocessing_image(image):
    image = image.detach().squeeze(0)
    reshaped_image = image.permute(1, 2, 0)
    np_image = reshaped_image.numpy()
    np_image = (np_image + 1) / 2
    srgb_image = np_image[..., :3]
    alpha_image = np_image[..., 3]
    clipped_image = np.clip(srgb_image, 0, 1)
    rgb_image = np.where(clipped_image <= 0.003130804953560372, clipped_image * 12.92, 1.055 * (clipped_image ** (1.0 / 2.4)) - 0.055)
    rgba_image = np.concatenate([rgb_image, alpha_image[..., np.newaxis]], axis=2)
    rgba_image = rgba_image * 255
    return rgba_image.astype(np.uint8)
