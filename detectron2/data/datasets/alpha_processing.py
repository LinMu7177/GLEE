import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Define the necessary transforms
clip_standard_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), interpolation=Image.BICUBIC),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    ),
])

mask_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5], std=[0.26])
])

def process_image_and_mask(image, mask):
    """
    Processes the input image and mask to produce image_torch and mask_torch.

    Args:
        image (numpy.ndarray): The input image array of shape (H, W, 3), values in [0, 255].
        mask (numpy.ndarray): The input mask array of shape (H, W), values in [0, 1].

    Returns:
        image_torch (torch.Tensor): The transformed image tensor.
        mask_torch (torch.Tensor): The transformed mask tensor.
    """
    # Ensure image and mask have the same height and width
    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask must have the same height and width.")

    # Combine image and mask into a single array
    rgba = np.concatenate((image, mask[..., np.newaxis]), axis=-1)

    # Separate the image and mask
    rgb = rgba[:, :, :3]
    mask = rgba[:, :, 3]

    # Convert arrays to PIL Images for transformations
    # rgb_pil = Image.fromarray(rgb.astype('uint8'), mode='RGB')
    # mask_pil = Image.fromarray(mask.astype('uint8'), mode='L')

    # Apply the transformations
    image_torch = clip_standard_transform(rgb)
    mask_torch = mask_transform(mask)

    return image_torch, mask_torch
