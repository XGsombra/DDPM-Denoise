import numpy as np
import os
import skimage.io as io
from natsort import natsorted
from glob import glob
from constants import CHANNEL_NUM, SIZE, CLEAN_DIR, LEVELS, MPRNET_DENOISED_DIR


def calc_psnr(x, gt):
    out = 10 * np.log10(1 / ((x - gt) ** 2).mean().item())
    return out


def calc_psnrs(denoised_path):
    clean_image_paths = natsorted(glob(os.path.join(CLEAN_DIR, '*.jpg')))
    image_num = len(clean_image_paths)
    clean_images = np.zeros((image_num, SIZE, SIZE, CHANNEL_NUM))
    for i in range(image_num):
        clean_images[i] = io.imread(clean_image_paths[i]).astype(float) / 255

    for noise_type in ['p', 'g']:
        for level in LEVELS:
            denoised_images = np.zeros_like(clean_images)
            for i in range(image_num):
                denoised_images[i] = io.imread(
                    os.path.join(denoised_path, f"{i}-{round(level, 2)}-{noise_type}.jpg")
                ).astype(float) / 255
            print(calc_psnr(denoised_images, clean_images))


if __name__ == "__main__":
    calc_psnrs(MPRNET_DENOISED_DIR)
