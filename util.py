import matplotlib.pyplot as plt
import numpy as np
import os
from natsort import natsorted
from glob import glob
from constants import CHANNEL_NUM, SIZE, CLEAN_DIR, LEVELS, MPRNET_DENOISED_DIR
from psnr_hvsm import psnr_hvsm
from imageio import imread
from skimage.metrics import structural_similarity as compare_ssim
import cv2 as cv


def calc_psnr_hvsm(denoised, clean):
    # Pass the luminance component of images to compare the psnr-hvs and psnr-hvsm
    y_denoised = cv.cvtColor(denoised.astype('float32'), cv.COLOR_RGB2YUV)[..., 0]
    y_clean = cv.cvtColor(clean.astype('float32'), cv.COLOR_RGB2YUV)[..., 0]
    return psnr_hvsm(y_clean, y_denoised)

def calc_ssim(denoised, clean):
    # Convert to grayscale first
    return compare_ssim(denoised, clean, channel_axis=2)

def calc_psnrs(denoised_path):
    """Calculate the psnr-hvsm of the denoised image in the denoised_path."""
    # Load the ground truth
    clean_image_paths = natsorted(glob(os.path.join(CLEAN_DIR, '*.jpg')))
    image_num = len(clean_image_paths)
    clean_images = np.zeros((image_num, SIZE, SIZE, CHANNEL_NUM))
    for i in range(image_num):
        clean_images[i] = imread(clean_image_paths[i]).astype(float) / 255

    # Load the denoised image and calculate the psnr-hvs and psnr-hvsm
    psnr_hvsm_avgs = {}
    for noise_type in ['g', 'p']:
        psnr_hvsm_avgs_lst = []
        for level in LEVELS:
            psnr_hvsm_total = 0
            denoised_images = np.zeros_like(clean_images)
            for i in range(image_num):
                denoised_images[i] = imread(
                    os.path.join(denoised_path, f"{i}-{round(level, 2)}-{noise_type}.png")
                ).astype(float) / 255
                psnr_hvsm_total += calc_psnr_hvsm(denoised_images[i], clean_images[i])
            psnr_hvsm_avgs_lst.append(round(psnr_hvsm_total / image_num, 2))
        psnr_hvsm_avgs[noise_type] = psnr_hvsm_avgs_lst
    return psnr_hvsm_avgs

def calc_ssims(denoised_path):
    """Calculate the ssim of the denoised image in the denoised_path."""
    # Load the ground truth
    clean_image_paths = natsorted(glob(os.path.join(CLEAN_DIR, '*.jpg')))
    image_num = len(clean_image_paths)

    # Load the denoised image and calculate the ssim
    ssim_avgs = {}
    for noise_type in ['g', 'p']:
        ssim_avgs_lst = []
        for level in LEVELS:
            ssim_total = 0
            for i in range(image_num):
                clean_images = imread(clean_image_paths[i]).astype(float) / 255
                denoised_images = imread(
                    os.path.join(denoised_path, f"{i}-{round(level, 2)}-{noise_type}.png")
                ).astype(float) / 255
                ssim = calc_ssim(denoised_images, clean_images)
                ssim_total += ssim
            ssim_avgs_lst.append(round(ssim_total / image_num, 2))
        ssim_avgs[noise_type] = ssim_avgs_lst
    return ssim_avgs


if __name__ == "__main__":
    print(calc_psnrs(MPRNET_DENOISED_DIR))
    # hvsm - {'g': [25.41, 21.05, 16.65, 12.69, 9.54], 'p': [18.65, 15.31, 11.26, 7.39, 4.46]}
    print(calc_ssims(MPRNET_DENOISED_DIR))
    # ssim - {'g': [0.85, 0.77, 0.68, 0.58, 0.5], 'p': [0.74, 0.68, 0.62, 0.54, 0.47]}

    # sigma   curr_step     hvsm      sism
    # 0.05       15         33.54
    # 0.1        35         27.71
    # 0.2        70         22.52     0.78
    # 0.4        150
