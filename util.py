import matplotlib.pyplot as plt
import numpy as np
import os
from natsort import natsorted
from glob import glob
from constants import CHANNEL_NUM, SIZE, CLEAN_DIR, LEVELS, MPRNET_DENOISED_DIR
from psnr_hvsm import psnr_hvs_hvsm
from imageio import imread
import cv2 as cv


def calc_psnr_hvs_hvsm(denoised, clean):
    # Pass the luminance component of images to compare the psnr-hvs and psnr-hvsm
    y_denoised = cv.cvtColor(denoised .astype('float32'), cv.COLOR_RGB2YUV)[..., 0]
    y_clean = cv.cvtColor(clean .astype('float32'), cv.COLOR_RGB2YUV)[..., 0]
    return psnr_hvs_hvsm(y_clean, y_denoised)

def calc_psnrs(denoised_path):
    """Calculate the psnr-hvs and psnr-hvsm of the denoised image in the denoised_path."""
    # Load the ground truth
    clean_image_paths = natsorted(glob(os.path.join(CLEAN_DIR, '*.jpg')))
    image_num = len(clean_image_paths)
    clean_images = np.zeros((image_num, SIZE, SIZE, CHANNEL_NUM))
    for i in range(image_num):
        clean_images[i] = imread(clean_image_paths[i]).astype(float) / 255

    # Load the denoised image and calculate the psnr-hvs and psnr-hvsm
    psnr_hvs_avgs = {}
    psnr_hvsm_avgs = {}
    for noise_type in ['g', 'p']:
        psnr_hvs_avgs_lst = []
        psnr_hvsm_avgs_lst = []
        for level in LEVELS:
            psnr_hvs_total = 0
            psnr_hvsm_total = 0
            denoised_images = np.zeros_like(clean_images)
            for i in range(image_num):
                denoised_images[i] = imread(
                    os.path.join(denoised_path, f"{i}-{round(level, 2)}-{noise_type}.png")
                ).astype(float) / 255
                psnr_hvs, psnr_hvsm = calc_psnr_hvs_hvsm(denoised_images[i], clean_images[i])
                psnr_hvs_total += psnr_hvs
                psnr_hvsm_total += psnr_hvsm
            psnr_hvs_avgs_lst.append(round(psnr_hvs_total / image_num, 2))
            psnr_hvsm_avgs_lst.append(round(psnr_hvsm_total / image_num, 2))
        psnr_hvs_avgs[noise_type] = psnr_hvs_avgs_lst
        psnr_hvsm_avgs[noise_type] = psnr_hvsm_avgs_lst
    return psnr_hvs_avgs, psnr_hvsm_avgs


if __name__ == "__main__":
    print(calc_psnrs(MPRNET_DENOISED_DIR))
