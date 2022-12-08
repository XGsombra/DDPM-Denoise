from time import process_time
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm
import numpy as np
from skimage.filters import gaussian

pics = [5,6,7,8,9,10]
sigmas = [0.05, 0.1, 0.2, 0.4]
dists = ['g']
params = {}
psnrs = {}
time = {}

for p in pics:
    # read clean image
    clean = io.imread(f'samples/clean/validation/{p}.jpg')
    for s in sigmas:
        for d in dists:
            best_img = np.zeros_like(clean)
            best_psnr = -2000
            best_sig = 0
            t = []
            sigs = []
            psnrs_cur = []
            
            # read noisy image
            fname = f'samples/noisy/validation/{p}-{s}-{d}.jpg'
            noisy = io.imread(fname)
            
            # tune for different parameters
            for sig in tqdm(range(0,150,10)):
                for size in [1,5,10,20,25,30]:
                    start = process_time()
                    bilateral = cv2.bilateralFilter(noisy, size, sig, sig)
                    end = process_time()
                    t.append(end - start)
                    
                    # calculate psnr, save the best one
                    psnr = peak_signal_noise_ratio(clean, bilateral)
                    psnrs_cur.append(psnr)
                    sigs.append(sig)

                    if psnr > best_psnr:
                        best_psnr = psnr
                        best_img = bilateral
                        best_sig = sig

            params[fname] = sigs
            psnrs[fname] = psnrs_cur
            time[fname] = t
            print(best_psnr)
            
            # save the denoised image with best psnr
            
            plt.imsave(f'samples/bilateral-denoised/{p}-{s}-{d}.jpg', best_img)
