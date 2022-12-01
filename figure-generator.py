import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt


# Image for the introduction
clean = io.imread(f"./samples/clean/0.jpg").astype(float) / 255
noisy = io.imread(f"./samples/noisy/0-0.2-g.jpg").astype(float) / 255
ddpm_denoised = io.imread(
    f"./samples/ddpm-denoised/0-0.2-g.png").astype(float) / 255
mprnet_denoised = io.imread(
    f"./samples/mprnet-denoised/0-0.2-g.png").astype(float) / 255

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
axs[0, 0].imshow(clean)
axs[0, 0].set_title("Ground Truth")
axs[0, 0].set_axis_off()
axs[0, 1].imshow(noisy)
axs[0, 1].set_title("Noisy Image with sigma=0.2")
axs[0, 1].set_axis_off()
axs[1, 0].imshow(ddpm_denoised)
axs[1, 0].set_title("Denoised by DDPM")
axs[1, 0].set_axis_off()
axs[1, 1].imshow(mprnet_denoised)
axs[1, 1].set_title("Denoised by MPRNet")
axs[1, 1].set_axis_off()
plt.savefig("introduction.png", bbox_inches='tight')
