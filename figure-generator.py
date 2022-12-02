import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

image_num = 5


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



# Image for PSNR and SSIM on each step
sigma = 0.1
noise_type = 'g'
clean = io.imread(f"./samples/clean/{img_id}.jpg").astype(float) / 255
noisy_img = io.imread(f"./samples/noisy/{img_id}-{sigma}-{noise_type}.jpg").astype(float) / 255
x = torch.Tensor([noisy_img.transpose([2, 0, 1])]).to(DEVICE)
curr_step = 35
total_step_count = 0

while curr_step > 0:
    total_step_count += 1
    x = diffusion.denoise(1, x=x, curr_step=curr_step, n_steps=1)[0, ...].cpu().detach().numpy().transpose([1,2,0])
    print(curr_step)
    print(calc_psnr_hvsm(x, clean))
    print(calc_ssim(x, clean))
    # plt.imshow(np.clip(x, a_min=0., a_max=1.))
    # plt.show()
    curr_step -= 1 # max(1, int(np.sqrt(curr_step)) // 2)
    # curr_step = int(curr_step / 1.1)
    x = torch.Tensor([x.transpose([2, 0, 1])])
print(f"total step is {total_step_count}")
plt.imshow(np.clip(x, a_min=0., a_max=1.)[0, ...].cpu().detach().numpy().transpose([1,2,0]))
plt.show()

