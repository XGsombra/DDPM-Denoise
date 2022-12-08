import skimage.io as io
import numpy as np
import scipy.io as sio
import unprocess, process from processing

def realistic(img):
    # add realistic noise to the image, by doing the inverse of ISP, turned to Bayer pattern, add noise, and process ISP again
    raw, features = unprocess(np.array(img))
    noisy_raw = add_noise(raw)
    noisy = process(np.array([noisy_raw]), np.array([features['red_gain']]), np.array([features['blue_gain']]), np.array([features['cam2rgb']]))
    return noisy

if __name__  == __main__:
    clean = io.imread(f'samples/clean/4.jpg').astype(float)/255

    noisy = realistic(clean)
    plt.imsave(f'samples/noisy/realistic/unprocess.jpg', np.array(noisy[0]).astype(float))
