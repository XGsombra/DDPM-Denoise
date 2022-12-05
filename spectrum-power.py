# This implementation is adapted from
# https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
# Posted by Bert Vandenbroucke
# Professional astronomer.
import cv2
import matplotlib.image as mpimg
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

image_high = mpimg.imread("samples/clean/validation/5.jpg") / 255
image_low = mpimg.imread("samples/clean/validation/9.jpg") / 255

def get_spectrum_power(image):
    image = cv2.cvtColor(image.astype('float32'), cv2.COLOR_RGB2GRAY)
    npix = image_high.shape[0]

    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)**2

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic = "mean",
                                         bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    return kvals, Abins

kvals_low, Abins_low = get_spectrum_power(image_low)
kvals_high, Abins_high = get_spectrum_power(image_high)

plt.loglog(kvals_low, Abins_low, label="low")
plt.loglog(kvals_high, Abins_high, label="high")
plt.grid(True)
plt.legend()
plt.xlabel("$k$")
plt.ylabel("$P(k)$")
plt.tight_layout()
plt.show()
plt.savefig("image_frequency.png", dpi=300, bbox_inches = "tight")