# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: TFM
#     language: python
#     name: tfm
# ---

# %%

# %%
import numpy as np
import pywt
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

# Example signal: a sinusoid with added noise
t = np.linspace(0, 1, 4001, endpoint=False)
signal = np.cos(2 * np.pi * 7 * t) + 0.1*np.random.randn(len(t)) 
# + np.cos(2 * np.pi * 18 * 1/t)*np.exp(-0.5*t**2) + 0.5*np.cos(2 * np.pi * 18 * t**4)

# Perform the Discrete Wavelet Transform (DWT)
wavelet = 'sym5'  # Daubechies wavelet with 4 vanishing moments
coeffs = pywt.wavedec(signal, wavelet, level=3)

# Plot the original signal and its DWT coefficients
fig, ax = plt.subplots(len(coeffs) + 1, 1, figsize=(10, 12))
ax[0].plot(t, signal)
ax[0].set_title('Original Signal')

for i, coeff in enumerate(coeffs):
    ax[i + 1].plot(coeff)
    ax[i + 1].set_title(f'Level {i} DWT Coefficients')

plt.tight_layout()
plt.show()

coeff_arr = np.array(coeff)
nrm = mpl.colors.Normalize(min(coeff_arr.min(), -coeff_arr.max()), max(coeff_arr.max(), -coeff_arr.min()))
plt.figure(figsize=(10, 6))

for i, ci in enumerate(coeffs):
    plt.imshow(ci.reshape(1, -1), extent=[0, 1000, i + 0.5, i + 1.5], aspect='auto', interpolation='nearest', cmap=cm.RdBu, norm=nrm)

plt.ylim(0.5, len(coeffs) + 0.5) # set the y-lim to include the six horizontal images

plt.show()

# plt.colorbar(label='Coefficient Magnitude')
# plt.xlabel('Time')
# plt.ylabel('Scale (Level)')
# plt.yscale('log', base=2)
# plt.gca().invert_yaxis()
# plt.title('DWT Coefficients')
# plt.show()


# %%
import pywt
import numpy as np
import matplotlib.pyplot as plt

# Choose a specific wavelet name (e.g., 'db4' for Daubechies 4 wavelet)
wavelet_name = 'haar'

# Initialize the wavelet object
wavelet = pywt.Wavelet(wavelet_name)

# Define levels for wavelet decomposition
levels = [1, 2, 3, 4, 5]

plt.figure(figsize=(14, 10))

# Plot scaling and wavelet functions for each level
for level in levels:
    # Compute the wavelet and scaling functions
    phi, psi, x = wavelet.wavefun(level=level)

    # Plot the scaling function (phi)
    plt.subplot(len(levels), 2, 2 * (level - 1) + 1)
    plt.plot(x, phi, label='')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.title(f'Level {level} - {wavelet_name} Scaling Function')
    # plt.legend()

    # Plot the wavelet function (psi)
    plt.subplot(len(levels), 2, 2 * (level - 1) + 2)
    plt.plot(x, psi, label='')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.title(f'Level {level} - {wavelet_name} Wavelet Function')
    # plt.legend()

plt.tight_layout()
plt.show()
