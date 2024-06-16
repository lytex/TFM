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
import numpy as np
import pywt
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

# Example signal: a sinusoid with added noise
t = np.linspace(0, 1, 400, endpoint=False)
signal = np.cos(2 * np.pi * 7 * t) + 0.1*np.random.randn(len(t))

# Perform the Discrete Wavelet Transform (DWT)
wavelet = 'haar'  # Daubechies wavelet with 4 vanishing moments
coeffs = pywt.wavedec(signal, wavelet, level=4)

# Plot the original signal and its DWT coefficients
fig, ax = plt.subplots(len(coeffs) + 1, 1, figsize=(10, 12))
ax[0].plot(t, signal)
ax[0].set_title('Original Signal')

for i, coeff in enumerate(coeffs):
    ax[i + 1].plot(coeff)
    ax[i + 1].set_title(f'Level {i} DWT Coefficients')

plt.tight_layout()
plt.show()

# Calculate the maximum length of the coefficients
max_len = max(len(coeff) for coeff in coeffs)

# Prepare an empty array to hold the coefficients
coeff_arr = np.zeros((len(coeffs), max_len))

# Fill the array with coefficients, aligning them to the left
# for i, coeff in enumerate(coeffs):
#     coeff_arr[i, :len(coeff)] = coeff

# Upsample coefficients so that each level is twice as large as the previous one
def upsample(coeff, factor):
    return np.repeat(coeff, factor)

# Fill the array with upsampled coefficients
for i, coeff in enumerate(reversed(coeffs)):
    # The upsampling factor is 2^i for the i-th level
    factor = 2 ** i
    upsampled_coeff = upsample(coeff, factor)

    # Ensure the length does not exceed the max_len
    if len(upsampled_coeff) > max_len:
        upsampled_coeff = upsampled_coeff[:max_len]
    coeff_arr[i, :len(upsampled_coeff)] = upsampled_coeff

# Reverse the order of the rows to have the finest scale at the bottom
coeff_arr = np.flipud(coeff_arr)

# Plot the coefficients using imshow
nrm = mpl.colors.Normalize(min(coeff_arr.min(), -coeff_arr.max()), max(coeff_arr.max(), -coeff_arr.min()))
plt.figure(figsize=(10, 6))
plt.imshow(coeff_arr, aspect='auto', interpolation='nearest', extent=[0, len(signal), 1, len(coeffs)], cmap=cm.RdBu, norm=nrm)
plt.colorbar(label='Coefficient Magnitude')
plt.xlabel('Time')
plt.ylabel('Scale (Level)')
plt.yscale('log', base=2)
plt.gca().invert_yaxis()
plt.title('DWT Coefficients')
plt.show()


# %%
import pywt
import numpy as np
import matplotlib.pyplot as plt

# Choose a specific wavelet name (e.g., 'db4' for Daubechies 4 wavelet)
wavelet_name = 'sym5'

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
