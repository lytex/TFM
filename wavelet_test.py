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
t = np.linspace(0, 1, 2001, endpoint=False)
signal = np.cos(2 * np.pi * 7 * t) + 0.1*np.random.randn(len(t)) + 7*(t>0.5) - 7*(t>0.75) \
+ np.cos(2 * np.pi * 18 * 1/(t+1e-16))*np.exp(-0.5*t**2) + 0.5*np.cos(2 * np.pi * 18 * t**4)
signal = np.mean([np.cos(2* np.pi * (4+x) * t) for x in np.linspace(0, 2, 21)], axis=0) + 0.2*np.random.randn(len(t))
signal = np.sin(50*t)*((t >= 0) & (t < 0.25)) + np.sin(50*t)*((t >= 0.25) & (t < 0.5)) + np.sin(220*t)*((t >= 0.5) & (t < 0.75)) + np.sin(230*t)*((t > 0.75) & (t <= 1))

max_level = 10
fig, ax = plt.subplots(len(coeffs) + 1, 1, figsize=(10, 12*max_level//5))
ax[0].plot(t, signal)
ax[0].set_title('Original Signal')

# t = np.linspace(0, 1, 201, endpoint=False)
signal = np.sin(50*t)*((t >= 0) & (t < 0.25)) + np.sin(50*t)*((t >= 0.25) & (t < 0.5)) + np.sin(220*t)*((t >= 0.5) & (t < 0.75)) + np.sin(230*t)*((t > 0.75) & (t <= 1))
# + 0.2*np.random.randn(len(t))
# signal = np.sin(200*t)/(200*t+1e-16)
# signal = 1-0.2*np.random.rand(len(t))+2*(t-0.5)**2

# Perform the Discrete Wavelet Transform (DWT)
wavelet = 'db10'  # Daubechies wavelet with 4 vanishing moments
coeffs = []
for i in range(1, max_level+1):
    coeffs.append(pywt.wavedec(signal, wavelet, level=i, mode='symmetric')[0])
# Plot the original signal and its DWT coefficients

for i, coeff in enumerate(coeffs):
    ax[i + 1].plot(coeff)
    ax[i + 1].set_title(f'Level {i+1} DWT Coefficients')

plt.tight_layout()
plt.show()

# coeff_arr = np.array(coeff)
# nrm = mpl.colors.Normalize(min(coeff_arr.min(), -coeff_arr.max()), max(coeff_arr.max(), -coeff_arr.min()))
# plt.figure(figsize=(10, 6))

# for i, ci in enumerate(coeffs):
#     plt.imshow(ci.reshape(1, -1), extent=[0, 1000, i + 0.5, i + 1.5], aspect='auto', interpolation='nearest', cmap=cm.RdBu, norm=nrm)

# plt.ylim(0.5, len(coeffs) + 0.5) # set the y-lim to include the six horizontal images

# plt.show()

# plt.colorbar(label='Coefficient Magnitude')
# plt.xlabel('Time')
# plt.ylabel('Scale (Level)')
# plt.yscale('log', base=2)
# plt.gca().invert_yaxis()
# plt.title('DWT Coefficients')
# plt.show()


# %%
list(zip(pywt.families(short=False), pywt.families(short=True)))
pywt.wavelist(kind='discrete')
# db, sym

# %%
import pywt
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# Choose a specific wavelet name (e.g., 'db4' for Daubechies 4 wavelet)
level = 1
for wavelet_name in sorted([f"sym{N}" for N in range(2, 10)] + [f"db{N}" for N in range(1, 10)], key=lambda x: x[-1]):
    
    # Initialize the wavelet object
    wavelet = pywt.Wavelet(wavelet_name)
    
    # Define levels for wavelet decomposition
    levels = [1, 2, 3, 4, 5, 6]
    
    plt.figure(figsize=(14, 5))
    
    # Plot scaling and wavelet functions for each level
    phi, psi, x = wavelet.wavefun(level=15)
    
    # Plot the scaling function (phi)
    ax = plt.subplot(2, 2, 2 * 1 + 1)
    plt.plot(x, phi, label='')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.title(f'{wavelet_name} Scaling Function')
    ax.fill_between(x, 0, phi, where=phi>0, color="blue")
    ax.fill_between(x, 0, phi, where=phi<0, color="red")
    # plt.legend()
    
    # Plot the wavelet function (psi)
    ax = plt.subplot(2, 2, 2 * 1 + 2)
    fig = plt.plot(x, psi, label='')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.title(f'{wavelet_name} Wavelet Function')
    ax.fill_between(x, 0, psi, where=psi>0, color="blue")
    ax.fill_between(x, 0, psi, where=psi<0, color="red")
    # plt.legend()
    
    plt.tight_layout()
    plt.show()
    ax.get_figure().savefig("plot/wavelets/"+wavelet_name+".png")

    plt.figure(figsize=(14, 5))

    # Plot the scaling function (phi)
    ax = plt.subplot(2, 2, 2 * 1 + 1)
    plt.plot(x, phi**2, label='')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.title(f'{wavelet_name} Scaling Function')
    ax.fill_between(x, 0, phi**2, where=phi**2>0, color="blue")
    ax.fill_between(x, 0, phi**2, where=phi**2<0, color="red")
    # plt.legend()

    
    # Plot the wavelet function (psi)
    ax = plt.subplot(2, 2, 2 * 1 + 2)
    fig = plt.plot(x, psi**2, label='')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.title(f'{wavelet_name} Wavelet Function')
    ax.fill_between(x, 0, psi**2, where=psi**2>0, color="blue")
    ax.fill_between(x, 0, psi**2, where=psi**2<0, color="red")
    # plt.legend()
    
    plt.tight_layout()
    plt.show()
    ax.get_figure().savefig("plot/wavelets/sq_"+wavelet_name+".png")

# %%
f

# %%
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

x = np.arange(0.0, 2, 0.01)
y1 = np.sin(5 * np.pi * x)
y2 = y1**2

fig, ax1 = plt.subplots(1, 1)

ax1.fill_between(x, 0, y1, where=y1>0, color="blue")
ax1.fill_between(x, 0, y1, where=y1<0, color="red")
ax1.set_ylabel('between y1 and 0')
plt.savefig("plot/wavelets/sin.png")


fig, ax2 = plt.subplots(1, 1)
ax2.fill_between(x, 0, y2, where=y2>0, color="blue")
ax2.fill_between(x, 0, y2, where=y2<0, color="red")
ax2.set_ylabel('between y1 and 1')
plt.savefig("plot/wavelets/sq_sin.png")

# ax3.fill_between(x, y1, y2)
# ax3.set_ylabel('between y1 and y2')
# ax3.set_xlabel('x')

# %%
import pywt
import numpy as np
import matplotlib.pyplot as plt

# Define the wavelet
wavelet = 'sym5'

# Get the wavelet object
wav = pywt.Wavelet(wavelet)

# Define the number of points to plot
num_points = 1000

# Generate x-axis values
x = np.linspace(-5, 5, num_points)

# Create a figure with subplots
fig, axs = plt.subplots(4, 1, figsize=(10, 12))
fig.suptitle(f'{wavelet} Wavelet')

# Plot scaling function (phi)
phi, psi, x_values = wav.wavefun(level=8)
axs[0].plot(x_values, phi)
axs[0].set_title('Scaling function (phi)')
axs[0].set_ylim(-1, 1)

# Plot wavelet function (psi)
axs[1].plot(x_values, psi)
axs[1].set_title('Wavelet function (psi)')
axs[1].set_ylim(-1, 1)

# Plot approximation coefficients
for level in range(1, 5):
    coeffs = np.array(wav.dec_lo)[::-1] * (len(wav.dec_lo))**(level/2)
    x_level = np.linspace(-5, 5, len(coeffs))
    axs[2].plot(x_level, coeffs, label=f'Level {level}')
axs[2].set_title('Approximation coefficients')
axs[2].legend()

# Plot detail coefficients
for level in range(1, 5):
    coeffs = np.array(wav.dec_hi)[::-1] * (len(wav.dec_hi))**(level/2)
    x_level = np.linspace(-5, 5, len(coeffs))
    axs[3].plot(x_level, coeffs, label=f'Level {level}')
axs[3].set_title('Detail coefficients')
axs[3].legend()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
