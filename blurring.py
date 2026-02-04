import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage import color, io  # Added 'io' for reading files

# ---------------------------------------------------------
# 1. Prepare the Image
# ---------------------------------------------------------
# Update this filename to match your actual file (e.g., 'img.jpg' or 'img.png')
filename = 'image.jpg'

try:
    # FIX 1: Read the file into an array first
    image_array = io.imread(filename)
except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Please check the name and extension.")
    exit()

# FIX 2: Handle RGBA images (remove the 4th Alpha channel if present)
if image_array.ndim == 3 and image_array.shape[2] == 4:
    image_array = image_array[:, :, :3]

# Convert to grayscale
image = color.rgb2gray(image_array)

# Ensure strictly float type for precision
image = image.astype(float)

# ---------------------------------------------------------
# 2. Create the Spatial Filter (Gaussian Kernel)
# ---------------------------------------------------------
def gaussian_kernel(size, sigma=1):
    k = np.linspace(-(size//2), size//2, size)
    x, y = np.meshgrid(k, k)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

kernel_size = 15
kernel = gaussian_kernel(kernel_size, sigma=3)

# ---------------------------------------------------------
# Method A: Spatial Domain Filtering (Convolution)
# ---------------------------------------------------------
# We use 'full' convolution. Output size will be (H_image + H_kernel - 1).
spatial_result = signal.convolve2d(image, kernel, mode='full')

# ---------------------------------------------------------
# Method B: Frequency Domain Filtering (Fourier)
# ---------------------------------------------------------
# The Convolution Theorem: Spatial Convolution == Frequency Multiplication.
# IMPORTANT: FFT uses circular convolution. To match Method A (linear convolution),
# we must Zero-Pad the inputs to the full output size.

# Target padding size
h_pad = image.shape[0] + kernel.shape[0] - 1
w_pad = image.shape[1] + kernel.shape[1] - 1
pad_shape = (h_pad, w_pad)

# 1. FFT of the Image (padded)
fft_image = np.fft.fft2(image, s=pad_shape)

# 2. FFT of the Kernel (padded)
fft_kernel = np.fft.fft2(kernel, s=pad_shape)

# 3. Multiply in Frequency Domain
fft_result = fft_image * fft_kernel

# 4. Inverse FFT to get back to Spatial Domain
fourier_result = np.fft.ifft2(fft_result)

# Take the real part (imaginary part is negligible noise)
fourier_result = np.real(fourier_result)

# ---------------------------------------------------------
# Compare and Plot Results
# ---------------------------------------------------------
difference = np.abs(spatial_result - fourier_result)
max_diff = np.max(difference)

print(f"Maximum difference between methods: {max_diff:.10e}")

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Plot Spatial Result
ax[0].imshow(spatial_result, cmap='gray')
ax[0].set_title('Method A: Spatial Convolution')
ax[0].axis('off')

# Plot Fourier Result
ax[1].imshow(fourier_result, cmap='gray')
ax[1].set_title('Method B: Fourier Multiplication')
ax[1].axis('off')

# Plot Difference
# We use a very small range for vmin/vmax to make any noise visible,
# though it should be black (zero).
ax[2].imshow(difference, cmap='gray')
ax[2].set_title(f'Difference (Max error: {max_diff:.2e})')
ax[2].axis('off')

plt.tight_layout()
plt.show()