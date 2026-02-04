# Image Blurring: Spatial vs Frequency Domain

A Computer Vision demonstration comparing two mathematically equivalent methods for image blurring using the Convolution Theorem.

## Concept

This project demonstrates a fundamental principle in signal processing and computer vision: **the Convolution Theorem**, which states that convolution in the spatial domain is equivalent to multiplication in the frequency domain.

### Two Methods for Image Blurring

#### Method A: Spatial Domain Filtering (Direct Convolution)
- Applies a Gaussian blur kernel directly to the image using 2D convolution
- Computationally intensive for large kernels
- Intuitive approach: each output pixel is a weighted average of neighboring input pixels

#### Method B: Frequency Domain Filtering (Fourier Transform)
- Converts both the image and kernel to the frequency domain using Fast Fourier Transform (FFT)
- Performs element-wise multiplication in the frequency domain
- Converts the result back to spatial domain using Inverse FFT
- More efficient for large kernels due to FFT's O(n log n) complexity

### The Convolution Theorem

```
f(x,y) ⊗ g(x,y) = F⁻¹{F{f(x,y)} · F{g(x,y)}}
```

Where:
- `⊗` represents 2D convolution
- `F{}` represents the Fourier transform
- `F⁻¹{}` represents the Inverse Fourier transform
- `·` represents element-wise multiplication

### Gaussian Kernel

The code uses a Gaussian kernel for blurring, defined as:

```
G(x,y) = exp(-(x² + y²) / (2σ²))
```

Where `σ` (sigma) controls the blur strength. The kernel is normalized so all weights sum to 1.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- SciPy
- scikit-image

## Installation

1. **Clone the repository:**
   ```bash
   git clone git@github.com:jithu2111/image-blurring.git
   cd image-blurring
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment:**
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

4. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare your image:**
   - Place an image file (`.jpg`, `.png`, etc.) in the project directory
   - The code currently uses `image.jpg` by default

2. **Update the filename (if needed):**
   - Open `blurring.py`
   - Modify line 10 to match your image filename:
     ```python
     filename = 'your_image.jpg'  # Change this to your image filename
     ```

3. **Run the script:**
   ```bash
   python blurring.py
   ```

## Expected Output

The script will:
1. Load and convert your image to grayscale
2. Apply Gaussian blur using both methods
3. Calculate the maximum difference between the two results
4. Display three side-by-side plots:
   - **Left:** Spatial domain convolution result
   - **Middle:** Frequency domain multiplication result
   - **Right:** Difference map (should be nearly black, indicating minimal error)

The console will print:
```
Maximum difference between methods: X.XXXXe-XX
```

This value should be extremely small (near zero), confirming that both methods produce equivalent results.

## Technical Details

### Image Preprocessing
- Handles both RGB and RGBA images (removes alpha channel if present)
- Converts to grayscale using scikit-image's `rgb2gray`
- Ensures float precision for accurate calculations

### Padding Strategy
To achieve linear convolution (matching spatial domain results), the code uses zero-padding:
- **Spatial domain:** Uses `mode='full'` convolution
- **Frequency domain:** Pads both image and kernel to size `(H + h - 1) × (W + w - 1)` before FFT

Without proper padding, FFT assumes circular convolution, which would produce different results at image boundaries.

### Kernel Parameters
- **Size:** 15×15 pixels (adjustable via `kernel_size` variable)
- **Sigma:** 3 (adjustable via `sigma` parameter)
- Larger sigma values create stronger blur effects

## Customization

You can modify these parameters in `blurring.py`:

```python
kernel_size = 15  # Size of the Gaussian kernel (must be odd)
sigma = 3         # Standard deviation of the Gaussian (blur strength)
```

## File Structure

```
Module 3 - Image Blurring/
├── blurring.py      # Main Python script
├── image.jpg        # Sample input image
├── img.png          # Sample input image
├── venv/            # Virtual environment (if created)
└── README.md        # This file
```
