import matplotlib.pyplot as plt
import numpy as np
import pyoidn
from PIL import Image


def denoise_image(noisy_image, albedo=None, normal=None):
    """
    Denoise a numpy image using Intel OIDN

    Args:
        noisy_image: numpy array of shape (H, W, 3) with float32 values in [0, 1]
        albedo: optional albedo buffer (same shape as noisy_image)
        normal: optional normal buffer (same shape as noisy_image)

    Returns:
        denoised_image: numpy array of same shape as input
    """

    # Ensure input is float32 and in correct range
    if noisy_image.dtype != np.float32:
        noisy_image = noisy_image.astype(np.float32)

    # Ensure values are in [0, 1] range
    if noisy_image.max() > 1.0:
        noisy_image = noisy_image / 255.0

    # Create OIDN device
    device = pyoidn.Device("cpu")  # Use 'cuda' for GPU if available
    device.commit()

    # Create filter
    filter = device.newFilter("RT")  # RT = Ray Tracing filter

    # Set input image
    filter.setImage("color", noisy_image)

    # Set auxiliary features if provided (improves denoising quality)
    if albedo is not None:
        if albedo.dtype != np.float32:
            albedo = albedo.astype(np.float32)
        if albedo.max() > 1.0:
            albedo = albedo / 255.0
        filter.setImage("albedo", albedo)

    if normal is not None:
        if normal.dtype != np.float32:
            normal = normal.astype(np.float32)
        # Normals should be in [-1, 1] range
        if normal.max() > 1.0:
            normal = (normal / 255.0) * 2.0 - 1.0
        filter.setImage("normal", normal)

    # Create output buffer
    output = np.zeros_like(noisy_image)
    filter.setImage("output", output)

    # Commit filter and execute
    filter.commit()
    filter.execute()

    return output


def load_and_denoise_example():
    """
    Example function showing how to load an image and denoise it
    """

    # Example 1: Create synthetic noisy image for demonstration
    print("Creating synthetic noisy image...")

    # Create a simple test image
    height, width = 256, 256
    x, y = np.meshgrid(
        np.linspace(0, 4 * np.pi, width), np.linspace(0, 4 * np.pi, height)
    )
    clean_image = np.zeros((height, width, 3), dtype=np.float32)
    clean_image[:, :, 0] = 0.5 + 0.3 * np.sin(x) * np.cos(y)  # Red channel
    clean_image[:, :, 1] = 0.5 + 0.3 * np.cos(x) * np.sin(y)  # Green channel
    clean_image[:, :, 2] = 0.5 + 0.3 * np.sin(x + y)  # Blue channel

    # Add noise
    noise_level = 0.1
    noisy_image = clean_image + np.random.normal(
        0, noise_level, clean_image.shape
    ).astype(np.float32)
    noisy_image = np.clip(noisy_image, 0, 1)

    # Denoise
    print("Denoising image...")
    denoised_image = denoise_image(noisy_image)

    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(clean_image)
    axes[0].set_title("Original Clean Image")
    axes[0].axis("off")

    axes[1].imshow(noisy_image)
    axes[1].set_title("Noisy Image")
    axes[1].axis("off")

    axes[2].imshow(denoised_image)
    axes[2].set_title("OIDN Denoised Image")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    return clean_image, noisy_image, denoised_image


def denoise_from_file(image_path):
    """
    Load an image from file and denoise it

    Args:
        image_path: path to input image file

    Returns:
        tuple: (original_image, denoised_image) as numpy arrays
    """

    # Load image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Denoise
    denoised = denoise_image(img_array)

    return img_array, denoised


def save_denoised_image(denoised_array, output_path):
    """
    Save denoised numpy array as image file

    Args:
        denoised_array: numpy array with float32 values in [0, 1]
        output_path: path for output image
    """

    # Convert back to uint8
    img_uint8 = (np.clip(denoised_array, 0, 1) * 255).astype(np.uint8)

    # Save using PIL
    img_pil = Image.fromarray(img_uint8)
    img_pil.save(output_path)
    print(f"Denoised image saved to: {output_path}")


# Example usage with auxiliary buffers (for ray-traced renders)
def denoise_with_auxiliary_buffers(color_image, albedo_image=None, normal_image=None):
    """
    Enhanced denoising using auxiliary buffers (albedo and normals)
    This provides much better results for ray-traced images

    Args:
        color_image: main noisy image (H, W, 3)
        albedo_image: surface albedo/diffuse color (H, W, 3), optional
        normal_image: surface normals (H, W, 3), optional

    Returns:
        denoised image
    """

    return denoise_image(color_image, albedo_image, normal_image)


if __name__ == "__main__":
    # Installation note
    print("To install OIDN Python bindings:")
    print("pip install pyoidn")
    print("\nNote: You may also need to install the OIDN library itself")
    print("from: https://www.openimagedenoise.org/downloads.html")
    print()

    # Run example
    try:
        clean, noisy, denoised = load_and_denoise_example()

        # Calculate PSNR for comparison
        def calculate_psnr(img1, img2):
            mse = np.mean((img1 - img2) ** 2)
            if mse == 0:
                return float("inf")
            return 20 * np.log10(1.0 / np.sqrt(mse))

        psnr_noisy = calculate_psnr(clean, noisy)
        psnr_denoised = calculate_psnr(clean, denoised)

        print(f"PSNR - Noisy vs Clean: {psnr_noisy:.2f} dB")
        print(f"PSNR - Denoised vs Clean: {psnr_denoised:.2f} dB")
        print(f"Improvement: {psnr_denoised - psnr_noisy:.2f} dB")

    except ImportError:
        print("pyoidn not installed. Install with: pip install pyoidn")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure OIDN is properly installed on your system")
