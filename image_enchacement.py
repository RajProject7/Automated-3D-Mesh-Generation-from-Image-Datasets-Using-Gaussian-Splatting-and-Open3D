import os
from PIL import Image, ImageEnhance, ImageFilter

# Function to process an image (enhance brightness, contrast, and reduce noise)
def process_image(input_path, output_path, brightness_factor=1.5, contrast_factor=1.3, noise_reduction_strength=3):
    # Open the image
    img = Image.open(input_path)
    
    # Step 1: Enhance Brightness
    brightness_enhancer = ImageEnhance.Brightness(img)
    bright_img = brightness_enhancer.enhance(brightness_factor)  # Adjust brightness by brightness_factor
    
    # Step 2: Enhance Contrast (optional)
    contrast_enhancer = ImageEnhance.Contrast(bright_img)
    contrast_img = contrast_enhancer.enhance(contrast_factor)  # Adjust contrast by contrast_factor
    
    # Step 3: Noise Reduction
    # Apply MedianFilter for noise reduction
    noise_reduced_img = contrast_img.filter(ImageFilter.MedianFilter(size=noise_reduction_strength))
    
    # Step 4: Save the processed image
    noise_reduced_img.save(output_path)
    print(f"Processed and saved: {output_path}")

# Function to process a batch of images
def process_batch_images(input_folder, output_folder, brightness_factor=1.5, contrast_factor=1.3, noise_reduction_strength=3):
    # Check if the output folder exists, create it if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Process only image files
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"enhanced_{filename}")
            
            # Process the image
            process_image(input_path, output_path, brightness_factor, contrast_factor, noise_reduction_strength)

# Example usage:
input_folder = 'input_images'  
output_folder = 'output_images'  

# Batch process all images in the input folder
process_batch_images(input_folder, output_folder, brightness_factor=1.5, contrast_factor=1.3, noise_reduction_strength=3)
