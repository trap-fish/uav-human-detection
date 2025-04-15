import os
import random
import shutil
import cv2
import numpy as np

def resize_with_letterbox(image_path, target_shape=(1,640,640,3), padding_value=(0, 0, 0)):
    """
    Resizes an image with letterboxing to fit the target size, preserving aspect ratio.
    
    Parameters:
        image_path (str): Path to the input image.
        target_shape (tuple): Target shape in NHWC format (batch_size, target_height, target_width, channels).
        padding_value (tuple): RGB values for padding (default is black padding).
        
    Returns:
        letterboxed_image (ndarray): The resized image with letterboxing.
        scale (float): Scaling ratio applied to the original image.
        pad_top (int): Padding applied to the top.
        pad_left (int): Padding applied to the left.
    """
    # Load the image from the given path
    image = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if image is None:
        raise ValueError(f"Error: Unable to load image from path: {image_path}")
    
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get the original image dimensions (height, width, channels)
    h, w, c = image.shape
    
    # Extract target height and width from target_shape (NHWC format)
    target_height, target_width = target_shape[1], target_shape[2]
    
    # Calculate the scaling factors for width and height
    scale_x = target_width / w
    scale_y = target_height / h
    
    # Choose the smaller scale factor to preserve the aspect ratio
    scale = min(scale_x, scale_y)
    
    # Calculate the new dimensions based on the scaling factor
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the image to the new dimensions
    resized_image = cv2.resize(image, (new_w, new_h),interpolation=cv2.INTER_LINEAR)
    
    # Create a new image with the target size, filled with the padding value
    letterboxed_image = np.full((target_height, target_width, c), padding_value, dtype=np.uint8)
    
    # Compute the position where the resized image should be placed (padding)
    pad_top = (target_height - new_h) // 2
    pad_left = (target_width - new_w) // 2
    
    # Place the resized image onto the letterbox background
    letterboxed_image[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized_image

    
    # Return the letterboxed image, scaling ratio, and padding (top, left)
    return letterboxed_image, scale, pad_top, pad_left

# Set your paths
rootdir = "/media/citi-ai/matthew/uav-human-detection/"
src_dir = os.path.join(rootdir, "datasets/filtered/visdrone_humans/train/images")
dst_dir = os.path.join(rootdir, "hailo-ai/shared_with_docker/visdrone/VisDrone2019-DET-calib/images")
# Create the destination directory if it doesn't exist
os.makedirs(dst_dir, exist_ok=True)
num_images = 2048
calib_images = random.sample(os.listdir(src_dir), num_images) # assumes only images are in the directory

for filename in calib_images:
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, filename)

    try:
        processed_img, _, _, _ = resize_with_letterbox(src_path)
        # Convert RGB back to BGR for saving with OpenCV
        processed_img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dst_path, processed_img_bgr)
        print(f"Saved: {dst_path}")
    except Exception as e:
        print(f"Failed to process {src_path}: {e}")
