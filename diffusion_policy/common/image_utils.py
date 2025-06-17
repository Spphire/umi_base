import cv2
import numpy as np

def center_pad_and_resize_image(image, target_size=(224, 224)):
    """
    Center the image, pad with black to make it square, then resize to target size
    
    Args:
        image: Input image
        target_size: Target size (height, width)
    
    Returns:
        Processed image
    """
    h, w = image.shape[:2]
    max_dim = max(h, w)
    
    # Create black square background
    square_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    
    # Calculate center position
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    
    # Place original image in center
    square_img[y_offset:y_offset+h, x_offset:x_offset+w] = image
    
    # Resize to target size
    resized_img = cv2.resize(square_img, target_size)
    
    return resized_img