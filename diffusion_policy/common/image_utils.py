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

def center_crop_and_resize_image(image, target_size=(224, 224), crop=True):
    # """
    # Center crop the image to square using min(h, w), then resize to target size
    
    # Args:
    #     image: Input image
    #     target_size: Target size (height, width)
    
    # Returns:
    #     Processed image
    # """
    # h, w = image.shape[:2]
    # min_dim = min(h, w)
    
    # # Calculate center crop coordinates
    # y_start = (h - min_dim) // 2
    # x_start = (w - min_dim) // 2
    
    # # Center crop to square
    # cropped_img = image[y_start:y_start+min_dim, x_start:x_start+min_dim]
    
    # # Resize to target size
    # resized_img = cv2.resize(cropped_img, target_size)
    
    # return resized_img
    """
    Process image by either center-cropping or padding to keep aspect ratio.

    Args:
        image: Input image (H, W, C)
        target_size: Target size (height, width)
        crop: 
            - True: center crop to square, then resize
            - False: resize with aspect ratio preserved and pad with black

    Returns:
        Processed image
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    if crop:
        # ---- Original behavior: center crop to square ----
        min_dim = min(h, w)
        y_start = (h - min_dim) // 2
        x_start = (w - min_dim) // 2

        cropped_img = image[
            y_start:y_start + min_dim,
            x_start:x_start + min_dim
        ]

        resized_img = cv2.resize(cropped_img, (target_w, target_h))
        return resized_img

    else:
        # ---- Resize with aspect ratio + black padding ----
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))

        # Create black canvas
        padded_img = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)

        # Center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2

        padded_img[
            y_offset:y_offset + new_h,
            x_offset:x_offset + new_w
        ] = resized

        return padded_img