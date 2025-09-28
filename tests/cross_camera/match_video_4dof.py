#!/usr/bin/env python
import argparse
import cv2
import numpy as np
import os
import subprocess
from tqdm import tqdm


def match_features(img1, img2):
    """
    Match features between two images and find the best transformation parameters
    Returns: (x, y, k) where x, y are center point coordinates and k is scale factor
    """
    # Convert to grayscale for feature detection
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Use simpler feature detection
    # Detect good features to track
    corners1 = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.01, minDistance=10)
    corners2 = cv2.goodFeaturesToTrack(gray2, maxCorners=100, qualityLevel=0.01, minDistance=10)
    
    # Work directly with corner points
    corners1_pts = corners1.reshape(-1, 2) if corners1 is not None else np.array([])
    corners2_pts = corners2.reshape(-1, 2) if corners2 is not None else np.array([])
    
    # Extract features around these keypoints using small patches
    des1 = []
    des2 = []
    patch_size = 7  # Use a small patch size
    
    for pt in corners1_pts:
        x, y = int(pt[0]), int(pt[1])
        # Make sure patch is within bounds
        if (x - patch_size >= 0 and x + patch_size < gray1.shape[1] and
            y - patch_size >= 0 and y + patch_size < gray1.shape[0]):
            patch = gray1[y-patch_size:y+patch_size+1, x-patch_size:x+patch_size+1]
            des1.append(patch.flatten())
    
    for pt in corners2_pts:
        x, y = int(pt[0]), int(pt[1])
        if (x - patch_size >= 0 and x + patch_size < gray2.shape[1] and
            y - patch_size >= 0 and y + patch_size < gray2.shape[0]):
            patch = gray2[y-patch_size:y+patch_size+1, x-patch_size:x+patch_size+1]
            des2.append(patch.flatten())
    
    # Simple matching based on sum of squared differences
    matches = []
    corners1_indices = []
    corners2_indices = []
    
    for i, desc1 in enumerate(des1):
        best_dist = float('inf')
        best_j = -1
        for j, desc2 in enumerate(des2):
            if len(desc1) == len(desc2):
                dist = np.sum((desc1 - desc2) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
        
        if best_j != -1:
            corners1_indices.append(i)
            corners2_indices.append(best_j)
            # Create a simple match object with distance for sorting
            match = type('Match', (), {'distance': best_dist, 'idx1': i, 'idx2': best_j})
            matches.append(match)
    
    # Filter matches by distance (keep best 25%)
    matches.sort(key=lambda x: x.distance)
    good_matches = matches[:max(10, len(matches)//4)]  # At least 10 matches
    
    if len(good_matches) < 10:
        print(f"Warning: Only {len(good_matches)} good matches found. Results may be unreliable.")
    
    # Extract location of good matches
    src_pts = []
    dst_pts = []
    for m in good_matches:
        i, j = m.idx1, m.idx2
        if i < len(corners1_indices) and j < len(corners2_indices):
            idx1 = corners1_indices[i]
            idx2 = corners2_indices[j]
            if idx1 < len(corners1_pts) and idx2 < len(corners2_pts):
                src_pts.append(corners1_pts[idx1])
                dst_pts.append(corners2_pts[idx2])
    
    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)
    
    if len(src_pts) < 4 or len(dst_pts) < 4:
        print("Error: Not enough matching points to compute transformation.")
        return img1.shape[1]//2, img1.shape[0]//2, 1.0  # Default values
        
    # Find homography
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    # Get image dimensions
    h1, w1 = img1.shape[:2]
    
    # Calculate scale and translation
    # Decompose homography matrix to get scale and translation
    # This is a simplification - a proper decomposition would be more complex
    scale = np.sqrt(H[0, 0]**2 + H[0, 1]**2)
    
    # Map center of second image through homography
    h2, w2 = img2.shape[:2]
    center_point = np.array([[[w2//2, h2//2]]], dtype=np.float32)
    
    try:
        dst = cv2.perspectiveTransform(center_point, H)
        x, y = dst[0][0]
    except Exception:
        # Fallback if transform fails
        print("Warning: Transform failed, using image centers")
        x, y = img1.shape[1] // 2, img1.shape[0] // 2
        
    # Return transformation parameters
    k = scale
    
    return x, y, k


def resize_and_position_frame(frame, x, y, k, target_shape):
    """Resize frame with scale k and position it at (x,y) in target_shape"""
    h, w = frame.shape[:2]
    new_w, new_h = int(w * k), int(h * k)
    
    # Resize the frame
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Create a blank canvas with the target shape
    result = np.zeros((*target_shape, 3), dtype=np.uint8)
    
    # Calculate positions to place the resized image
    # x and y represent the center point position
    x_offset = int(x - new_w // 2)
    y_offset = int(y - new_h // 2)
    
    # Ensure we don't go out of bounds
    src_x_start = max(0, -x_offset)
    src_y_start = max(0, -y_offset)
    dst_x_start = max(0, x_offset)
    dst_y_start = max(0, y_offset)
    
    src_x_end = min(new_w, target_shape[1] - x_offset)
    src_y_end = min(new_h, target_shape[0] - y_offset)
    dst_x_end = min(target_shape[1], x_offset + new_w)
    dst_y_end = min(target_shape[0], y_offset + new_h)
    
    # Check if there's any valid region to copy
    width = src_x_end - src_x_start
    height = src_y_end - src_y_start
    
    if width > 0 and height > 0:
        result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = resized[src_y_start:src_y_end, src_x_start:src_x_end]
    
    return result


def blend_frames(frame1, frame2, alpha=0.5):
    """Blend two frames with the given alpha value"""
    return cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)


def match_and_render_videos(video_path1, video_path2, output_path, frame_offset=0, alpha=0.5):
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    """
    Match two videos and render them together with the second video semi-transparent
    
    Args:
        video_path1: Path to the first video
        video_path2: Path to the second video
        output_path: Path to save the output video
        frame_offset: Frame offset between videos (can be negative)
        alpha: Opacity of the second video (0 = transparent, 1 = opaque)
    """
    # Open videos
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)
    
    # Check if videos are opened successfully
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both videos.")
        return False
    
    # Get video properties
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Assert that videos have the same resolution and frame rate
    assert width1 == width2 and height1 == height2, "Videos must have the same resolution"
    assert abs(fps1 - fps2) < 0.01, "Videos must have the same frame rate"
    
    # Initialize temp_dir for potential ffmpeg use
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
    
    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        ffmpeg_available = True
        print("ffmpeg is available, will use it as a fallback if OpenCV VideoWriter fails")
    except (subprocess.SubprocessError, FileNotFoundError):
        ffmpeg_available = False
    
    # For OpenCV VideoWriter, try with an AVI format
    output_path_opencv = output_path
    if output_path.lower().endswith('.mp4'):
        output_path_opencv = output_path.rsplit('.', 1)[0] + '.avi'
        print(f"Using .avi format for OpenCV: {output_path_opencv}")
    
    # Try different codec options
    fourcc = 0  # Default codec
    if output_path_opencv.lower().endswith('.avi'):
        # XVID codec - commonly available for AVI
        fourcc = 1145656920  # XVID as integer value
    
    out = cv2.VideoWriter(output_path_opencv, fourcc, fps1, (width1, height1))
    
    # If OpenCV writer failed and ffmpeg is available, we'll use ffmpeg later
    use_ffmpeg = ffmpeg_available and not out.isOpened()
    if not out.isOpened() and not use_ffmpeg:
        print("Error: Could not initialize video writer. Trying with default codec...")
        # Try with default codec
        out = cv2.VideoWriter(output_path_opencv, 0, fps1, (width1, height1))
        
        if not out.isOpened():
            print("Error: All attempts to create a video writer have failed.")
            return False
            
    # If we're using ffmpeg, prepare a temporary directory for frames
    if use_ffmpeg:
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        print(f"Will save frames to {temp_dir} and use ffmpeg to create the video")
    
    # Read first frames for feature matching
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        print("Error: Could not read first frame from one or both videos.")
        return False
    
    # Match features and find transformation parameters
    x, y, k = match_features(frame1, frame2)
    print(f"Transformation parameters: center=({x:.2f}, {y:.2f}), scale={k:.2f}")
    
    # Reset video captures
    cap1.release()
    cap2.release()
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)
    
    # Determine start frames based on frame_offset
    start_frame1 = max(0, -frame_offset)
    start_frame2 = max(0, frame_offset)
    
    # Skip to start frames
    for _ in range(start_frame1):
        cap1.read()
    for _ in range(start_frame2):
        cap2.read()
    
    # Calculate output frame count
    output_frame_count = min(frame_count1 - start_frame1, frame_count2 - start_frame2)
    
    # Process frames
    frame_files = []  # To store paths of saved frames for ffmpeg
    
    for frame_idx in tqdm(range(output_frame_count), desc="Processing frames"):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # Resize and position frame2
        transformed_frame2 = resize_and_position_frame(frame2, x, y, k, (height1, width1))
        
        # Blend frames
        blended_frame = blend_frames(frame1, transformed_frame2, alpha)
        
        if use_ffmpeg:
            # Save frame to temporary directory
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(frame_path, blended_frame)
            frame_files.append(frame_path)
        else:
            # Write to output using OpenCV
            out.write(blended_frame)
    
    # Release resources
    cap1.release()
    cap2.release()
    
    # If using OpenCV VideoWriter, release it
    if not use_ffmpeg:
        out.release()
        output_file_to_check = output_path_opencv
    else:
        # Use ffmpeg to create video from saved frames
        try:
            ffmpeg_cmd = [
                "ffmpeg", "-y",  # Overwrite if file exists
                "-framerate", str(fps1),
                "-i", os.path.join(temp_dir, "frame_%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "23",  # Quality level (lower is better)
                output_path
            ]
            print(f"Running ffmpeg: {' '.join(ffmpeg_cmd)}")
            subprocess.run(ffmpeg_cmd, check=True)
            
            # Clean up temporary frames
            print("Cleaning up temporary frames...")
            for frame_file in frame_files:
                os.remove(frame_file)
            os.rmdir(temp_dir)
            
            output_file_to_check = output_path
            
        except subprocess.SubprocessError as e:
            print(f"Error running ffmpeg: {e}")
            # Try to copy the OpenCV output if it exists
            if os.path.exists(output_path_opencv) and os.path.getsize(output_path_opencv) > 0:
                print(f"Using OpenCV output as fallback: {output_path_opencv}")
                output_file_to_check = output_path_opencv
            else:
                print("Failed to create video output.")
                return False
    
    # Check if the output file exists and has a valid size
    if os.path.exists(output_file_to_check):
        file_size = os.path.getsize(output_file_to_check)
        if file_size > 0:
            print(f"Output video saved to {output_file_to_check} (Size: {file_size/1024:.2f} KB)")
            print(f"Absolute path: {os.path.abspath(output_file_to_check)}")
            
            # If the OpenCV output path is different from the requested output path
            # and we couldn't use ffmpeg, copy the file to the requested path
            if output_file_to_check != output_path and not use_ffmpeg:
                import shutil
                try:
                    shutil.copy2(output_file_to_check, output_path)
                    print(f"Copied output file to requested path: {output_path}")
                except (shutil.Error, IOError) as e:
                    print(f"Error copying output file: {e}")
            
            return True
        else:
            print(f"Warning: Output file exists but has zero size: {output_file_to_check}")
    else:
        print(f"Warning: Output file was not created at {output_file_to_check}")
    
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Match and blend two videos')
    parser.add_argument('video1', help='Path to the first video')
    parser.add_argument('video2', help='Path to the second video')
    parser.add_argument('output', help='Path to save the output video')
    parser.add_argument('--frame-offset', type=int, default=0, help='Frame offset between videos')
    parser.add_argument('--alpha', type=float, default=0.5, help='Opacity of the second video (0=transparent, 1=opaque)')
    
    args = parser.parse_args()
    
    match_and_render_videos(
        args.video1,
        args.video2,
        args.output,
        args.frame_offset,
        args.alpha
    )
