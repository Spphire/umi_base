#!/usr/bin/env python

import cv2
import numpy as np
import argparse
import os
import time


def extract_first_frame(video_path):
    """Extract the first frame from a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        The first frame as a numpy array
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame from video: {video_path}")
        
    return frame


def get_video_properties(video_path):
    """Get video properties.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (width, height, fps, frame_count)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    return width, height, fps, frame_count


def find_feature_matches(img1, img2):
    """Find feature matches between two images.
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        Tuple of (good_matches, keypoints1, keypoints2)
    """
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Create feature detector and descriptor
    try:
        # Try SIFT first (available in newer OpenCV or with contrib modules)
        detector = cv2.SIFT_create()
    except AttributeError:
        try:
            # Fall back to ORB which is usually available
            detector = cv2.ORB_create()
        except AttributeError:
            # For very old versions
            detector = cv2.ORB()
    
    # Detect keypoints and compute descriptors
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        raise ValueError("Failed to extract descriptors from images")
    
    # Create BFMatcher object
    norm_type = cv2.NORM_L2 if 'SIFT' in str(detector) else cv2.NORM_HAMMING
    matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    
    # Match descriptors and find good matches
    good_matches = []
    
    if 'SIFT' in str(detector):
        # For SIFT, use ratio test
        matches = matcher.knnMatch(des1, des2, k=2)
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    else:
        # For ORB or others, use distance threshold
        matches = matcher.match(des1, des2)
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        # Take top N matches
        good_matches = matches[:min(50, len(matches))]
    
    return good_matches, kp1, kp2


def find_optimal_transform(img1, img2, good_matches, kp1, kp2):
    """Find the optimal transformation (scale and center) to match img2 to img1.
    
    Args:
        img1: First image
        img2: Second image
        good_matches: List of good matches
        kp1: Keypoints from first image
        kp2: Keypoints from second image
        
    Returns:
        Tuple of (scale, center_x, center_y)
    """
    if len(good_matches) < 4:
        raise ValueError("Not enough good matches found for transformation calculation")
    
    # Get corresponding points
    src_pts = np.array([kp2[m.trainIdx].pt for m in good_matches])
    dst_pts = np.array([kp1[m.queryIdx].pt for m in good_matches])
    
    # Calculate the mean distance ratio to estimate scale
    src_dists = []
    dst_dists = []
    
    # Calculate pairwise distances to estimate scale
    for i in range(len(src_pts)):
        for j in range(i+1, len(src_pts)):
            src_dist = np.linalg.norm(src_pts[i] - src_pts[j])
            dst_dist = np.linalg.norm(dst_pts[i] - dst_pts[j])
            if src_dist > 0 and dst_dist > 0:
                src_dists.append(src_dist)
                dst_dists.append(dst_dist)
    
    # Calculate scale as the ratio of distances
    if len(src_dists) > 0 and len(dst_dists) > 0:
        scale_ratios = [dst/src for dst, src in zip(dst_dists, src_dists)]
        scale = float(np.median(scale_ratios))
    else:
        # Default to 1.0 if we can't compute a valid scale
        scale = 1.0
    
    # Calculate the transformation center
    h1, w1 = img1.shape[:2]
    
    center_x = int(w1 // 2)
    center_y = int(h1 // 2)
    
    return scale, center_x, center_y


def render_video(video1_path, video2_path, output_path, scale, center_x, center_y, frame_offset=0, alpha=0.5):
    """Render the second video with transparency over the first video.
    
    Args:
        video1_path: Path to the first video
        video2_path: Path to the second video
        output_path: Path to save the output video
        scale: Scale factor for the second video
        center_x: X-coordinate of the center for placing the second video
        center_y: Y-coordinate of the center for placing the second video
        frame_offset: Frame offset between videos (default: 0)
        alpha: Transparency of the second video (default: 0.5)
    """
    # Open both videos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if not cap1.isOpened() or not cap2.isOpened():
        raise ValueError("Could not open video files")
    
    # Get video properties
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    
    # Calculate total frames for both videos
    total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer with a codec that should work across OpenCV versions
    # MPEG-4 codec (works in most environments)
    fourcc = int(0x7634706D)  # ASCII for 'mp4v'
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width1, height1))
    
    # Calculate effective frame ranges considering offset
    effective_start2 = max(0, frame_offset)
    effective_end2 = min(total_frames2, total_frames1 + frame_offset)
    
    effective_start1 = max(0, -frame_offset)
    effective_end1 = min(total_frames1, total_frames2 - frame_offset)
    
    # Skip frames if necessary due to offset
    for _ in range(effective_start1):
        cap1.read()
    
    for _ in range(effective_start2):
        cap2.read()
    
    # Calculate the dimensions of the resized second video
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    new_width2 = int(width2 * scale)
    new_height2 = int(height2 * scale)
    
    # Process frames
    frame_count = min(effective_end1 - effective_start1, effective_end2 - effective_start2)
    for i in range(frame_count):
        if i % 100 == 0:
            print(f"Processing frame {i}/{frame_count}")
            
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # Create a copy of frame1 for the output
        output_frame = frame1.copy()
        
        # Resize the second frame
        resized_frame2 = cv2.resize(frame2, (new_width2, new_height2))
        
        # Calculate the position to place the resized second frame
        x_offset = center_x - new_width2 // 2
        y_offset = center_y - new_height2 // 2
        
        # Create a region of interest (ROI) in the first frame
        roi_x_start = max(0, x_offset)
        roi_x_end = min(width1, x_offset + new_width2)
        roi_y_start = max(0, y_offset)
        roi_y_end = min(height1, y_offset + new_height2)
        
        # Calculate the corresponding region in the second frame
        sec_x_start = max(0, -x_offset)
        sec_x_end = sec_x_start + (roi_x_end - roi_x_start)
        sec_y_start = max(0, -y_offset)
        sec_y_end = sec_y_start + (roi_y_end - roi_y_start)
        
        # Only blend if we have valid regions to blend
        if roi_x_end > roi_x_start and roi_y_end > roi_y_start:
            # Extract ROI from frame1
            roi = output_frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
            
            # Extract corresponding region from resized frame2
            sec_roi = resized_frame2[sec_y_start:sec_y_end, sec_x_start:sec_x_end]
            
            # Blend the frames using weighted addition
            blended_roi = cv2.addWeighted(roi, 1-alpha, sec_roi, alpha, 0)
            
            # Place the blended region back into the output frame
            output_frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = blended_roi
        
        # Write the combined frame to the output video
        out.write(output_frame)
    
    # Release resources
    cap1.release()
    cap2.release()
    out.release()
    
    print(f"Output video saved to: {output_path}")


def match_videos(video1_path, video2_path, output_path, frame_offset=0, alpha=0.5):
    """Match and align two videos with the second video overlay.
    
    Args:
        video1_path: Path to the first video
        video2_path: Path to the second video
        output_path: Path to save the output video (second video overlaid on first)
        frame_offset: Frame offset between videos (default: 0, can be negative)
        alpha: Transparency level of the second video (default: 0.5)
    """
    start_time = time.time()
    
    # Check if videos exist
    if not os.path.exists(video1_path):
        raise FileNotFoundError(f"Video file not found: {video1_path}")
    
    if not os.path.exists(video2_path):
        raise FileNotFoundError(f"Video file not found: {video2_path}")
    
    # Get video properties and assert they have the same resolution and fps
    width1, height1, fps1, frame_count1 = get_video_properties(video1_path)
    width2, height2, fps2, frame_count2 = get_video_properties(video2_path)
    
    assert width1 == width2, f"Videos have different widths: {width1} vs {width2}"
    assert height1 == height2, f"Videos have different heights: {height1} vs {height2}"
    assert abs(fps1 - fps2) < 0.1, f"Videos have different frame rates: {fps1} vs {fps2}"
    
    print(f"Video 1: {width1}x{height1}, {fps1} fps, {frame_count1} frames")
    print(f"Video 2: {width2}x{height2}, {fps2} fps, {frame_count2} frames")
    
    # Extract the first frames from both videos
    print("Extracting first frames...")
    frame1 = extract_first_frame(video1_path)
    frame2 = extract_first_frame(video2_path)
    
    # Find feature matches between the first frames
    print("Finding feature matches...")
    good_matches, kp1, kp2 = find_feature_matches(frame1, frame2)
    print(f"Found {len(good_matches)} good matches between frames")
    
    if len(good_matches) < 10:
        print("Warning: Low number of good matches. Transformation might not be accurate.")
    
    # Find the optimal transformation
    print("Finding optimal transformation...")
    scale, center_x, center_y = find_optimal_transform(frame1, frame2, good_matches, kp1, kp2)
    print(f"Optimal transformation: scale={scale:.4f}, center=({center_x}, {center_y})")
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Render the videos together
    print("Rendering videos...")
    render_video(video1_path, video2_path, output_path, scale, center_x, center_y, frame_offset, alpha)
    
    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description='Match and align two videos.')
    parser.add_argument('video1', type=str, help='Path to the first video')
    parser.add_argument('video2', type=str, help='Path to the second video')
    parser.add_argument('output', type=str, help='Path to save the output video')
    parser.add_argument('--frame-offset', type=int, default=0, 
                        help='Frame offset between videos (default: 0)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Transparency level of the second video (default: 0.5)')
    
    args = parser.parse_args()
    
    try:
        match_videos(args.video1, args.video2, args.output, args.frame_offset, args.alpha)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
