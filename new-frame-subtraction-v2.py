import cv2
import numpy as np
import os
import glob
from concurrent.futures import ThreadPoolExecutor

# Customizable parameters
START_FRAME_OFFSET = 24  # Skip first N frames
END_FRAME_OFFSET = 24  # Stop N frames before the last frame
INTENSITY_SCALING_FACTOR = 25  # Scale frame difference by this factor
APPLY_HISTOGRAM_EQUALIZATION = True  # Set to True to test its effects
GAUSSIAN_BLUR_KERNEL = 0  # Use 0 to disable, or 3-5 for mild smoothing
#NUM_THREADS = min(2, os.cpu_count())  # Automatically set thread count based on CPU cores
NUM_THREADS = os.cpu_count()
OUTPUT_DIR = 'subtracted_videos_10sec_strongintensity_smoothed'
WINDOW_SECONDS = 10


def bgsubtract(video_path):
    """ Performs frame subtraction on the given video file. """
    output_video_path = f'{OUTPUT_DIR}/{os.path.basename(video_path)}'
    
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get FPS and total frame count
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define start and end frames
    start_frame = min(START_FRAME_OFFSET, total_frames - 1)
    end_frame = max(total_frames - END_FRAME_OFFSET, start_frame + fps)

    print(f"Processing {video_path} | FPS: {fps} | Start: {start_frame} | End: {end_frame} | Total: {total_frames}")

    # Set up the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v for MP4 files
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)

    # Move to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Initialize frame buffer
    frame_buffer = []
    
    # Fill buffer with the first `fps` frames
    for i in range(fps * WINDOW_SECONDS):
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading initial frames from {video_path}")
            cap.release()
            out.release()
            return
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_buffer.append(frame_gray)

    # Process frames within the defined range
    for current_frame in range(start_frame + (fps * WINDOW_SECONDS), end_frame):
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video ends early

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get the reference frame (fps frames before)
        reference_frame = frame_buffer.pop(0)  # Remove the oldest frame
        frame_buffer.append(frame_gray)  # Add the newest frame

        # Perform frame subtraction
        frame_diff = cv2.absdiff(frame_gray, reference_frame)

        # Apply intensity scaling
        frame_diff = np.clip(frame_diff * INTENSITY_SCALING_FACTOR, 0, 255).astype(np.uint8)

        # Apply histogram equalization if enabled
        if APPLY_HISTOGRAM_EQUALIZATION:
            frame_diff = cv2.equalizeHist(frame_diff)

        # Apply mild Gaussian blur if needed
        if GAUSSIAN_BLUR_KERNEL > 0:
            frame_diff = cv2.GaussianBlur(frame_diff, (GAUSSIAN_BLUR_KERNEL, GAUSSIAN_BLUR_KERNEL), 0)

        # Write to output video
        out.write(frame_diff)

    cap.release()
    out.release()
    print(f"Finished processing {video_path}")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Automatically find all .mp4 files in the script's directory
video_files = glob.glob("*.mp4")

# Multi-threading: Process videos in parallel
with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    executor.map(bgsubtract, video_files)

print("All videos have been processed!")
