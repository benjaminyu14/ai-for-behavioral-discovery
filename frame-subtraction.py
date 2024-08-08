# Benjamin Yu August 2024
# Given a video, will perform frame subtraction to determine fluctuation in video's lighting
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os
import multiprocessing


global first_frame
first_cap = cv2.VideoCapture('2024-07-03 16:00:19.341102.mp4') # set to desired video to take subtractor frame from
ret, first_frame = first_cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    exit()

def bgsubtract(row):
    video_path = row['filename']
    output_video_path = f'bg-subtracted-ant-vids/{video_path}'
    cap = cv2.VideoCapture(video_path)


    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    

    # Convert first frame to grayscale
    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Define the frame range
    start_frame = 4 # start at 4 to remove potential flawed frames
    end_frame = int(row['endframe']) - 50



    fps = cap.get(cv2.CAP_PROP_FPS)

   
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = first_frame_gray.shape
    out = cv2.VideoWriter(output_video_path, fourcc, 25.0, (width, height), isColor=False)

    # Iterate over the specified frame range
    for frame_num in range(start_frame, end_frame + 1):

        # Set current frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        
        ret, current_frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {frame_num}.")
            continue

        # Convert to gray scale
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Absolute subtraction with the first frame
        frame_diff = cv2.absdiff(current_frame_gray, first_frame_gray)
        
        # Intensity scaling: Multiply the resulting matrix by 151 (max pixel value in original frame)
        modified_frame_diff = frame_diff * 151
        
        # Clip the values to [0, 255]
        modified_frame_diff_clipped = np.clip(modified_frame_diff, 0, 255).astype(np.uint8)
        
        # Write to output video
        out.write(modified_frame_diff_clipped)

    cap.release()
    out.release()

    print(f"Output video saved as {output_video_path}")

csv = pd.read_csv('dataset.csv')

# Can set max_workers to however many threads available
# num_threads = os.cpu_count()
# print(f'num of threads = {num_threads}')

with ProcessPoolExecutor(max_workers=15) as executor:
    for _, row in csv.iterrows():
        executor.submit(bgsubtract, row)
