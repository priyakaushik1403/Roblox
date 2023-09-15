# import the necessary packages
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv
import time

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

# Load the video file
video_path = "./Videos/AdoptMe(2).mov"
cap = cv2.VideoCapture(video_path)

# Read the first frame
ret, prev_frame = cap.read()
prev_gray_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Initialize frame counter and time counter
frame_number = 1
current_time = time.time()

# Initialize lists to store MSE and SSIM values
mse_values = []
ssim_values = []

# Function to save frames
def save_frames(imageA, imageB, frame_number_A, frame_number_B):
    output_name_A = f'./ImageFrames/AdoptMe/frame_{frame_number_A}.png'
    output_name_B = f'./ImageFrames/AdoptMe/frame_{frame_number_B}.png'
    cv2.imwrite(output_name_A, imageA)
    cv2.imwrite(output_name_B, imageB)

# Skip frames for 2 seconds (adjust the skip_time value as needed)
skip_time = 2

# Loop over the frames of the video
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the current frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the time difference between the current frame and the previous frame
    time_difference = time.time() - current_time

    # Perform image comparison if the time difference is greater than or equal to 2 seconds
    if time_difference >= skip_time:
        # Perform image comparison between consecutive frames
        m = mse(prev_gray_frame, gray_frame)
        s = ssim(prev_gray_frame, gray_frame)

        # Check if SSIM is less than 0.8 and MSE is greater than 2000
        if s < 0.8 and m > 2000:
            # Print the comparison values
            print(f"Comparison: Frame {frame_number - 1} vs. Frame {frame_number}, MSE: {m:.2f}, SSIM: {s:.2f}")

            # Save the frames as images
            save_frames(prev_frame, frame, frame_number - 1, frame_number)

        # Set the current frame as the previous frame for the next iteration
        prev_gray_frame = gray_frame
        prev_frame = frame

        # Update the current time
        current_time = time.time()

    # Increment the frame counter
    frame_number += 1

# Release the video capture object
cap.release()