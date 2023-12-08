import cv2
import numpy as np
import os
import glob

script_dir = os.path.dirname(__file__)

# Path to the main.py directory (one level up from script_dir)
main_dir = os.path.dirname(script_dir)

input_folder = os.path.join(main_dir, 'output','Post_processing_v09')
output_folder = os.path.join(main_dir, 'output','WB_whole_root')
area_threshold = 10000  # Set the threshold area for small white regions

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the input folder
for image_path in glob.glob(os.path.join(input_folder, '*.*')):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to make non-black pixels white
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and remove small contours
    for cnt in contours:
        if cv2.contourArea(cnt) < area_threshold:
            cv2.drawContours(binary, [cnt], -1, (0, 0, 0), -1)

    # Save the cleaned image in the output folder
    base_name = os.path.basename(image_path)
    save_path = os.path.join(output_folder, base_name)
    cv2.imwrite(save_path, binary)