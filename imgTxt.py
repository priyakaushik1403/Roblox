import cv2
import pytesseract as pt
import pandas as pd
import numpy as np
import os
from natsort import natsorted

# def remove_background(image_path):
#     # Load the image
#     image = cv2.imread(image_path)

#     # Convert the image from BGR to RGB (OpenCV uses BGR by default)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Define the color range for the background
#     lower_bound = np.array([0, 0, 0])  # Lower bound of the color range (black)
#     upper_bound = np.array([150, 150, 150])  # Upper bound of the color range (gray)

#     # Create a mask by thresholding the image based on the color range
#     mask = cv2.inRange(image_rgb, lower_bound, upper_bound)

#     # Invert the mask to select the foreground instead of the background
#     inverted_mask = cv2.bitwise_not(mask)

#     # Apply the mask to the original image
#     foreground = cv2.bitwise_and(image_rgb, image_rgb, mask=inverted_mask)

#     return foreground

if __name__ == "__main__":
    input_folder = "./Processed Image/Adoptme/"
    image_files = natsorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
    
    # Create a list to store the data for the CSV
    data_list = []
    # Initialize a set to store unique lines of text
    unique_lines = set()
    
    # Initialize a set to store unique lines of message
    unique_message = set()

    for image_file in image_files:
        input_image_path = os.path.join(input_folder, image_file)
        # output = remove_background(input_image_path)
        # cv2.imshow('Image', output)
        # cv2.waitKey(0)
        
    # gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray,(3,3),0)
    # thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 1)
        text = pt.image_to_string(input_image_path, lang="eng")
        print(text)
        lines = text.split('\n')
        for line in lines:
            cleaned_line = line.replace(" ", "").strip()  # Remove spaces from the line before comparing
            if cleaned_line and cleaned_line not in unique_lines:  # Check if line is not empty and not already in set
                if ":" in line:
                    username, message = line.split(":", 1)  # Split the line into two parts from the first ":"
                    message_clean = message.replace(" ", "").strip()
                    if message_clean and message_clean not in unique_message:
                        data_list.append({"Filename": input_image_path, "Username": username.strip(), "Text Message": message.strip()})
                        unique_message.add(message_clean)                        
                else:
                    data_list.append({"Filename": input_image_path, "Text Message": line.strip()})
                unique_lines.add(cleaned_line)
                   
    df = pd.DataFrame(data_list)

# Save the DataFrame to a CSV file
    output_csv_path = "Adoptme.csv"
    df.to_csv(output_csv_path, index=False)
    print("Text data with filenames saved to CSV:", output_csv_path) 
#     # Display the image in a window
#     cv2.imshow('Image', output)

#         # Wait for a key press and close the window when a key is pressed

# import cv2
# import pytesseract as pt
# import pandas as pd
# import os
# from natsort import natsorted

# if __name__ == "__main__":
#     input_folder = "./Processed Image/Adoptme/"
#     image_files = natsorted([f for f in os.listdir(input_folder) if f.endswith('.png')])

#     # Create a list to store the data for the CSV
#     data_list = []

#     for image_file in image_files:
#         input_image_path = os.path.join(input_folder, image_file)
#         # Uncomment this if you want to remove the background from the image
#         # output = remove_background(input_image_path)
#         # cv2.imshow('Image', output)
#         # cv2.waitKey(0)

#         # Read text and bounding box positions from the image file
#         boxes_data = pt.image_to_boxes(input_image_path, lang="eng")
#         lines = boxes_data.split('\n')
#         for line in lines:
#             char_info = line.split()
#             if len(char_info) == 6:  # Ensure there are six elements in the line
#                 char = char_info[0]
#                 # Extract bounding box coordinates
#                 x_min, y_min, x_max, y_max = map(int, char_info[1:-1])
#                 data_list.append({
#                     "Filename": input_image_path,
#                     "Character": char,
#                     "Box Position": (x_min, y_min, x_max, y_max)
#                 })

#     df = pd.DataFrame(data_list)

#     # Save the DataFrame to a CSV file
#     output_csv_path = "Adoptme.csv"
#     df.to_csv(output_csv_path, index=False)

#     print("Bounding box positions for each character saved to CSV:", output_csv_path)