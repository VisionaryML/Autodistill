import cv2
import numpy as np

# Function to read the text file and extract contour points
def read_contours_from_file(file_path, image_width, image_height):
    contours = []
    with open(file_path, 'r') as f:
        current_contour = []
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                class_id = int(parts[0])  # Extract the class ID
                # Parse the x, y points, assuming that each pair of x, y is a contour point
                for i in range(1, len(parts), 2):
                    x = float(parts[i]) * image_width  # Normalize back to image width
                    y = float(parts[i + 1]) * image_height  # Normalize back to image height
                    current_contour.append((int(x), int(y)))
            # End of a contour
            if len(current_contour) > 0:
                contours.append(np.array(current_contour, dtype=np.int32))
                current_contour = []
    return contours

# Function to draw contours on the image
def draw_contours_on_image(image, contours):
    # Load the image
    
    # Draw each contour (class ID is not used in drawing, but could be for different colors or thicknesses)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  # Draw contours in green with thickness 2
    
    # Display the image with contours
    cv2.imshow('Contours', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = r'D:\Autodistill\datasets\train\images\A_DFS75MG_BayerRG8_050721_182207_57.jpg'  # The path to your image
file_path = r'D:\Autodistill\datasets\train\labels\A_DFS75MG_BayerRG8_050721_182207_57.txt'     # The path to your text file containing contour data

image = cv2.imread(image_path)
# Define image dimensions (replace with the actual size of your image)
image_height,image_width, _ = image.shape # Example dimensions, modify as per your image

# Read contours from the file
contours = read_contours_from_file(file_path, image_width, image_height)

# Draw the contours on the image
draw_contours_on_image(image, contours)
