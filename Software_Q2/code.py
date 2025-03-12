
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import pandas as pd

def lane_masks(image):
    
    
    #converted to hsv to get black-yellow and black-white images
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # White Lane Mask
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # bitwise_and takes the intersection of all the white parts in the image
    white_lane_mask = cv2.bitwise_and(image, image, mask=white_mask)
    white_lane_mask[np.where((white_lane_mask == [0, 0, 0]).all(axis=2))] = [0, 0, 0]  # Ensure black background
    white_lane_mask[np.where((white_lane_mask != [0, 0, 0]).any(axis=2))] = [255, 255, 255]  # Ensure white lanes

    # similarly yellow mask
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    
    yellow_lane_mask = np.zeros_like(image)  
    yellow_lane_mask[yellow_mask > 0] = [0, 255, 255]  

    
    cv2_imshow(white_lane_mask)
    cv2_imshow(yellow_lane_mask)

def preprocess(image):
    
    cv2_imshow(image)
    
    # converted to grayscale for gaussian blurring
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale, (5, 5), 0)
    cv2_imshow(blur)
    
    # canny edge detection with appropriate thresholds
    edges = cv2.Canny(blur, 50, 150)
    cv2_imshow(edges)
    
    # masking the required region of interest to remove unwanted edges
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    imshape = image.shape
    vertices = np.array([
        [
            (0, imshape[0]),  # bottom left
            (imshape[1] * .35, imshape[0] * .6),  # top left
            (imshape[1] * .65, imshape[0] * .6),  # top right
            (imshape[1], imshape[0])  # bottom right
        ]
    ], dtype=np.int32)
    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
    cv2_imshow(masked_edges)
    
    # Hough transform to the masked edges to get the endpoints of all the lines present in the image
    endpoints = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, 50, 10)
    
    
    # Arrays to store line slope and intercept
    left_lines = []
    right_lines = []
    middle_lines = []
    
    # Blank image to draw lines
    line_image = np.copy(image) * 0
    
    # Line drawing for debugging
    for line in endpoints:
        for x1, y1, x2, y2 in line:
            # Draw all detected lines for visualization
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Calculation of slope and intercept
            if abs(x2 - x1) < 1:  # Near-vertical lines
                middle_lines.append((x1 + x2) / 2)  # Average x position
            else:
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                
                # Classify lines by slope
                if -0.9 < m < -0.2:  # Left lane
                    left_lines.append((m, b))
                elif 0.2 < m < 0.9:  # Right lane
                    right_lines.append((m, b))
                # Lines with very small or very large slopes are ignored
    
    # Debugging image for line detection
    debug_lines = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    cv2_imshow(debug_lines)
    
    # Reset line image for final output
    line_image = np.copy(image) * 0
    
    # Process left and right lanes
    if left_lines and right_lines:
        # Calculate median parameters for left lane
        left_m, left_b = np.median(left_lines, axis=0)
        
        # Calculate median parameters for right lane
        right_m, right_b = np.median(right_lines, axis=0)
        
        # Calculate intersection point
        x_intersect = (left_b - right_b) / (right_m - left_m)
        y_intersect = right_m * x_intersect + right_b
        
        # Calculate bottom points
        left_bottom = (imshape[0] - left_b) / left_m
        right_bottom = (imshape[0] - right_b) / right_m
        
        # Draw left and right lanes
        cv2.line(
            line_image,
            (int(left_bottom), imshape[0]),
            (int(x_intersect), int(y_intersect)),
            (255, 0, 0), 10)
        cv2.line(
            line_image,
            (int(right_bottom), imshape[0]),
            (int(x_intersect), int(y_intersect)),
            (255, 0, 0), 10)
    
    # Process middle lane
    if middle_lines:
        # Get median x position of middle lane markers
        middle_x = np.median(middle_lines)
        
        # Check if we also have the intersection point from left/right lanes
        if left_lines and right_lines:
            # Draw middle lane to intersection
            cv2.line(
                line_image,
                (int(middle_x), imshape[0]),
                (int(x_intersect), int(y_intersect)),
                (0, 0, 255), 10)
        else:
            # If no intersection, draw a vertical line
            cv2.line(
                line_image,
                (int(middle_x), imshape[0]),
                (int(middle_x), int(imshape[0] * 0.6)),
                (0, 0, 255), 10)
    
    # If we couldn't detect middle lane through vertical lines
    # get it from the left and right lanes
    elif left_lines and right_lines:
        # Calculating middle point between left and right bottom points
        middle_bottom = (left_bottom + right_bottom) / 2
        cv2.line(
            line_image,
            (int(middle_bottom), imshape[0]),
            (int(x_intersect), int(y_intersect)),
            (0, 0, 255), 10)
    
    # Draw the lines on the image
    lane_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    cv2_imshow(lane_edges)
    
    




image=cv2.imread('/content/Screenshot from 2025-03-11 15-22-34.png')
preprocess(image)
lane_masks(image)






