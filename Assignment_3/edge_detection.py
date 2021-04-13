#!/usr/bin/env python
"""
Specify filepath of image with text, find number of letters in the image, save three images as jpg (image with region of interest (ROI), image cropped to ROI, and image with contours around letters)
Parameters:
    filepath: str <filepath-to-image>
Usage:
    edge_detection.py -f <filepath-to-image>
Example:
    $ python3 edge_detection.py -f /data/Jefferson_Memorial.jpg/
## Task
- Use computer vision to extract specific features from images and save intermediate steps as .jpg
"""

# importing libraries
import os
import sys
sys.path.append(os.path.join(".."))
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Define main function
def main():
    """
    ARGPARSE
    """
    # argparse 
    ap = argparse.ArgumentParser()
    # adding arguments
    ap.add_argument("-f", "--filepath", required = True, help= "Filepath to image")
    # parsing arguments
    args = vars(ap.parse_args())
    # get image filepath
    image_path = args["filepath"]
    
    
    # load image
    image = cv2.imread(image_path)
    
    """
    REGION OF INTEREST
    """
    # defining x and y coordinates for the ROI based on the image shape
    x_start = int(image.shape[0]//2.3)
    y_start = int(image.shape[1]//4.85)
    x_end = int(image.shape[0]//1.12)
    y_end = int(image.shape[1]//1.55)
    
    
    # defining image with green rectangular box (ROI)
    ROI_image = cv2.rectangle(image.copy(), (x_start, y_start), (x_end, y_end), (0,255,0), (2))
    # save ROI image as jpg
    cv2.imwrite("image_with_ROI.jpg", ROI_image)
    # print that file has been saved
    print("image_with_ROI.jpg is saved in your current directory")
   
    
    """
    CROPPING
    """
    # cropping image using numpy slicing and the specified x and y coordinates
    image_cropped = image[y_start:y_end, x_start: x_end]
    # save cropped image as jpg
    cv2.imwrite("image_cropped.jpg", image_cropped)
    # print that file has been saved
    print("image_cropped.jpg is saved in your current directory")
    
    
    """
    CANNY EDGE DETECTION
    """
    # Convert image to grey color scale
    grey_image = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)
    # blur using 3x3 kernel
    blurred = cv2.blur(grey_image, (3,3))
    
    # Get value of the max frequency on the greyscale 
    y, x, _ = plt.hist(grey_image.flatten(), 256, [0,256])
    x_middle = int(x[np.where(y == y.max())])
    # x_max is used to find the min and max value for edge detection
    min_value = int(x_middle - 65)
    max_value = int(x_middle + 5)
    # canny edge detection using blurred image
    canny = cv2.Canny(blurred, min_value, max_value)
    
    
    """
    CONTOURS
    """
    # using np copy function so the contours are not destroying the original image
    (contours, _) = cv2.findContours(canny.copy(), 
                    cv2.RETR_EXTERNAL, # keeping only the outer contours
                    cv2.CHAIN_APPROX_SIMPLE)
    # drawing contours on the original image
    image_letters = cv2.drawContours(image_cropped.copy(), # draw contours on original image
                        contours, # our list of contours
                        -1, # which contours to draw (-1 takes all at once, 1 takes the first contour, 2 the second and so on)
                        (0,255,0), # contour colour
                        2) # contour pixel width
    # save image with contours as jpg
    cv2.imwrite("image_letters.jpg", image_letters)
    # print that file has been saved
    print("image_letters.jpg is saved in your current directory")
    
    print(f"There appears to be {len(contours)} letters in the image! Obs. some might be due to noise and artifacts...")
    
# Define behaviour when called from command line
if __name__=="__main__":
    main()
