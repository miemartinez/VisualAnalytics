#!/usr/bin/env python
"""
Specify directory of images and target image, find chi squared distance between target and comparison images, save output as csv
Parameters:
    path: str <path-to-image-dir>
    target_image: str <filename-of-target-image>
Usage:
    image_search.py -p <path-to-image> -t <filename-of-target>
Example:
    $ python3 image_search_MAM.py -p ../data/flowers/ -t image_0001.jpg
## Task
- Save csv showing filename and distance comparing the target image to every image in a directory
"""

# importing libraries
import os
import sys
sys.path.append(os.path.join(".."))
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import argparse

# Define main function
def main():
    
    # argparse 
    ap = argparse.ArgumentParser()
    # adding arguments
    ap.add_argument("-p", "--path", required = True, help= "Path to directory of images")
    ap.add_argument("-t", "--target_image", required = True, help= "Filename of the target image")
    # parsing arguments
    args = vars(ap.parse_args())
    
    # get path to image directory
    image_directory = args["path"]
    # get name of the target image
    target_name = args["target_image"]
    
    # empty dataframe to save data
    data = pd.DataFrame(columns=["filename", "distance"])
    
    # read target image
    target_image = cv2.imread(os.path.join(image_directory, target_name))
    # create histogram for all 3 color channels
    target_hist = cv2.calcHist([target_image], [0,1,2], None, [8,8,8], [0,256, 0,256, 0,256])
    # normalise the histogram
    target_hist_norm = cv2.normalize(target_hist, target_hist, 0,255, cv2.NORM_MINMAX)
    
    # for each image (ending with .jpg) in the directory
    for image_path in Path(image_directory).glob("*.jpg"):
        # only get the image name by splitting the image_path (using dummy variable _)
        _, image = os.path.split(image_path)
        # if the image is not the target image
        if image != target_name:
            # read the image and save as comparison image
            comparison_image = cv2.imread(os.path.join(image_directory, image))
            # create histogram for comparison image
            comparison_hist = cv2.calcHist([comparison_image], [0,1,2], None, [8,8,8], [0,256, 0,256, 0,256])
            # normalise the comparison image histogram
            comparison_hist_norm = cv2.normalize(comparison_hist, comparison_hist, 0,255, cv2.NORM_MINMAX)    
            # calculate the chi-square distance
            distance = round(cv2.compareHist(target_hist_norm, comparison_hist_norm, cv2.HISTCMP_CHISQR), 2)
            # append info to dataframe
            data = data.append({"filename": image, 
                                "distance": distance}, ignore_index = True)
    
    # sort values based on distance to target image
    data = data.sort_values("distance")
    # save as csv in current directory
    data.to_csv(f"{target_name}_comparison.csv")
    # print that file has been saved
    print(f"output file is saved in current directory as {target_name}_comparison.csv")
    # print the filename closest to the target image by choosing the first row
    print(f"The image {data['filename'].iloc[0]} is the closest to the {target_name}")
    
# Define behaviour when called from command line
if __name__=="__main__":
    main()