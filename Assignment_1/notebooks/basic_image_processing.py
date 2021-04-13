#!/usr/bin/env python
# coding: utf-8

# ## Assignment 1 - Basic image processing

# __Create or find small dataset of images, using an online data source such as Kaggle. At the very least, your dataset should contain no fewer than 10 images.__
# 
# Find the data in the folder "cars". The folder contains 15 images of different lamborghinis.
# 
# Write a Python script which does the following:
# 
# - __For each image, find the width, height, and number of channels__

# In[ ]:


# Importing libraries
import os
import sys
import numpy as np
#import pandas as pd
import cv2
from pathlib import Path


# In[79]:


# defining image path
image_path = os.path.join("..", "data", "cars")
# for loop to find height, width and channels for each .jpg file
for filename in Path(image_path).glob("*.jpg"):
    # splitting image path to isolate filename
    file_path, filename = os.path.split(filename) 
    path_to_image = os.path.join(file_path, filename)
    # reading in the image
    image = cv2.imread(path_to_image)
    # calculating height, width and channel
    height = image.shape[0]
    width = image.shape[1]
    channel = image.shape[2]
    # printing the width, height and channel for each image
    print(f"{filename} has a width of {width}, a height of {height} and {channel} channels.")
  


# - __For each image, split image into four equal-sized quadrants (i.e. top-left, top-right, bottom-left, bottom-right)__ <br>
#     Couldn't figure out how to make the images equal-sized. <br>
# - __Save each of the split images in JPG format__ 
# 

# In[81]:


# function for splitting
def split_and_save(direction, image, filepath, filename, width, height):
    # calculating half width and height
    split_width = int(width/2)
    split_height = int(height/2)
    # splitting and saving for top left 
    if direction == "top_left":
        split_top_left = image[0:split_width, 0:split_height]
        outfile_top_left = os.path.join(filepath, "split_top_left_" + str(filename))
        cv2.imwrite(outfile_top_left, split_top_left)
    # splitting and saving for top right 
    elif direction == "top_right":
        split_top_right = image[split_width:width, 0:split_height]
        outfile_top_right = os.path.join(filepath, "split_top_right_" + str(filename))
        cv2.imwrite(outfile_top_right, split_top_right)
    # splitting and saving for bottom left 
    elif direction == "bottom_left":
        split_bottom_left = image[0:split_width, split_height:height]
        outfile_bottom_left = os.path.join(filepath, "split_bottom_left_" + str(filename))
        cv2.imwrite(outfile_bottom_left, split_bottom_left)
    # splitting and saving for bottom right 
    elif direction == "bottom_right":
        split_bottom_right = image[split_width:width, split_height:height]
        outfile_bottom_right = os.path.join(filepath, "split_bottom_right_" + str(filename))
        cv2.imwrite(outfile_bottom_right, split_bottom_right)
    # condition for misspellings
    else:
        print("Please choose between 'top_left', 'top_right', 'bottom_left' and 'bottom_right'")
    


# In[82]:


# for loop for splitting and saving all images
for filename in Path(image_path).glob("*.jpg"):
    file_path, filename = os.path.split(filename) 
    path_to_image = os.path.join(file_path, filename)
    # defining where to save the split images
    output_path = os.path.join(file_path, "split_images")
    # reading image
    image = cv2.imread(path_to_image)
    # Finding height, width and channels
    height, width, channel = image.shape
    # Splitting the image and saving top left 
    split_and_save(direction = "top_left", 
                   image = image, 
                   filepath = output_path, 
                   filename = filename,
                   width = width, 
                   height = height)
    # Splitting the image and saving top right
    split_and_save(direction = "top_right", 
                   image = image, 
                   filepath = output_path, 
                   filename = filename,
                   width = width, 
                   height = height)
    # Splitting the image and saving bottom left
    split_and_save(direction = "bottom_left", 
                   image = image, 
                   filepath = output_path, 
                   filename = filename,
                   width = width, 
                   height = height)
    # Splitting the image and saving bottom right
    split_and_save(direction = "bottom_right", 
                   image = image, 
                   filepath = output_path, 
                   filename = filename,
                   width = width, 
                   height = height)


# - __Create and save a file containing the filename, width, height for all of the new images.__ <br>
#     Couldn't get pandas to work. Tried to install in terminal but without success.

# In[83]:


# defining image path
image_path = os.path.join("..", "data", "cars", "split_images")
# for loop to find height, width and channels for each .jpg file
for filename in Path(image_path).glob("*.jpg"):
    # splitting image path to isolate filename
    file_path, filename = os.path.split(filename) 
    path_to_image = os.path.join(file_path, filename)
    # reading in the image
    image = cv2.imread(path_to_image)
    # calculating height, width and channel
    height = image.shape[0]
    width = image.shape[1]
    channel = image.shape[2]
    # printing the width, height and channel for each image
    print(f"{filename} has a width of {width}, a height of {height} and {channel} channels.")


# In[ ]:




