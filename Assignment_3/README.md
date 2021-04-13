# Assignment 3 - Sentiment Analysis
This repository contains all of the code and data related to Assignment 3 for Visual Analytics.
In the data folder there is a jpg file of the text at Jefferson Memorial. The output of the python script is also provided in the data folder.

The script edge_detection.py takes the filepath to an image as input.
The outputs of the script are three jpg files that display the region of interest (ROI) on the original image, image cropped to the ROI and the original image with contours around letters, respectively. <br>
__Parameters:__ <br>
```
    filepath: str <filepath-to-image> 
```
    
__Usage:__ <br>
```
    edge_detection.py -f <filepath-to-image>
```
    
__Example:__ <br>
```
    $ python3 edge_detection.py -f /data/Jefferson_Memorial.jpg/
```

No other dependencies are needed than what is in the cv101 environment :) 

Activate the environment and then you can run the script with the dependencies:
```
$ cd /<directory_of_virtual_environment>/
$ source cv101/bin/activate
$ cd /<directory_of_the_python_script>/
$ python3 edge_detection.py -f filepath-to-image
```
The resulting jpg files will appear in the current directory (in this case in the directory of the python script).
