# Assignment 2 - Simple image search
This repository contains the code related to Assignment 2 for Visual Analytics.
The script takes a path to an image directory and the filename of a target image within the directory as inputs.
The output of the script is a csv file that holds the chi squared distance between target image and all other images in the directory.

Parameters:
    path: str <path-to-image-dir>
    target_image: str <filename-of-target-image>
Usage:
    image_search.py -p <path-to-image> -t <filename-of-target>
Example:
    $ python3 image_search_MAM.py -p ../data/flowers/ -t image_0001.jpg


For running the code, one has to enter the virtual environment cv101. This can be accessed by running the code:
    
```
$ cd cds-visual
$ source cv101/bin/activate
```
    
After this, the code can be run:
    
```
$ cd src
$ python3 image_search_MAM.py -p path2directory -t target_image
``` 

    
__However, this assumes that the script is saved within a "src" folder.__

The resulting csv file will appear in the current directory (in this case the src folder).
