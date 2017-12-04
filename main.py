"""

Example usage:
  python main.py 0 --verbosity 0 --proc_path ./test_images
  python main.py 1 --verbosity 1 --proc_path ./
  python main.py 0 --verbosity 0 --proc_path /Users/blakejacquot/Desktop/temp/training_images/single_images

Pipeline:
In each frame look for vehicles with sliding window technique.
Record every position with a positive ID on car
If windows overlap, assign position of detection to the centroid.
Filter out false positives by comparing one frame to the next
Estimate how centroid is moving frame to frame

First, Decide what features to use: probably something color and gradient based, b
Second, choose and train a classifier. SVM is prbably best for speed and accuracy
Third, implement sliding window technique to search for vehicles in test images. Try multiscale and  tilting schemes
to see what works best. But minimize the number of search windows (don't search sky)
Fourth, try on video and reject spurious detections

"""

# Import standard libraries
#import glob
#import os
#import math
#import numpy as np
import matplotlib.pyplot as plt
#import pickle
#import statistics

from moviepy.editor import VideoFileClip

import helper_functions
import lesson_functions

import argparse
import cv2
import glob
import os
import sys


def proc_pipeline(img, video = '0'):
    """ Process verbose image pipline on an image with intermediate states plotted
    and saved.

    Args:

    Returns:

    """

    img_result = img
    return img_result


def proc_video_pipeline(img):
    result = proc_pipeline(img, video = 1)
    return result

def main(args):
    """
    Docstring
    """

    # Set up HOG variables
    orientations = 8
    pixels_per_cell = 8
    cells_per_block = 2

    if args.demo_HOG:
        helper_functions.demo_HOG(orientations, pixels_per_cell, cells_per_block)



#    if args.image_or_vid == 0: # Image
#        print('Processing images')
#        search_phrase = os.path.join(args.proc_path, '*.png')
#        print(search_phrase)
#        images = glob.glob(search_phrase)
#        print(images)
#        for fname in images:
#            img = cv2.imread(fname)
#            print(fname)
#            curr_name = fname[-9:-4]
#            print(curr_name)
#            #ret_img = proc_pipeline(objpoints, imgpoints, img, verbose, outdir = dir_output_images, name = curr_name)

    if args.train_model: # Image
        # Make search terms for cars and not cars
        path_not_car = os.path.join(args.proc_path, 'non-vehicles', '**', '*.png')
        path_car = os.path.join(args.proc_path, 'vehicles','**', '*.png')

        # Get file paths of cars and not cars
        image_paths_not_car = glob.glob(path_not_car, recursive=True)
        image_paths_car = glob.glob(path_car, recursive=True)

        # Train model
        #helper_functions.get_labeled_features(image_paths_car, image_paths_not_car, orientations, pixels_per_cell, cells_per_block)
        svc = helper_functions.train_model()


if __name__ == "__main__":
    # Make command line parser and add required and optional arguments
    parser = argparse.ArgumentParser()
    #parser.add_argument('-i', '--image_or_vid', type=int, help = '0 = jpg, 1 = mp4') # required
    parser.add_argument('-v', '--verbosity', action = 'store_true', help="Increase output verbosity") # optional
    parser.add_argument('-p', '--proc_path', help="Path to single video or directory of images") # optional
    parser.add_argument('-d', '--demo_HOG', action = 'store_true', help="Enable HOG demo") # optional
    parser.add_argument('-t', '--train_model', action = 'store_true', help="Process test images") # optional
    args = parser.parse_args()
    main(args)