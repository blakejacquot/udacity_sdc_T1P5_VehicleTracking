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

    if args.demo_HOG == 1:
        print('Demonstrating histogram of gradients (HOG)')
        #helper_functions.demo_HOG(args)
        image_demo_path = './images_pipeline_demo' # Hardcoded path
        image_car_0 = os.path.join(image_demo_path, 'image_car_0.png')
        image_notcar_0 = os.path.join(image_demo_path, 'image_notcar_0.png')

        img_car_0 = cv2.imread(image_car_0)
        img_notcar_0 = cv2.imread(image_notcar_0)

        # Make base figure
        f, ((ax0, ax1)) = plt.subplots(1, 2, figsize = (30, 12))
        ax0.imshow(img_car_0)
        ax0.set_title('Car')
        ax1.imshow(img_notcar_0)
        ax1.set_title('Not Car')
        out_path = os.path.join('./output_images', 'car_not_car' + '.png')
        f.savefig(out_path, bbox_inches='tight', format='png')



#        helper_functions.get_hog_features(img, orient, pix_per_cell, cell_per_block,
#                        vis=False, feature_vec=True)











    if args.image_or_vid == 0: # Image
        search_phrase = os.path.join(args.proc_path, '*.png')
        images = glob.glob(search_phrase)
        for fname in images:
            img = cv2.imread(fname)
            print(fname)
            curr_name = fname[-9:-4]
            print(curr_name)
            #ret_img = proc_pipeline(objpoints, imgpoints, img, verbose, outdir = dir_output_images, name = curr_name)

    if args.image_or_vid == 1: # Video
        output = 'project_video_processed.mp4'
        in_clip = VideoFileClip("project_video.mp4")
        out_clip = in_clip.fl_image(proc_video_pipeline)
        out_clip.write_videofile(output, audio=False)

    helper_functions.hello_world()


if __name__ == "__main__":

    # Make command line parser and add required and optional arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('image_or_vid', type=int, help = '0 = jpg, 1 = mp4') # required
    parser.add_argument('-v', '--verbosity', type=int, help="Increase output verbosity") # optional
    parser.add_argument('-p', '--proc_path', help="Path to single video or directory of images") # optional
    parser.add_argument('-d', '--demo_HOG', type=int, help="0 or 1") # optional

    args = parser.parse_args()

    # Set internal variables based on arguments.
    #if args.image_or_vid == 0:
    #    print('Processing jpg')
    #else:
    #    print('Processing mp4')

    #if args.verbosity == 1:
    #    print('Verbosity turned on')
    #else:
    #    print('Verbosity turned off')

    main(args)