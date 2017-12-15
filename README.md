# Vehicle Detection

In this project, the goal is to write a software pipeline to detect vehicles in a video.

[//]: # (Image References)
[image0]: ./examples/carnotcar.png
[image1]: ./examples/hogycrcb.png

[image2]: ./examples/searchboxes.png
[image3]: ./examples/detections.png
[image4]: ./examples/heatmaps.png

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

# Writeup

Code is found in [Vehicle-Detection.ipynb](https://github.com/blakejacquot/udacity_sdc_T1P5_VehicleTracking/blob/master/Vehicle-Detection.ipynb).

Functions are found in [helperfunctions.py](https://github.com/blakejacquot/udacity_sdc_T1P5_VehicleTracking/blob/master/helperfunctions.py) and [lesson_functions.py](https://github.com/blakejacquot/udacity_sdc_T1P5_VehicleTracking/blob/master/lesson_functions.py).

---

### Histogram of Oriented Gradients (HOG)

I started by reading in all the `vehicle` and `non-vehicle` images.

`get_hog_features()` processes on a single channel. I found that `YCrCb` color space worked OK for classification. HOG is processed on each channel separately.

Here is an example of converting a car image to `YCrCb` and then processing HOG.

![alt text][image1]

I trained a linear SVM in `train_test_svm`. It returns the model and scaling parameters.

### Sliding Window Search

Sliding window search is implemented as part of `find_cars()`.

Below are examples of all positive IDs for boxes, corresponding heatmap, and results after thresholding.

![alt text][image2]  
![alt text][image3]  
![alt text][image4]  
---

### Video Implementation

Here's a [link to my video result](./project_out.mp4)

---

### Discussion

I initially trained using grayscale for HOG and HSV for color histogram and spatial features. Though I got good results for the training images, when I got to jpg from the video, things didn't perform well. I'm not sure why, but I got much better results with YCrCb.
