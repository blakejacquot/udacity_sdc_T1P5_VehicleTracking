# Vehicle Detection

In this project, the goal is to write a software pipeline to detect vehicles in a video.

[//]: # (Image References)
[image1]: ./examples/hog_img.png

[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

# Writeup

Code is found in [workspace.ipynb](https://github.com/blakejacquot/udacity_sdc_T1P5_VehicleTracking/blob/master/workspace.ipynb).

All functions are found in [helper_functions.py](https://github.com/blakejacquot/udacity_sdc_T1P5_VehicleTracking/blob/master/helper_functions.py).

---

### Histogram of Oriented Gradients (HOG)

I started by reading in all the `vehicle` and `non-vehicle` images.

`get_hog_features()` processes on a single channel. I found that `YCrCb` color space worked OK for classification. HOG is processed on each channel separately.

I tried various parameters and found the following worked well. Since I also used color histogram features and spatial features, I'm including all parameters below.

colorspace='YCrCb' # RGB, HSV, LUV, HLS, YUV, YCrCb  
orient=9  
pix_per_cell=8  
cell_per_block=2  
hog_channel='ALL'  
spatial_size=(16, 16)  
hist_bins=16  
hist_range=(0, 256)  

Here is an example of converting a car image to `YCrCb` and then processing HOG.

![alt text][image1]

I trained a linear SVM in `train_test_svm`. It returns the model and scaling parameters.

### Sliding Window Search

Sliding window search is implemented as part of `find_cars()`. It is similar to that used in the lessons. I didn't use any overlap and instead divided the searchable area into even 64x64 squares. I didn't search in the skyline to try and speed up the process.

---

### Video Implementation

Here's a [link to my video result](./project_out.mp4)

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:



---

### Discussion

I initially trained using grayscale for HOG and HSV for color histogram and spatial features. Though I got good results for the training images, when I got to jpg from the video, things didn't perform well. I'm not sure why, but I got much better results with YCrCb.
