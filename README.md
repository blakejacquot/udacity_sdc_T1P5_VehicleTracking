# udacity_sdc_T1P5_VehicleTracking

## Example Usage

    python main.py 0 -v -p /Users/blakejacquot/Desktop/temp/training_images/single_images -d

## Notes

test123

Links for labeled training images of
[cars](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and
[non-cars](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip).

The images come from GTI vehicle image database, KITTI vision benchmark suite, and examples
extracted from the project video.

### Big concepts covered in this unit:
Supervised learning classifiers.
-Linear decision surface: Just a simple line on scatter plot.
-Naive Bayes: Curvy-ish line on scatter plot.
-Support Vector Machine: Maximizes margin (seems sort of like MSE). Good for non-linear classification with kernel trick, resulting in new variable.
-Decision Trees: Make clumpy x-y scatterplot linearly separable by taking it in stages. Prone to overfitting.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training
set of images and train a classifier Linear SVM classifier. (NOT DONE)
* Optionally, you can also apply a color transform and append binned color features, as
well as histograms of color, to your HOG feature vector. (NOT DONE)
* Note: for those first two steps don't forget to normalize your features and randomize a
selection for training and testing. (NOT DONE)
* Implement a sliding-window technique and use your trained classifier to search for
vehicles in images. (NOT DONE)
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement
on full project_video.mp4) and create a heat map of recurring detections frame by frame to
reject outliers and follow detected vehicles. (NOT DONE)
* Estimate a bounding box for vehicles detected. (NOT DONE)

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4
[im1]: ./output_images/car_not_car.png
[im2]: ./output_images/car_features.png
[im3]: ./output_images/notcar_features.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.
[Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md)
is a template writeup for this project you can use as a guide and a starting point.

You are reading it.

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

HOG features are extracted from images in helper_functions.py

    def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
        """

        Args:
            image: (M, N) array
            orient: Number of orientation bins
            pix_per_cell: Size (in pixels) of a cell
            cells_per_block: Number of cells in each block
            vis: Boolean. Visualize HOG.
            feature_vec: TBD

        Returns:
            features: ndarray which is HOG for the image as 1D (flattened) array.
            hog_image: ndarray which is visualization of HOG image
        """
        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(img, orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                                      transform_sqrt=False,
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(img, orientations=orient,
                           pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block),
                           transform_sqrt=False,
                           visualise=vis, feature_vector=feature_vec)
            return features

Below are examples of running this on an image with and without car.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:
![alt text][im2]![alt text][im3]


#### 2. Explain how you settled on your final choice of HOG parameters.

I used
    orientations = 8
    pixels_per_cell = 8
    cells_per_block = 2


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

After extracting features, I used the SVM model. Before training, I normalized with
sklearn.preprocessing.StandardScaler() and split into train, test, validation sets.

After playing around with parameters, I found that the following worked pretty well.

    Spatial Binning of Color: size = (16, 16)
    Histograms of Color: nbins = 32
    Histogram of Oriented Gradient (HOG): orient = 8, pix_per_cell = 8, cell_per_block = 2

****
****
****
****
****
****
****
****
****
Template text below
****



---


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

