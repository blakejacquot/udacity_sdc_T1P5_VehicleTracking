import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib

from collections import deque

import glob
import os
import time

import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import random

import lesson_functions as lf

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from skimage.feature import hog

from scipy.ndimage.measurements import label

from moviepy.editor import VideoFileClip
import lesson_functions as lf

def print_hello():
    print('Goodbye world')
    
def get_car_notcar_paths(car_path, notcar_path, verbose = False):
    cars = []
    notcars = []
    
    # Get vehicle and non-vehicle paths
    cars = glob.glob(os.path.join(car_path, '*', '*.png'))
    notcars = glob.glob(os.path.join(notcar_path, '*', '*.png'))

    if verbose == True:
        print('Number of car images = ', len(cars))
        print('Number of not car images = ', len(notcars))
        rand_car = cars[random.randint(1, len(cars))]
        rand_notcar = notcars[random.randint(1, len(notcars))]
        print('Random car path = ', rand_car)
        print('Random not car path = ', rand_notcar)
        
        # Plot examples
        img_car = mpimg.imread(rand_car)
        img_notcar = mpimg.imread(rand_notcar)
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(img_car)
        plt.title('Example Car RGB Image')
        plt.subplot(122)
        plt.imshow(img_notcar)
        plt.title('Example Not Car RGB Image')
        plt.show()
        
        print('Mean, min, max of an image', np.mean(img_car), \
              np.min(img_car), np.max(img_car))
        
    return cars, notcars

def train_model(car_features, notcar_features):
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)  

    print('X shape', np.shape(X))
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    print('scaled_X shape', np.shape(scaled_X))


    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    print('y', np.shape(y))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    print('X_train ', np.shape(X_train))


    #print('Using spatial binning of:',spatial,
    #    'and', histbin,'histogram bins')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    
    print(type(svc), type(X_scaler))
    
    return svc, X_scaler


def initiate_globals():
    global svc, X_scaler, q
    q = deque()
    pkl_file = open('data.pkl', 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    svc = data['svc']
    X_scaler = data['X_scaler']
  

def process_video(video_path, output_path):    
    initiate_globals()
    clip1 = VideoFileClip(video_path)
    clip = clip1.fl_image(process_frame)
    clip.write_videofile(output_path, audio=False)
    
def process_frame(image, threshold = 6, cachedepth = 10, verbose_return = False):    
    cspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    spatial_size=(32, 32) # (32,32), (16,16)
    hist_bins=32

    # HOG parameter
    orient = 9 # HOG orientations. 6 to 12.
    pix_per_cell = 8  #HOG pixels per cell
    cell_per_block = 2 # HOG cells per block. Normalization happens over block.
    hog_channel = 'ALL'

    y_start = 400
    y_stop = 656

    spatial_feat = True
    hist_feat = True
    hog_feat = True

    windows = []
  
    #Get windows for searching at different scales. Larger scale = larger cars
    scale = 1.0
    out_img, curr_windows = lf.find_cars(image, y_start, y_stop, scale, svc,
                                         X_scaler, orient, pix_per_cell,
                                         cell_per_block, spatial_size, hist_bins)

    windows += curr_windows
    
    scale = 2.0
    out_img, curr_windows = lf.find_cars(image, y_start, y_stop, scale, svc,
                                         X_scaler, orient, pix_per_cell,
                                         cell_per_block, spatial_size, hist_bins)
    
    windows += curr_windows
    
    # Find detection events
    hot_windows = lf.search_windows(image, windows, svc, X_scaler,color_space=cspace,
                                    spatial_size=spatial_size, hist_bins=hist_bins, 
                                    orient=orient, pix_per_cell=pix_per_cell, 
                                    cell_per_block=cell_per_block, 
                                    hog_channel=hog_channel,
                                    spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)                       
    
    heat = np.zeros_like(image[:,:,0]).astype(np.float)    
    currheat = lf.add_heat(heat,hot_windows)

    # Get composite heat map with queue
    #print(cachedepth, len(q))
    heatadd = np.zeros_like(image[:,:,0]).astype(np.float)
    if len(q) < cachedepth:
        heatadd += currheat
        q.appendleft(currheat)
    else:
        #heatadd = currheat
        q.pop()
        q.appendleft(currheat)
        for h in q:
            heatadd += h

    
    heatthresh = lf.apply_threshold(heatadd, threshold)

        
    # Apply threshold to heatmap
    heatmap = np.clip(heatthresh, 0, 255)
    labels = label(heatmap)    
    draw_img = lf.draw_labeled_bboxes(np.copy(image), labels)

    if verbose_return == True:
        return draw_img, windows, currheat, labels
    else:
        return draw_img

