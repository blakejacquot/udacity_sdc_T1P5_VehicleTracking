import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import glob
import os
import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#%matplotlib inline

import random

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from skimage.feature import hog

from scipy.ndimage.measurements import label

# Define a function to return HOG features and visualization
def get_hog_features(img, orient = 9, pix_per_cell = 8, cell_per_block = 2, vis=False, feature_vec=True):  
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features
    
    
# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256), channel='ALL'):
    # Compute the histogram of the color channels separately
    if channel == 'ALL':
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    else:
        hist_features = np.histogram(img[:,:,channel], bins=nbins, range=bins_range)
        #hist_features = np.concatenate(channel_hist[0])
    
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(image, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), channel='ALL'):
    ## Create a list to append feature vectors to
    #features = []
    ## Iterate through the list of images
    #for file in imgs:
    # Read in each one by one
    #image = mpimg.imread(file)
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    else: feature_image = np.copy(image)
    
    #if channel != 'ALL':
    #    tmp_img = feature_image[:,:,channel]
    #    feature_image = tmp_img
    
    # Apply bin_spatial() to get spatial color features
    spatial_features = bin_spatial(feature_image, size=spatial_size)
    # Apply color_hist() also with a color space option now
    hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range, channel=channel)
    # Append the new feature vector to the features list
    #features.append(np.concatenate((spatial_features, hist_features)))
    feature_vector = np.concatenate((spatial_features, hist_features))
    ## Return list of feature vectors
    return feature_vector

def get_all_feature(img, verbose = False):
    # HOG parameters
    orient = 9 # HOG orientations. 6 to 12.
    pix_per_cell = 8  #HOG pixels per cell
    cell_per_block = 2 # HOG cells per block. Normalization happens over block.

    # Color histogram parameters
    cspace_v='HSV'
    spatial_size_v=(32, 32)
    hist_bins_v=32
    hist_range_v=(0, 256)
    channel = 'ALL'

    # Transform for HOG
    img_hog = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # convert to grayscale 2D array
    
    # Get HOG features
    hog_features = get_hog_features(img_hog, orient, 
                            pix_per_cell, cell_per_block, 
                            vis=False, feature_vec=False)
    hog_features_flattened = hog_features.flatten()
    #hog_scaled = hog_features_flattened - np.min(hog_features_flattened)
    #hog_scaled = hog_scaled / (2*np.max(hog_scaled))
    #hog_scaled = hog_scaled - 1
    #hog_scaled = hog_features_flattened - (np.max(hog_features_flattened) + np.min(hog_features_flattened) ) / 2
    #hog_scaled = hog_scaled / np.max(hog_features_flattened)
    #hog_scaled = hog_scaled / np.mean(hog_scaled)

    # Get color histogram features
    hist_features = extract_features(img, cspace=cspace_v, spatial_size=spatial_size_v,
                            hist_bins=hist_bins_v, hist_range=hist_range_v, channel=channel)
    hist_scaled = hist_features - np.mean(hist_features)
    
    ## Scale HOG feature vector
    #X_HOG = np.vstack(hog_features_flattened).astype(np.float64)                   
    ##X1 = hog_features_flattened.astype(np.float64)                   
    #X_HOG_scaler = StandardScaler().fit(X_HOG) # Fit a per-column scaler
    #scaled_X_HOG = X_HOG_scaler.transform(X_HOG) # Apply the scaler to X

    ## Scale histogram feature vector
    #X_hist = np.vstack(hist_features).astype(np.float64)                   
    #X_hist_scaler = StandardScaler().fit(X_hist) # Fit a per-column scaler
    #scaled_X_hist = X_hist_scaler.transform(X_hist) # Apply the scaler to X
    
    # Contatenate feature vector
    #feature_vector = np.concatenate((scaled_X_HOG, scaled_X_hist))
    feature_vector = np.concatenate((hog_features_flattened, hist_features))

    
    ## Scale concatenated feature vector
    #X_all_scalar = StandardScaler().fit(feature_vector)
    #scaled_feature_vector = X_all_scalar.transform(feature_vector)
    
    if verbose == True:
        print('HOG feature characteristics')
        print('Shape of native HOG features', np.shape(hog_features))
        print('Shape of HOG features after flattening', np.shape(hog_features_flattened))
        print('Shape, min, max, median, mean, var', np.shape(hog_features_flattened), np.min(hog_features_flattened), np.max(hog_features_flattened), np.median(hog_features_flattened), np.mean(hog_features_flattened), np.var(hog_features_flattened))
        #print('HOG scaled: Shape, min, max, median, mean, var', np.shape(hog_scaled), np.min(hog_scaled), np.max(hog_scaled), np.median(hog_scaled), np.mean(hog_scaled), np.var(hog_scaled))
        print(' ')
        print('Histogram feature characteristics')
        print('Shape of histogram features', np.shape(hist_features))
        print('Shape, min, max, median, mean, var', np.shape(hist_features), np.min(hist_features), np.max(hist_features), np.median(hist_features), np.mean(hist_features), np.var(hist_features))
        print('Hist scaled: Shape, min, max, median, mean, var', np.shape(hist_features), np.min(hist_scaled), np.max(hist_scaled), np.median(hist_scaled), np.mean(hist_scaled), np.var(hist_scaled))
        print(' ')
        #print('*******************')
        #print('Characteristics of scaled HOG features')
        #print('Shape, min, max, median, mean, var', np.shape(scaled_X_HOG), np.min(scaled_X_HOG), np.max(scaled_X_HOG), np.median(scaled_X_HOG), np.mean(scaled_X_HOG), np.var(scaled_X_HOG))
        #print(' ')
        #print('Characteristics color histogram features')
        #print('Shape, min, max, median, mean, var', np.shape(scaled_X_hist), np.min(scaled_X_hist), np.max(scaled_X_hist), np.median(scaled_X_hist), np.mean(scaled_X_hist), np.var(scaled_X_hist))
        #print(' ')
        #print('*******************')
        print('Characteristics of entire feature vector')
        #print('Shape, min, max, median, mean, var', np.shape(scaled_feature_vector), np.min(scaled_feature_vector), np.max(scaled_feature_vector), np.median(scaled_feature_vector), np.mean(scaled_feature_vector), np.var(scaled_feature_vector))
        print('Shape, min, max, median, mean, var', np.shape(feature_vector), np.min(feature_vector), np.max(feature_vector), np.median(feature_vector), np.mean(feature_vector), np.var(feature_vector))
        print(' ')
        #fig = plt.figure(figsize=(12,4))
        #plt.subplot(121)
        #plt.plot(X_HOG)
        #plt.title('HOG raw')
        #plt.subplot(122)
        #plt.plot(scaled_X_HOG)
        #plt.title('HOG normalized')
        #fig.tight_layout()  
        #print(' ')
        #fig = plt.figure(figsize=(12,4))
        #plt.subplot(121)
        #plt.plot(X_hist)
        #plt.title('Hist raw')
        #plt.subplot(122)
        #plt.plot(scaled_X_hist)
        #plt.title('Hist normalized')
        #fig.tight_layout()  
        #print(' ')
        #fig = plt.figure(figsize=(12,4))
        #plt.subplot(121)
        #plt.plot(feature_vector)
        #plt.title('All raw')
        #plt.subplot(122)
        #plt.plot(scaled_feature_vector)
        #plt.title('All normalized')
        #fig.tight_layout()  
        #print(' ')
        
    return feature_vector