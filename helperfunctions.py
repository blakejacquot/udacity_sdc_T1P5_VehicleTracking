import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib

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
    global svc, X_scaler, heatcache, heatcounter, cachedepth, threshold
    global heatm, heatmm, heatmmm, heatmmmm
    heatm = 0
    heatmm = 0
    heatmmm = 0
    heatmmmm = 0
    heatcounter = 1
    #heatcache = 0
    cachedepth = 5
    threshold = 5 #3
    
    pkl_file = open('data.pkl', 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    svc = data['svc']
    X_scaler = data['X_scaler']
  

def process_video(video_path, output_path):    
    initiate_globals()
    clip1 = VideoFileClip(video_path)
    clip = clip1.fl_image(process_frame )
    clip.write_videofile(output_path, audio=False)
    
def process_frame(image):
    #global svc, X_scaler, heatcache, heatcounter
    #global svc, X_scaler, heatcache, heatcounter, cachedepth, threshold
    global heatm, heatmm, heatmmm, heatmmmm

    #print(threshold)
    
    cspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    spatial_size=(32, 32) # (32,32), (16,16)
    hist_bins=32
    hist_range=(0, 256)
    channel = 'ALL'

    # HOG parameter
    orient = 9 # HOG orientations. 6 to 12.
    pix_per_cell = 8  #HOG pixels per cell
    cell_per_block = 2 # HOG cells per block. Normalization happens over block.
    hog_channel = 'ALL'

    x_start = None
    x_stop = None
    y_start = 400
    y_stop = 656
    xy_window= (128, 128) #(128, 128)
    xy_overlap=(0.5, 0.5) #(0.5, 0.5)

    spatial_feat = True
    hist_feat = True
    hog_feat = True

    scale = 1.5    
    
    out_img, windows = lf.find_cars(image, y_start, y_stop, scale, svc, X_scaler, orient, pix_per_cell,
                                    cell_per_block, spatial_size, hist_bins)

    #out_img, windows = find_cars(image, svc, X_scaler)

    hot_windows = lf.search_windows(image, windows, svc, X_scaler, color_space=cspace, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       
    
    heat = np.zeros_like(image[:,:,0]).astype(np.float)    
    heat = lf.add_heat(heat,hot_windows)

    #print('Heat', np.max(heat), np.shape(heat))
    
    #    heatm = heatmm = heatmmm = heatmmmm = 0

    if heatmmmm == None:
        heatthresh = lf.apply_threshold(heat,threshold)
        heatmmmm = heatmmm
        heatmmm = heatmm
        heatmm = heatm
        heatm = heat
    else:
        heatadd = heat + heatm + heatmm + heatmmm + heatmmmm
        heatthresh = lf.apply_threshold(heatadd,threshold)
        heatmmmm = heatmmm
        heatmmm = heatmm
        heatmm = heatm
        heatm = heat        
            
    heatmap = np.clip(heatthresh, 0, 255)
    labels = label(heatmap)    
    draw_img = lf.draw_labeled_bboxes(np.copy(image), labels)

    
    return draw_img

# Define a single function that can extract features using hog sub-sampling and make predictions
#def find_cars1(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, #spatial_size, hist_bins):
def find_cars(image, svc, X_scaler):

    # Reiterate variables for good measure

    cspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    spatial_size=(32, 32) # (32,32), (16,16)
    hist_bins=32
    hist_range=(0, 256)
    channel = 'ALL'

    # HOG parameter
    orient = 9 # HOG orientations. 6 to 12.
    pix_per_cell = 8  #HOG pixels per cell
    cell_per_block = 2 # HOG cells per block. Normalization happens over block.
    hog_channel = 'ALL'

    x_start = None
    x_stop = None
    y_start = 300
    y_stop = 656

    spatial_feat = True
    hist_feat = True
    hog_feat = True

    scale = 1.5
    
    #image = mpimg.imread(testimgs[0])
    draw_image = np.copy(image)

    image = image.astype(np.float32)/255

    x_start = 600
    x_stop = 1100
    y_start = 350
    y_stop = 500
    xy_window= (8, 8) #(128, 128)
    xy_overlap=(0.5, 0.5) #(0.5, 0.5)    
    windows2 = lf.slide_window(image, x_start_stop=[x_start, x_stop], y_start_stop=[y_start, y_stop], 
                        xy_window=xy_window, xy_overlap=xy_overlap)

    x_start = None
    x_stop = None
    y_start = 350
    y_stop = 656
    xy_window= (32, 32) #(128, 128)
    xy_overlap=(0.2, 0.2) #(0.5, 0.5)    
    windows3 = lf.slide_window(image, x_start_stop=[x_start, x_stop], y_start_stop=[y_start, y_stop], 
                        xy_window=xy_window, xy_overlap=xy_overlap)

    x_start = None
    x_stop = None
    y_start =400
    y_stop = 656
    xy_window= (64, 64) #(128, 128)
    xy_overlap=(0.2, 0.2) #(0.5, 0.5)    
    windows4 = lf.slide_window(image, x_start_stop=[x_start, x_stop], y_start_stop=[y_start, y_stop], 
                        xy_window=xy_window, xy_overlap=xy_overlap)
    
    x_start = None
    x_stop = None
    y_start = 400
    y_stop = 656    
    xy_window= (128, 128) #(128, 128)
    xy_overlap=(0.9, 0.9) #(0.5, 0.5)
    windows1 = lf.slide_window(image, x_start_stop=[x_start, x_stop], y_start_stop=[y_start, y_stop], 
                        xy_window=xy_window, xy_overlap=xy_overlap)

    
    #print('***************')
    #print(windows1, windows2, windows3, windows4)
    windows = windows1 + windows3 + windows4
    #print(windows)

    hot_windows = lf.search_windows(image, windows, svc, X_scaler, color_space=cspace, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       
    #print('Hot windows', hot_windows)
    window_img = lf.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    
    #plt.imshow(window_img)
    #plt.show()

    return window_img, hot_windows
    

def get_current_windows():
    pass
    

def find_cars1(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = lf.convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
        
    # Compute individual channel HOG features for the entire image
    hog1 = lf.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = lf.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = lf.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 

    # 1=8x8 , 2=16x16, 4=32x32, 8 = 64x64, 16 = 128x128
    cells_per_step = 16  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    windows1 = get_window_list(cells_per_step)
    
    get_current_windows()
    
    window_list = [] # BCJ added    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            
            #print(xpos, ypos)
            #print(xpos+nblocks_per_window, ypos+nblocks_per_window)
            #print(' ')
            
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            
            #print(xleft, ytop)
            #print('********')

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = lf.bin_spatial(subimg, size=spatial_size)
            hist_features = lf.color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
                        
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
                # BCJ added
                startx = xbox_left
                endx = xbox_left+win_draw
                starty = ytop_draw+ystart
                endy = ytop_draw+win_draw+ystart                               
                window_list.append(((startx, starty), (endx, endy)))

    return draw_img, window_list

    
    
    
