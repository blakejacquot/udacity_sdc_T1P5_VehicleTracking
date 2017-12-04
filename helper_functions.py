import numpy as np
import cv2
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from lesson_functions import *
from sklearn.model_selection import train_test_split
import pickle
import time

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

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

def bin_spatial(img, size=(32, 32)):
    """
    Resizes each channel to pre-determined size. This is followed by a 'ravel' operation
    on the numpy array, which returns a single, flattened 1-D feature vector for the image.

    It does not care what color space is used. But we will probably use it for RGB.
    """
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def get_feature_vector(img, orientations, pixels_per_cell, cells_per_block):
    """

    Args:
        img: ndaray of size (x, y, 3). This is the RGB image.

    Returns:
        feature_vector: 1-D vector
    """

    # List to hold feature vectors
    features = []

    # Get spatial binning features
    features_spatial = bin_spatial(img)

    # Convert to YCrCb frames
    img_YCrCb = convert_color(img, conv='RGB2YCrCb')
    img_Y = img_YCrCb[:,:,0]
    img_Cr = img_YCrCb[:,:,1]
    img_Cb = img_YCrCb[:,:,2]

    # Get color histogram features
    features_color_hist = color_hist(img_YCrCb)

    # Get HOG features
    features_hog = get_hog_features(img_Y, orientations, pixels_per_cell, \
        cells_per_block, vis=False, feature_vec=True)

    # Concatenate feature vector
    #features.append(features_hog)
    #features.append(features_spatial)
    #features.append(features_color_hist)
    features.append(np.concatenate((features_spatial, features_color_hist, features_hog)))

    return features



def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
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
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features)))
    # Return list of feature vectors
    return features



def get_labeled_features(image_paths_car, image_paths_not_car, orientations, pixels_per_cell, \
    cells_per_block):
    """
    """

    # Determine how many images to use in training
    num_car_images = len(image_paths_car)
    num_not_car_images = len(image_paths_not_car)
    if num_car_images <= num_not_car_images:
        num_images_to_use = num_car_images
    else:
        num_images_to_use = num_not_car_images

    # Get car features
    features = []
    for index in range(num_images_to_use):
        fname = image_paths_car[index]
        img = cv2.imread(fname)
        curr_feature_vector = get_feature_vector(img, orientations, pixels_per_cell, cells_per_block)
        features.append(curr_feature_vector)
    features_car = np.concatenate(features)

    # Get not car features
    features = []
    for index in range(num_images_to_use):
        fname = image_paths_not_car[index]
        img = cv2.imread(fname)
        curr_feature_vector = get_feature_vector(img, orientations, pixels_per_cell, cells_per_block)
        features.append(curr_feature_vector)
    features_not_car = np.concatenate(features)

    # Stack and scale features
    X = np.vstack((features_car, features_not_car)).astype(np.float64) # Create an array stack of feature vectors
    X_scaler = StandardScaler().fit(X) # Fit a per-column scaler
    scaled_X = X_scaler.transform(X) # Apply the scaler to X

    # Define label vector
    y = np.hstack((np.ones(len(features_car)), np.zeros(len(features_not_car))))


    print(scaled_X.shape, y.shape)

    # Split data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    # Save the data for easy access
    pickle_file = './labeled_features.pickle'
    with open(pickle_file, 'wb') as pfile:
        pickle.dump(
            {
                'train_dataset': X_train,
                'train_labels': y_train,
                'test_dataset': X_test,
                'test_labels': y_test,
                'X_scaler': X_scaler
            },
            pfile, pickle.HIGHEST_PROTOCOL)

def train_model():

    print('Loading pickled data')
    pickle_file = './labeled_features.pickle'
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
        X_train = pickle_data['train_dataset']
        y_train = pickle_data['train_labels']
        X_test = pickle_data['test_dataset']
        y_test = pickle_data['test_labels']
        X_scaler = pickle_data['X_scaler']
        #parameters = pickle_data['parameters']


    print('Training SVC model')
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
    return svc

def demo_HOG(orientations, pixels_per_cell, cells_per_block):
    print('Demonstrating histogram of gradients (HOG)')
    #helper_functions.demo_HOG(args)

    # Set up variables for demo images
    image_demo_path = './images_pipeline_demo' # Hardcoded path
    image_car_0 = os.path.join(image_demo_path, 'image_car_0.png')
    image_notcar_0 = os.path.join(image_demo_path, 'image_notcar_0.png')
    img_car_0 = cv2.imread(image_car_0)
    img_notcar_0 = cv2.imread(image_notcar_0)

    # Make initial figure for inclusion in readme
    f, ((ax0, ax1)) = plt.subplots(1, 2, figsize = (30, 12))
    ax0.imshow(img_car_0)
    ax0.set_title('Car')
    ax1.imshow(img_notcar_0)
    ax1.set_title('Not Car')
    out_path = os.path.join('./output_images', 'car_not_car' + '.png')
    f.savefig(out_path, bbox_inches='tight', format='png')
    print('Saved figure ', out_path)

    # Convert color space
    img_car0_YCrCb = helper_functions.convert_color(img_car_0, conv='RGB2YCrCb')
    img_notcar0_YCrCb = helper_functions.convert_color(img_notcar_0, conv='RGB2YCrCb')
    #Here is an example using the YCrCb color space and HOG parameters of orientations=8, pixels_per_cell=(8, 8) and cells_per_block=(2, 2):


    # Get HOG image and feature vector for RBG image
    img_car0_gray = cv2.cvtColor(img_car_0, cv2.COLOR_RGB2GRAY)
    features_car, img_car_0_hog = helper_functions.get_hog_features(img_car0_gray, orientations, pixels_per_cell, cells_per_block, vis=True, feature_vec=True)
    print(img_car0_gray.shape, features_car.shape, img_car_0_hog.shape)
    img_notcar0_gray = cv2.cvtColor(img_notcar_0, cv2.COLOR_RGB2GRAY)
    features_notcar, img_notcar_0_hog = helper_functions.get_hog_features(img_notcar0_gray, orientations, pixels_per_cell, cells_per_block, vis=True, feature_vec=True)

    # Get HOG image and feature vectors for YCrCb image
    features_Y_car, img_car_0_Y_hog = helper_functions.get_hog_features(img_car0_YCrCb[:,:,0], orientations, pixels_per_cell, cells_per_block, vis=True, feature_vec=True)
    features_Cr_car, img_car_0_Cr_hog = helper_functions.get_hog_features(img_car0_YCrCb[:,:,1], orientations, pixels_per_cell, cells_per_block, vis=True, feature_vec=True)
    features_Cb_car, img_car_0_Cb_hog = helper_functions.get_hog_features(img_car0_YCrCb[:,:,2], orientations, pixels_per_cell, cells_per_block, vis=True, feature_vec=True)
    features_Y_notcar, img_notcar_0_Y_hog = helper_functions.get_hog_features(img_notcar0_YCrCb[:,:,0], orientations, pixels_per_cell, cells_per_block, vis=True, feature_vec=True)
    features_Cr_notcar, img_notcar_0_Cr_hog = helper_functions.get_hog_features(img_notcar0_YCrCb[:,:,1], orientations, pixels_per_cell, cells_per_block, vis=True, feature_vec=True)
    features_Cb_notcar, img_notcar_0_Cb_hog = helper_functions.get_hog_features(img_notcar0_YCrCb[:,:,2], orientations, pixels_per_cell, cells_per_block, vis=True, feature_vec=True)


    # Make car figure demonstrating the operations.
    f, ((ax0, ax1), (ax2, ax3), (ax4, ax5), (ax6, ax7)) = plt.subplots(4, 2, figsize = (8, 12))
    ax0.imshow(img_car_0)
    ax0.set_title('RGB Car')
    ax1.imshow(img_car_0_hog)
    ax1.set_title('HOG image for grayscale of RGB')
    ax2.imshow(img_car0_YCrCb[:,:,0])
    ax2.set_title('Y channel')
    ax3.imshow(img_car_0_Y_hog)
    ax3.set_title('HOG image for Y channel')
    ax4.imshow(img_car0_YCrCb[:,:,1])
    ax4.set_title('Cr channel')
    ax5.imshow(img_car_0_Cr_hog)
    ax5.set_title('HOG image for Cr channel')
    ax6.imshow(img_car0_YCrCb[:,:,2])
    ax6.set_title('Cb channel')
    ax7.imshow(img_car_0_Cb_hog)
    ax7.set_title('HOG image for Cb channel')
    out_path = os.path.join('./output_images', 'car_features' + '.png')
    f.savefig(out_path, bbox_inches='tight', format='png')
    print('Saved figure ', out_path)

    # Make not car figure demonstrating the operations.
    f, ((ax0, ax1), (ax2, ax3), (ax4, ax5), (ax6, ax7)) = plt.subplots(4, 2, figsize = (8, 12))
    ax0.imshow(img_notcar_0)
    ax0.set_title('RGB Not Car')
    ax1.imshow(img_notcar_0_hog)
    ax1.set_title('HOG image for grayscale of RGB')
    ax2.imshow(img_notcar0_YCrCb[:,:,0])
    ax2.set_title('Y channel')
    ax3.imshow(img_notcar_0_Y_hog)
    ax3.set_title('HOG image for Y channel')
    ax4.imshow(img_notcar0_YCrCb[:,:,1])
    ax4.set_title('Cr channel')
    ax5.imshow(img_notcar_0_Cr_hog)
    ax5.set_title('HOG image for Cr channel')
    ax6.imshow(img_notcar0_YCrCb[:,:,2])
    ax6.set_title('Cb channel')
    ax7.imshow(img_notcar_0_Cb_hog)
    ax7.set_title('HOG image for Cb channel')
    out_path = os.path.join('./output_images', 'notcar_features' + '.png')
    f.savefig(out_path, bbox_inches='tight', format='png')
    print('Saved figure ', out_path)