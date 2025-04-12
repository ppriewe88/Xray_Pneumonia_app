from tensorflow import keras
import os

IMGSIZE = 256
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH_TRAIN = os.path.join(PROJECT_PATH, "data/train/")
DATA_PATH_TEST = os.path.join(PROJECT_PATH, "data/test/")

def get_train_val_data(batch_size, img_size, channel_mode):
    
    '''
    Function that returns training and validation image datasets from the
    images found in [project_folder]/data/train.
    
    Parameters
    ----------
    batch_size : positive int
        Size of data batches to be loaded in the memory.
    img_size : positive int
        The size to which the images will be resized to is img_size * img_size.
    channel_mode : {grayscale, rgb, rgba}
        Number of color channels of images from the dataset.
        
    Returns
    -------
    train_data: tf.data.Dataset 
        Training dataset comprising of 80% of the images in the input dataset.
    val_data: tf.data.Dataset 
        Validation dataset comprising of 20% of the images in the input dataset.
    '''
    
    train_data, val_data = keras.utils.image_dataset_from_directory(
        DATA_PATH_TRAIN,
        labels='inferred',              # labels are generated from the directory structure
        label_mode='binary',            # 'binary' => binary cross-entropy
        color_mode=channel_mode,        
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,                   # shuffle images before each epoch
        seed=0,                         # shuffle seed
        validation_split = 0.2,
        subset="both",                  # return a tuple of datasets (train, val)
        interpolation='bilinear',       # interpolation method used when resizing images
        follow_links=False,             
        crop_to_aspect_ratio=False
        )
    return train_data, val_data
    
    
def get_test_data(batch_size, img_size, channel_mode):
    
    '''
    Function that returns the test image dataset from the images 
    found in [project_folder]/test/train.
    
    Parameters
    ----------
    batch_size : positive int
        Size of data batches to be loaded in the memory.
    img_size : positive int
        The size to which the images will be resized to is img_size * img_size.
    channel_mode : {grayscale, rgb, rgba}
        Number of color channels of images from the dataset.
        
    Returns
    -------
    test_data: tf.data.Dataset 
        Test dataset comprising of all the images in the input dataset.
    '''
    
    test_data = keras.utils.image_dataset_from_directory(
        DATA_PATH_TEST,
        labels='inferred',              # labels are generated from the directory structure
        label_mode='binary',            # 'binary' => binary cross-entropy
        color_mode=channel_mode,        
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,                   # shuffle images before each epoch
        seed=0,                         # shuffle seed
        validation_split = None,
        subset=None,                   
        interpolation='bilinear',       # interpolation method used when resizing images
        follow_links=False,             
        crop_to_aspect_ratio=False
        )
    return test_data