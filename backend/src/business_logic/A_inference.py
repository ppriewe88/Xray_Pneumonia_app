import numpy as np
from tensorflow import keras
import mlflow
from PIL import Image
import io
import os
from fastapi import HTTPException
import random
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./")))
from business_logic.B_logging import save_performance_data_csv, save_performance_data_mlflow
from business_logic.D_modelswitch import check_challenger_takeover, switch_champion_and_challenger

' ##############################################################################################'
' ######################### image preprocessing, model loading, prediction #####################'

def return_verified_image_as_numpy_arr(image_bytes):
    '''
    Verification and reformatting function.
    Verifies image type of input. Returns formatted numpy array.
    
    Parameters
    ----------
    
    image_bytes: image as binary stream
        Input Image, converted to bytes.
        
    Returns
    -------
    Validated image in numpy array format. 
    '''   
    try: 
        
        # convert bytes to a PIL image, then ensure its integrity
        image = Image.open(io.BytesIO(image_bytes))
        image.verify() # can't be used if i want to process the image
    
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    # load image again (as it has been deconstructed by .verify())
    validated_image = Image.open(io.BytesIO(image_bytes))

    # convert the PIL image to np.array
    validated_image_as_numpy = np.asarray(validated_image)
    return validated_image_as_numpy

def load_model_from_registry(model_name, alias):
    """
    Is used to load an mlflow model from its registry (i.e., to fetch the corresponding artifact).
    Model is fetched according to given model name and alias. The model and its signature data are returned.

    Parameters
    ----------
    model_name : string
        The registered model's name.
    alias : string
        The registered model's alias.
        
    Returns
    -------
    model: mlflow model
    input_shape: tuple
        Reflects the models required signature shape (specification of data structure for the model input during predictions)
    input_type:
        The models required input data type in element level.
    """

    # start_loading = time.time()
    print("start loading model")
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@{alias}")
    # end_loading = time.time()
    # print("loading time: ", end_loading - start_loading)
    print("model loaded")
    # extract signature
    signature = model.metadata.signature
    input_shape = signature.inputs.to_dict()[0]['tensor-spec']['shape'] 
    input_type = signature.inputs.to_dict()[0]['tensor-spec']['dtype']
    
    return model, input_shape, input_type

def get_modelversion_and_tag(model_name, model_alias):
    """
    Fetches modelversion and tag by given model name and alias.
    Both infos are retrieved from the file system of the mlflow registry of the project.

    Parameters
    ----------
    model_name : string
    model_alias : string
        
    Returns
    -------
    version_number : string
        Version number of registered mlflow model (registry model)
    tag : string
        Tag of registered model's version (registry model)
    """ 

    # get absolute path of the project dir
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    
    # get path of model folder and aliases subfolder (both are used later)
    aliases_path = os.path.join(project_folder ,f"models/mlruns/models/{model_name}/aliases")
    
    # aliases_path = os.path.abspath(os.path.join("..",f"models/mlruns/models/{model_name}/aliases"))
    model_path = os.path.dirname(aliases_path)
    
    
    # find alias (containing version number) in aliases subfolder, read version number from found file
    alias_file = os.path.join(aliases_path, model_alias)
    with open(alias_file, 'r') as file:
        version_number = int(file.read().strip())
        
    # use version number to find folder of model version
    version_dir = os.path.join(model_path, f"version-{version_number}")
    if not os.path.exists(version_dir):
        raise FileNotFoundError(f"Folder {version_dir} does not exist")
    
    # find subfolder containing tags
    tags_dir = os.path.join(version_dir, "tags")
    if not os.path.exists(tags_dir):
        raise FileNotFoundError(f"folder 'tags' in {version_dir} is missing")
    
    # extract titel from first tag-file found
    tag_files = os.listdir(tags_dir)
    if not tag_files:
        raise FileNotFoundError(f"No tags found in {tags_dir}")
    
    tag = tag_files[0].strip()

    return version_number, tag

def resize_image(
    image,
    signature_shape,
    signature_dtype
    ):
    '''
    Function that resizes a grayscale image such that it agrees 
    with the signature and data type of the ML classifier's input.
    
    Parameters
    ----------
    
    image: PIL image/numpy array
        Image to be resized. Can only be in grayscale.
    signature_shape: tuple
        Shape of the ML classifier input.
    signature_dtype: data type
        Data type of the ML classifier input.
        
    Returns
    -------
    image_array: numpy array
        Reshaped numpy array with signature_dtype entries. 
    '''
    # convert image to numpy array
    image_array = np.asarray(image)
    image_array = image_array.reshape((*image_array.shape,1))

    # if ML model input has more than one channel, populate each channel with the same pixel values
    if signature_shape[-1] > 1:
        img_array_tuple = tuple([image_array for i in range(signature_shape[-1])])
        image_array = np.concatenate(img_array_tuple, axis = -1)

    # resizing according to signature_shape. Using helper function from keras
    resized_image = keras.ops.image.resize(
        image_array,
        size = (signature_shape[1], signature_shape[2]),
        interpolation="bilinear",
        )
    
    # converting to numpy and retyping according to signature_type
    image_array = resized_image.numpy().reshape(signature_shape)
    image_array = image_array.astype(signature_dtype)

    return image_array

def make_prediction(model, image_as_array):
    """
    Simple function to return a prediction of a given model on a given input array.

    Parameters
    ----------
    model : mlflow model
        Mlflow model object. Has to be retrieved earlier by pufunc loading (mlflow)
    image_as_array : numpy array
        Image representation.
        
    Returns
    -------
    pred_reshaped: numpy array
        Model prediction as numpy array.
    """

    prediction = model.predict(image_as_array)
    pred_reshaped = float(prediction.flatten())

    return pred_reshaped

' ##############################################################################################'
' ######################### bulk prediction functions##################### #####################'

def get_image_paths(n_samples):
    '''
    Returns a list of the paths of the images to be classified.
    
    Parameters
    ----------
    n_samples: int
        Number of images to be classified.
    
    Returns
    -------
    selected_images: list of Path objects
        List of image paths. 
    '''
    # Get absolute path of the project dir
    project_folder = Path(__file__).resolve().parent.parent

    # paths to image folders
    normal_folder = project_folder / "data" / "test" / "NORMAL"
    pneumonia_folder = project_folder / "data" / "test" / "PNEUMONIA"
    tracking_csv_path = project_folder / "models" / "performance_tracking" / "performance_data_champion.csv"

    # load images of both classes
    normal_images = list(normal_folder.glob("*"))
    pneumonia_images = list(pneumonia_folder.glob("*"))

    # put the images together in one list
    all_images = normal_images + pneumonia_images
    
    # filter out images that were already analysed
    if tracking_csv_path.exists(): 
        # import logging dataframe to get names of already analysed images
        df_performance = pd.read_csv(tracking_csv_path)
        analysed_images = set(df_performance["filename"])

        # filter out already analysed images
        all_images = [image for image in all_images if image.name not in analysed_images]

    if n_samples <= len(all_images):
        # select a random sample from the remaining images
        selected_images = random.sample(all_images, n_samples)
    else:
        print("Chosen no. of images larger than remaining images.")
        print(f"Sending all {len(all_images)} remaining images...")
        selected_images = all_images
        random.shuffle(selected_images)
        
    return selected_images

def predict_log_switch(selected_image_paths):
    """
    Function that takes several image paths as input and classifies the
    corresponding images, logs the results in csv form and optionally
    in mlflow, and performs the switch between challenger and champion
    when needed.
    
    Parameters
    ----------
    selected_images: list of Path objects
        List of image paths returned by the get_image_paths() function. 
    
    Returns
    -------
    None
    """
    # set tracking uri for mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    # set model name
    model_name = "Xray_classifier"
    
    # load all three models and save the outputs for later use
    models = []
    input_shapes = []
    input_types = []
    aliases = ["champion", "challenger", "baseline"]
    
    for alias in aliases:
        # get model and signature
        model, input_shape, input_type  = load_model_from_registry(model_name = model_name, alias = alias)
        # store the outputs 
        models.append(model)
        input_shapes.append(input_shape)
        input_types.append(input_type)
        print(f"Model with alias {alias} loaded.")
        
    for i, image_file in enumerate(selected_image_paths):
        
        # get class from parent folder name
        data_class = image_file.parent.name
        label = 0 if data_class == "NORMAL" else 1
        
        api_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # make predictions and logging for each of the three models
        for alias, model, input_shape, input_type in zip(aliases, models, input_shapes, input_types):
            
            # get model version and tag for logging
            model_version, model_tag = get_modelversion_and_tag(model_name=model_name, model_alias=alias)
            
            # open image and resize it according to model signature
            with Image.open(image_file, "r") as img:
                formatted_image = resize_image(image=img, signature_shape = input_shape, signature_dtype=input_type)
            
            # make prediction
            y_pred = make_prediction(model, image_as_array=formatted_image)
            accuracy_pred = int(label == np.around(y_pred))
            
            # logging and precalculations in csv-file
            logged_csv_data = save_performance_data_csv(alias = alias, 
                                                       timestamp = api_timestamp, 
                                                       y_true = label, 
                                                       y_pred = y_pred, 
                                                       accuracy=accuracy_pred, 
                                                       file_name=image_file.name, 
                                                       model_version=model_version, 
                                                       model_tag=model_tag)

            # switch off mlflow tracking (if needed)
            mlflow_tracking = False 
            if mlflow_tracking:
                # logging in mlflow performance runs, if switched on
                save_performance_data_mlflow(log_counter = logged_csv_data["log_counter"], 
                                                alias = alias, 
                                                timestamp = logged_csv_data["timestamp"], 
                                                y_true = label, 
                                                y_pred = y_pred, 
                                                accuracy = accuracy_pred, 
                                                file_name = logged_csv_data["filename"], 
                                                model_version = model_version, 
                                                model_tag = model_tag)

        # check if switch should be made
        if check_challenger_takeover(last_n_predictions = 20, window = 50):
            switch_champion_and_challenger()
            # swap the challenger and champion aliases
            # instead of loading the models again
            aliases[0], aliases[1] = aliases[1], aliases[0]
        
        print(f"Prediction no. {i+1} of {len(selected_image_paths)} with class {data_class} done.")