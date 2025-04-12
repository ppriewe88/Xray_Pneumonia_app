import numpy as np
from tensorflow import keras
import mlflow
from PIL import Image
import io
import os
from fastapi import HTTPException

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
