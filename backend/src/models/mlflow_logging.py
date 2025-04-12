
import os

import mlflow
import tensorflow as tf
from tensorflow import keras
import numpy as np

# set the data path -> it's used later as a way to log the dataset 
# alternative way to log the dataset: mlflow.log_input(...)
DATA_PATH = os.path.join("..","data/train/")
DATA_PATH = os.path.abspath(DATA_PATH)


def log_mlflow_run(
    model,
    run_name,
    epochs,
	batch_size,
	loss_function,
	optimizer,
	learning_rate,
	top_dropout_rate,
	model_summary_string,
    run_tag,
    signature_batch,
    val_accuracy,
    test_accuracy,
	custom_params,
    fig
):
    
    '''
    Function that takes as arguments all the parameters/artifacts to be logged in mlflow
    and handles all the mlflow-related logging. Does not return anything.
    
    Parameters
    ----------
    model : keras.Model
        Keras model to be logged.
    run_name : str
        String that will be displayed as the run title in the mlflow UI.
    epochs : non-negative int
        Number of epochs used for training.
	batch_size : non-negative int
        Batch size used in training.
	loss_function : 
        Loss function used in training.
	optimizer : str/object
        Optimizer used in training.
	learning_rate : non-negative float
        Learning rate used in training.
	top_dropout_rate : float between 0 and 1
        Dropout rate used in the dropout layer.
	model_summary_string : str
        String for model summary (comes from a helper function).
    run_tag : str
        String explaining what this run was for
    signature_batch : tensorflow.python.data.ops.take_op._TakeDataset object
        Needed for infer the signature of the model in mlflow.
        Obtained from train_data.take(1).
    val_accuracy : float between 0 and 1
        Validation accuracy to be logged.
    test_accuracy : float between 0 and 1
        Test accuracy to be logged.
	custom_params : dict
        Dictionary for additional parameteres to be logged.
    fig : matplotlib figure object
        Figure to be logged.
    
    '''
    
    # Check if model is a keras model
    if not isinstance(model, keras.Model):
        raise TypeError(
            f"Invalid model object. Expected keras.Model, but got {type(model).__name__} instead."
        )
    
    # Save standard parameters in the dictionary
    params_dict = {
        "epochs": epochs,
        "batch size": batch_size,
        "loss function": loss_function,
        "optimizer": optimizer,
        "learning rate": learning_rate,
        "dense layer dropout rate": top_dropout_rate,
        "dataset": DATA_PATH
        }
    
    # Add custom parameters to the parameter dict
    try:
        params_dict.update(custom_params)
    except:
        raise TypeError("custom_params can only be a dictionary.")

    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    # Sets the current active experiment to the "own_model_training" experiment and
    # returns the Experiment metadata
    mlflow.set_experiment("X-Ray Pneumonia")


    # Define a run name for this iteration of training.
    # If this is not set, a unique name will be auto-generated for your run.

    # Start an MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        
        # Log the hyperparameters
        mlflow.log_params(params_dict)

        # Log the metrics
        
        metrics_dict = {
            "validation accuracy": val_accuracy,
            "test accuracy": test_accuracy
            }
        
        mlflow.log_metrics(metrics_dict)
        
        # Log figures
        mlflow.log_figure(fig, "learn_curve_accuracy.png")

        # log model summary as text artifact
        mlflow.log_text(model_summary_string, "model_summary.txt")

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", run_tag)
        
        # Get input and output examples to help infer the signature of the model
        batch_as_nparray = list(signature_batch)[0][0].numpy()
        input_example = batch_as_nparray[0]
        input_example = np.expand_dims(input_example, axis=0)
        
        prediction_example = model(input_example).numpy()
        
        # Infer the model signature                
        signature = mlflow.models.infer_signature(input_example, prediction_example)

        # Log the model
        mlflow.keras.log_model(
            model = model,
            artifact_path = "model_artifact",
            signature = signature
        )
