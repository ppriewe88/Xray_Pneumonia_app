import numpy as np
import mlflow
import os
from mlflow import MlflowClient
import csv
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns


' ##############################################################################################'
' ######################### logging of prediction data #########################################'

def save_performance_data_csv(alias, timestamp, y_true, y_pred, accuracy, file_name, model_version, model_tag):
    """
    Recieves data from a model's prediction to generate performance review. 
    Saves the retrieved data and some additional calculations in a csv-file under a specified path.
    Also returns the data for further processing (i.e. mlflow-logging).

    Parameters
    ----------
    alias : string
        Alias of mlflow registry model version
    timestamp : string
        Contains time of API call.
    y_true : integer (0 or 1)
        True label of image
    y_pred : float (0 <= y_pred <=1)
        Predicted label of image
    accuracy : int (0 or 1)
        Accuracy of prediction
    file_name: string
        Name of image file used for prediction
    model_version : int
        Version number of mlflow registry model version
    model_tag : string
        Tag of mlflow registry model version

    Returns
    -------
    data : dictionary
        Dictionary of data to be logged into csv-file.
    """ 

    # get absolute path of the project dir to later find required csv-files
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # get path of folder for performance tracking (where csv files are located)
    tracking_path = os.path.join(project_folder ,f"models/performance_tracking")
    
    # make folder, if not existing yet
    os.makedirs(tracking_path, exist_ok=True)
    file_path = os.path.join(tracking_path, f'performance_data_{alias}.csv')
    
    # initializing standard values for cumulative and global values
    log_counter = 1
    
    # Calculate consecutive values from last row's values and current values
    if os.path.exists(file_path):
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            
            if rows:
                last_row = rows[-1]
                # get values of last row to determine cumulations and global accuracy
                log_counter = int(last_row['log_counter']) + 1

    # prepare data for output (formatting)
    data = {
        'log_counter': log_counter,
        'timestamp': timestamp,
        'y_true': y_true,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'filename': file_name,
        'model_version': model_version,
        'model_tag': model_tag,
        "model_alias": alias,
        "model_switch": False
    }
    
    # Check if file exists already
    file_exists = os.path.isfile(file_path)
    
    # Open file in append mode 
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        # Write header only if file is newly created
        if not file_exists:
            writer.writeheader()
        # Append new row
        writer.writerow(data)

    # print runtime and execution confirmation
    #print(f"Data has been saved in {file_path}.")

    return data

def save_performance_data_mlflow(log_counter, alias, timestamp, y_true, y_pred, accuracy, file_name, model_version, model_tag):
    """
    For a given alias, it stores the received logging data from model predicitons (runs) in a unique mlflow run of the corresponding experiment.
    
    Parameters
    ----------
    log_counter: int
        Run number of (csv-logged) run that is to be stored (note: unique mlflow run number usually differs!)
    alias : string
        Alias of mlflow registry model version
    timestamp : string
        Contains time of API call.
    y_true : integer (0 or 1)
        True label of image
    y_pred : float (0 <= y_pred <=1)
        Predicted label of image
    accuracy : int (0 or 1)
        Accuracy of prediction
    file_name: string
        Name of image file used for prediction
    model_version : int
        Version number of mlflow registry model version
    model_tag : string
        Tag of mlflow registry model version

    Returns
    -------
    None
    """ 
    
    # set experiment name for model (logging performance for each model in separate experiment)
    mlflow.set_experiment(f"performance {alias}")

    # logging of metrics
    with mlflow.start_run():
        
        # log the metrics
        metrics_dict = {
            'log counter': log_counter,
            "y_true": y_true,
            "y_pred": y_pred,
            "accuracy": accuracy,
            }
        mlflow.log_metrics(metrics_dict)

        # log model version and tag
        params = {
            'timestamp': timestamp,
            "model version": model_version,
            "model tag": model_tag,
            'image file name': file_name,
            }
        mlflow.log_params(params)
