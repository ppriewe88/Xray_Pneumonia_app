import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Query, Response
from enum import Enum
import mlflow
from datetime import datetime
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from business_logic import A_inference as inf
from business_logic import B_logging as lg
from business_logic import C_reporting as rp
from business_logic import D_modelswitch as ms
from fastapi.middleware.cors import CORSMiddleware # middleware. requirement for frontend-suitable endpoint
import matplotlib.pyplot as plt
import io

""" 
run app by running "fastapi run FastAPIserver.py" in terminal.
Go to localhost = 127.0.0.1. Add "/docs" to url to get to API-frontend and check endpoints!
Works when called from any directory level in project folder. Best, start from subfolder "api". 
Here the explicit call to be run from terminal: uvicorn api_server:app --host 0.0.0.0 --port 8000 (127.0.0.1 for local)
"""


' ######################### helper class for label input in prediction endpoint #################'
# class for input in uploading-endpoint
class Label(int, Enum):
    NEGATIVE = 0
    POSITIVE = 1

' ################################################ creating app  ################################'
# make app
app = FastAPI(title = "Deploying an ML Model for Pneumonia Detection")

" ################################ middleware block for frontend-suitable endpoint ###############"
# CORS-Middleware. Required for communication with frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # allow requests from frontend
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP-methods
    allow_headers=["*"],  # allow all headers
)

' ################################################## root endpoint ####################################################'
# root
@app.get("/")
def home():
    """
    Serves as root for API.
    """
    return "root of this API"


' ############################### frontend-suitable model serving/prediction endpoint ###############################'
# endpoint for uploading image
@app.post("/upload_image_from_frontend")
async def upload_image_and_integer_from_frontend( 
    label: int = Form(...),
    file: UploadFile = File(...)
):
    """
    Lets the user upload an image file (no directory restricions, but type validation included) 
    and insert the label of the image (0=normal or 1=pneumonia).

    Image file will be passed through preprocessing and then to a classifier model.
    User will get back classications of up to three models 
    (floats between 0 and 1, represents class 1 probability) with the aliases champion, challenger, and baseline.

    Results (i.e. performance) of the classifiers will as well be logged into csv-files, and into mlflow-logged runs.
    Hence, all information of the given predictions is returned to the user and tracked in the file system.

    Parameters
    ----------
    label : object of class Label, see definition on top of this script
        Hold as human level prediction of the image
    file : UploadFile (FastAPI-form)
        Serves byte object of input file.
        
    Returns
    -------
    y_pred_as_str : string containing dictionaries
        Contains three nested dictionaries with prediction values and logging parameters. 
        One for each model alias, i.e. champion, challenger, baseline.
    """
    label = Label(label)

    # read the uploaded file into memory as bytes
    image_bytes = await file.read()

    # validate image and return as numpy
    img = inf.return_verified_image_as_numpy_arr(image_bytes)

    # set tracking uri for mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    # vessel for API-output
    y_pred_as_str = {}
    
    model_name = "Xray_classifier"
    api_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # ########################### load, predict, log metric for champion and challenger ################'
    for  alias in ["champion", "challenger", "baseline"]:
        
        # get model and signature
        model, input_shape, input_type  = inf.load_model_from_registry(model_name = model_name, alias = alias)
        
        # get model version and tag for logging
        model_version, model_tag = inf.get_modelversion_and_tag(model_name=model_name, model_alias=alias)

        # resize image according to signature
        formatted_image = inf.resize_image(image=img, signature_shape = input_shape, signature_dtype=input_type)
        
                # make prediction
        y_pred = inf.make_prediction(model, image_as_array=formatted_image)
        accuracy_pred = int(label == np.around(y_pred))

        # logging and precalculations in csv-file
        logged_csv_data = lg.save_performance_data_csv(alias = alias, 
                                                       timestamp = api_timestamp, 
                                                       y_true = label.value, 
                                                       y_pred = y_pred, 
                                                       accuracy=accuracy_pred, 
                                                       file_name=file.filename, 
                                                       model_version=model_version, 
                                                       model_tag=model_tag)

        # switch off mlflow tracking (if needed)
        mlflow_tracking = False 
        if mlflow_tracking:
            # logging in mlflow performance runs, if switched on
            lg.save_performance_data_mlflow(log_counter = logged_csv_data["log_counter"], 
                                            alias = alias, 
                                            timestamp = logged_csv_data["timestamp"], 
                                            y_true = label, 
                                            y_pred = y_pred, 
                                            accuracy = accuracy_pred, 
                                            file_name = logged_csv_data["filename"], 
                                            model_version = model_version, 
                                            model_tag = model_tag)
            
        # update dictionary for API-output
        y_pred_as_str.update({f"prediction {alias}": str(y_pred)})
    
    print(f"Currently at run with log_counter number {logged_csv_data['log_counter']}.")
    # check if switch should be made
    if ms.check_challenger_takeover(last_n_predictions = 20, window = 50):
        ms.switch_champion_and_challenger()

    return y_pred_as_str


' ############################### bulk prediction endpoint ##########################################################'
# endpoint for analysing more images
@app.post("/predict_several_images")
async def predict_several_images( 
    n_samples: int
):
    """
    Classifies several images without needing to load the keras models several times.
    The images are chosen randomly.
    
    Parameters
    ----------
    n_samples: int
        Number of images to be classified.
    
    Returns
    -------
        String confirming that all images were succesfully classified.
    """
    
    # get the image paths
    selected_image_paths = inf.get_image_paths(n_samples)

    # peform classification + logging + model switch when needed
    inf.predict_log_switch(selected_image_paths)
       
    return "All predictions done."

' ############################### performance review endpoint ############################################################'
# endpoint for uploading image
@app.post("/get_performance_review_from_mlflow")
async def get_performance_mlflow(
    last_n_predictions: int,
    ):
    """
    Endpoint to provide performance report, based on existing mlflow tracking experiments.

    Returns global average values, statistics of last_n_predictions, and confusion matrix (via function call).

    Parameters
    ----------
    last_n_predictions : int
        Serves for function call to get report. Specifically needed for calculation of average value of last n runs.
    
    Returns
    -------
    performance_dict : dictionary
        Contains three dictionaries with performance tracking values of champion, challenger, baseline.
    """

    # gets the dictionary for all three models
    performance_dict = rp.get_performance_indicators_mlflow(num_steps_short_term = last_n_predictions)

    return performance_dict


' ############################### performance review endpoint CSV #####################################################'
# endpoint for uploading image
@app.post("/get_performance_review_from_csv")
async def get_performance_csv(
    last_n_predictions: int,
    ):
    """
    Endpoint to provide performance report, based on csv-loggings of tracked predictions.

    Returns global average values, statistics of last_n_predictions, and confusion matrix (via function call).

    Parameters
    ----------
    No parameters
    
    Returns
    -------
    merged_csv_dict : dictionary
        Contains three dictionaries with performance tracking values of champion, challenger, baseline.
    """
    # get results generated from csv
    csv_perf_dict_champion = rp.get_performance_indicators_csv(alias = "champion", last_n_predictions=last_n_predictions)
    csv_perf_dict_challenger = rp.get_performance_indicators_csv(alias = "challenger",last_n_predictions=last_n_predictions)
    csv_perf_dict_baseline = rp.get_performance_indicators_csv(alias = "baseline", last_n_predictions=last_n_predictions)
    merged_csv_dict = {
    **csv_perf_dict_baseline,
    **csv_perf_dict_challenger,
    **csv_perf_dict_champion,
    }

    return merged_csv_dict


' ######################## plotting endpoint: performance curve and aliases ############################################'
# endpoint for plot generation
@app.post("/get_comparsion_plot")
async def plot_model_comparison(window: int = 50):
    '''
    Endpoint that displays a plot showing the moving average accuracy
    of the champion and challenger models.  
    
    Plot also indicates, which underlying model is champion or challenger at which run number.
    '''

    # create the figure
    figure = rp.generate_model_comparison_plot(window, scaling =  "log_counter")

    # create an in-memory buffer to hold the figure
    buffer = io.BytesIO()
    
    # save the plot in the buffer as a png
    plt.savefig(buffer, format="png")
    
    # close the fig
    plt.close(figure)
    
    # move the file pointer back to the start of the buffer so it can be read
    buffer.seek(0)
    
    # extract the binary image from the buffer
    binary_image = buffer.getvalue()
    
    # send the binary image as a png response to the client
    return Response(binary_image, media_type="image/png")

' ######################## plotting endpoint: confusion matrix #################################################'
# endpoint for plot generation
@app.post("/get_confusion_matrix_plot")
async def plot_confusion_matrix(window: int = 50):
    '''
    Endpoint that displays a plot showing the confusion matrix of the champion model for the last n predictions.  
    '''

    # create the figure
    figure = rp.generate_confusion_matrix_plot(window)
    
    # create an in-memory buffer to hold the figure
    buffer = io.BytesIO()
    
    # save the plot in the buffer as a png
    plt.savefig(buffer, format="png")
    
    # close the fig
    plt.close(figure)
    
    # move the file pointer back to the start of the buffer so it can be read
    buffer.seek(0)
    
    # extract the binary image from the buffer
    binary_image = buffer.getvalue()
    
    # send the binary image as a png response to the client
    return Response(binary_image, media_type="image/png")

' ################################ host specification ########################################################### '

# my localhost adress
host = "127.0.0.1"

# run server
if __name__ == "__main__":
    uvicorn.run(app, host=host, port=8000, root_path="/")
# GUI at http://127.0.0.1:8000/docs


