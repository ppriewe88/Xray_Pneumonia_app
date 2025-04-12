"""
This script contains the training routine for fine_tuning of artifacts that were generated during transfer learning.
It also contains the setup for experiment tracking via mlflow.
If you run this script to train, and you want to log with mlflow, you have to start the tracking server of mlflow. 
To do so, in the directory of this script (i.e. folder models), run the following command in terminal to start mlflow server for tracking experiments:

mlflow server --host 127.0.0.1 --port 8080

Then check the localhost port 8080 to access the MLFlow GUI for tracking!
Run this script to conduct training experiments (runs). If mlflow server is running, the experiment will be tracked as a run.
You will find the result in the mlflow tracking server ("backend-store"), i.e. models/mlartifacts/[...newly generated run id...]
"""


from tensorflow import keras
from keras.saving import load_model
import os
import tempfile
import time
import training_helpers
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from data.helpers import get_train_val_data, get_test_data, IMGSIZE
from mlflow_logging import log_mlflow_run

# %%
' ################# enter transfer learning experiment ID + run ID (mlflow GUI), that you want to finetune  #######'

# manually selected experiment and run ID and run name. Extracted from MLFlow frontend.
mlflow_experiment_ID = "168696106470894349"
mlflow_run_ID = "863b7a1434c040b89c06974255b10e13"
mlflow_old_run_name = "optimized params"
mlflow_run_name = f'{mlflow_old_run_name} - finetuning nr. 2'

# %%
' ################### extract params logged during transfer learning ######################'

# get paths of the run, to later extract the stored model artifact and the logged parameters
run_path = os.path.join("..", f"models/mlruns/{mlflow_experiment_ID}/{mlflow_run_ID}")
params_path = os.path.join(run_path, "params")

# empty dict as vessel to extract logged parameters
params = {}

if os.path.exists(params_path):
    # list file names
    param_files = os.listdir(params_path)
    
    # extract param from each file
    for param_file in param_files:
        # get file path
        param_file_path = os.path.join(params_path, param_file)
        # open file, extract value
        with open(param_file_path, "r") as file:
            param_value = file.read().strip()
        # retrieve numerical representation, if possible
        if param_value.isdigit():
            param_value = int(param_value)
        elif param_value.isnumeric():
            param_value = float(param_value)
        # store extracted value in parameters dict
        params[param_file] = param_value
else:
    print("params folder not found! Check if experiment and run ID are correct and exist!!")


# %%
' ################################### params for finetuning #############################'

CHOSEN_EPOCHS = 15
BATCHSIZE = 10
chosen_learning_rate = 0.00001
chosen_loss = params["loss function"]
unfrozen_last_layers = 5
early_stopping = True
# update params dictionary to override old epochs and learning rate, and batch size
try:
    params["learning rate"] = chosen_learning_rate
    params["epochs"] = CHOSEN_EPOCHS
    params["batch size"] = BATCHSIZE
except:
    raise ValueError("learning rate and epochs parameter not found in extracted params of old run!" 
                     "Check spelling and old run params!")

# custom params for mlflow logging
tag = "finetuning"
custom_params = {"unfrozen last layers": unfrozen_last_layers}
mlflow_tracking = True

# %%
' ################################# load model artifact, unfreeze layers #################################'

# get model path
try:
    model_path = os.path.join("..", f"models/mlartifacts/{mlflow_experiment_ID}/{mlflow_run_ID}/artifacts/model_artifact/data/model.keras")
    model_path = os.path.abspath(model_path) 
except:
    raise NameError("model path not found. Check the arctifact path, i.e. the child path of the mlflow_run_ID in the try clause!!" 
                    "Possible the name can have changed")

# load model
model = load_model(model_path, compile=False, safe_mode=True)

# extract base_model
base_model = model.layers[1]

# unfreeze base model, but keep only last fex layers unfrozen (MobileNet gast 85 layers!)
base_model.trainable = True
for layer in base_model.layers[:-unfrozen_last_layers]:
    layer.trainable = False

for layer_number, layer in enumerate(model.layers):
    print(layer_number, layer.name, layer.trainable)

# %%
' ################################ compile again, create summary ##################################'

model.compile(loss=chosen_loss, 
              optimizer = keras.optimizers.Adam(learning_rate=chosen_learning_rate), 
              metrics=['binary_accuracy'])

# print model summary
model.summary()
# get model summary as string
model_summary_str = training_helpers.generate_model_summary_string(model)

# %%
' ######################################### getting training and validation data ################################'

# get the data
train_data, val_data = get_train_val_data(BATCHSIZE, IMGSIZE, channel_mode = "rgb")

# %%
' ########################################## callbacks #######################'
# define callbacks. create empty vessel
chosen_callbacks = []
# model checkpoint: create temp path for temp storage of best model
temp_dir = tempfile.TemporaryDirectory()
current_dir = os.getcwd()
checkpoint_path = os.path.join(current_dir, "temp_model.h5")

# define checkpoint callback
checkpoint = keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor="val_binary_accuracy",
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch"
)

chosen_callbacks.append(checkpoint)

# define early stopping callback
if early_stopping:
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.002,
        patience=10,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=0,
    )
    chosen_callbacks.append(early_stopping)

# %% 
' ################################### training ####################################'
# training
start_time = time.time()

history = model.fit(train_data,
          batch_size = BATCHSIZE, epochs = CHOSEN_EPOCHS,
          validation_data=val_data,
          callbacks = chosen_callbacks
          );

end_time = time.time()
training_time = end_time - start_time
print(f"train time:  {training_time:.2f} seconds = {training_time/60:.1f} minutes")

# delete temp path of model checkpoint
if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

# %%
' ########################################## prediction on validation and test set ########'

# get training data again (generators have been consumed during training and need to be reconstructed)
train_data, val_data = get_train_val_data(BATCHSIZE, IMGSIZE, channel_mode= "rgb")
val_loss, val_binary_accuracy = model.evaluate(val_data, verbose = 1)

# get test data
test_data = get_test_data(BATCHSIZE, IMGSIZE, channel_mode= "rgb")
test_loss, test_binary_accuracy = model.evaluate(test_data, verbose = 1)
print('Val loss:', val_loss)
print('Val binary accuracy:', val_binary_accuracy)
print('test loss:', test_loss)
print('test binary accuracy:', test_binary_accuracy)

# %%
'####################### generate plot of learning curves ################'
# create learning curve (for logging with MLFlow)
learning_curves = training_helpers.generate_plot_of_learning_curves(history)

# %%
' ########################### MLFlow model logging #######################'
# get batch for signature
batch = train_data.take(1)
# start logging the run
log_mlflow_run(model,
               run_name = mlflow_run_name, 
               epochs = CHOSEN_EPOCHS, 
               batch_size = BATCHSIZE, 
               loss_function = chosen_loss, 
               optimizer= params["optimizer"], 
               learning_rate = chosen_learning_rate, 
               top_dropout_rate =  params["dense layer dropout rate"], 
               model_summary_string = model_summary_str, 
               run_tag = tag, 
               signature_batch = batch, 
               val_accuracy = val_binary_accuracy, 
               test_accuracy = test_binary_accuracy, 
               custom_params = custom_params, 
               fig = learning_curves)