"""
This script contains the training routine for transfer learning with MobileNet and/or ResNet.
It also contains the setup for experiment tracking via mlflow.

If you run this script to train, and you want to log with mlflow, you have to start the tracking server of mlflow. 
To do so, in the directory of this script (i.e. folder transfer_learning), run the following command in terminal to start mlflow server for tracking experiments:

mlflow server --host 127.0.0.1 --port 8080

Then check the localhost port to access the MLFlow GUI for tracking!
Run this script to conduct training experiments (runs). If mlflow server is running, the experiment will be tracked as a run.
You will find the result in the mlflow tracking server ("backend-store"), i.e. models/mlartifacts/[...newly generated run id...]
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tempfile
import os
import time
import training_helpers
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.helpers import get_train_val_data, get_test_data, IMGSIZE
from mlflow_logging import log_mlflow_run


# %%
' ####################### params #######################'

# params for training
BATCHSIZE = 10
CHOSEN_EPOCHS = 20
dense_layer_top_neurons = 128
dense_layer_top_activation = "relu"
dropout_rate_top = 0.4
chosen_loss = "binary_crossentropy"
chosen_optimizer = "adam"
chosen_learning_rate = 0.001
early_stopping = True

# param for base model selection
selected_model = "MobileNet" # MobileNet or ResNet

# custom params for mlflow logging
mlflow_run_name = "optimized params"
custom_params = {"top dense layer activation": dense_layer_top_activation}
mlflow_tracking = True

# %%
' ######################################### getting training and validation data ################################'

train_data, val_data = get_train_val_data(BATCHSIZE, IMGSIZE, channel_mode="rgb")

 # %%
' ################################################## defining the model #########################'
# base model
if selected_model == "ResNet":
    tag = "ResNet152V2 with Dense top"
    # ResNet
    base_model = keras.applications.ResNet152V2(
        input_shape = (IMGSIZE,IMGSIZE, 3),
        include_top = False,
        weights = "imagenet",)
elif selected_model == "MobileNet":
    tag = "MobileNet"
    # MobileNet
    base_model = keras.applications.MobileNet(
        input_shape = (IMGSIZE,IMGSIZE, 3),
        include_top=False,
        weights="imagenet")

base_model.trainable = False

# complete model setup
inputs = tf.keras.layers.Input(shape = (IMGSIZE, IMGSIZE, 3))
x = keras.layers.Rescaling(scale = 1./255)(inputs)
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(dense_layer_top_neurons, activation=dense_layer_top_activation)(x)
x = layers.Dropout(dropout_rate_top)(x)
output = layers.Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=output)

# %%
' ######################################### compile and summary #######################################'
# compile model
model.compile(loss=chosen_loss, 
              optimizer = keras.optimizers.Adam(learning_rate=chosen_learning_rate), 
              metrics=['binary_accuracy'])

# print model summary
model.summary()

# get model summary as string
model_summary_str = training_helpers.generate_model_summary_string(model)

# %%
' ########################################## callbacks #######################'
# define callbacks. create empty vessel
chosen_callbacks = []
# model checkpoint: create temp path for temp storage of best model
current_dir = os.getcwd()
checkpoint_path = os.path.join(current_dir, "temp_model.keras")

# define checkpoint callback
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_path,
    monitor="val_binary_accuracy",
    mode = "max",
    save_best_only=True,
    save_weights_only=False
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
' ############################################ training #########################'

start_time = time.time()

history = model.fit(train_data,
          batch_size = BATCHSIZE, epochs = CHOSEN_EPOCHS,
          validation_data=val_data,
          callbacks = chosen_callbacks
          );

end_time = time.time()
training_time = end_time - start_time
print(f"train time:  {training_time:.2f} seconds = {training_time/60:.1f} minutes")

# Load the best model
model = keras.models.load_model(checkpoint_path)
# delete temp path of model checkpoint
if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        
# %%
' ########################################## prediction on validation and test set ########'
# get training data again (generators have been consumed during training and need to be reconstructed)
train_data, val_data = get_train_val_data(BATCHSIZE, IMGSIZE, channel_mode="rgb")
val_loss, val_binary_accuracy = model.evaluate(val_data, verbose = 1)

# get test data
test_data = get_test_data(BATCHSIZE, IMGSIZE, channel_mode="rgb")
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
               optimizer= chosen_optimizer, 
               learning_rate = chosen_learning_rate, 
               top_dropout_rate =  dropout_rate_top, 
               model_summary_string = model_summary_str, 
               run_tag = tag, 
               signature_batch = batch, 
               val_accuracy = val_binary_accuracy, 
               test_accuracy = test_binary_accuracy, 
               custom_params = custom_params, 
               fig = learning_curves)
