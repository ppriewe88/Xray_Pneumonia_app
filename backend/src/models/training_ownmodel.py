import time
import os
import sys

import numpy as np
from tensorflow import keras


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.helpers import get_train_val_data, get_test_data, IMGSIZE
import training_helpers
from mlflow_logging import log_mlflow_run



BATCH_SIZE = 128
EPOCHS = 5

loss_func = "binary_crossentropy"
learning_rate = 0.01
momentum = 0.8
optimizer = keras.optimizers.SGD(learning_rate = learning_rate, momentum = momentum)
dropout_rate = 0.3

mlflow_logging = True

train_data, val_data = get_train_val_data(BATCH_SIZE, IMGSIZE, channel_mode = "grayscale")

# compute class weights

def get_class_weights():

    all_labels = np.array([])
    
    for _, y in train_data:
        y_flat = y.numpy().flatten()
        all_labels = np.concatenate((all_labels,y_flat))

    label_counts = [np.sum(all_labels == i) for i in range(2)]

    class_weights = 0.5 * np.sum(label_counts) / label_counts
    class_weights = np.around(class_weights,2)

    class_weights_dict = dict(enumerate(class_weights))
    
    return class_weights_dict
class_weights_dict = get_class_weights()

# construct the neural net
def get_model(dropout_rate): 
    
    inputs = keras.layers.Input(shape=(IMGSIZE, IMGSIZE, 1))
    
    # Rescaling
    x = keras.layers.Rescaling(scale = 1./255)(inputs)
    
    # First block
    x = keras.layers.Conv2D(
        filters=8,
        kernel_size = (3,3),
        strides = (1,1),
        padding = 'same', # such that the output has the same size as the input
        activation = 'relu',
        kernel_regularizer = None
        )(x) # output shape = (IMGSIZE, IMGSIZE, 8)
    
    x = keras.layers.MaxPooling2D(pool_size=(4, 4))(x) # output shape = (IMGSIZE/4, IMGSIZE/4, 8) 
    
    # Second block 
    x = keras.layers.Conv2D(
        filters=16,
        kernel_size = (3,3),
        strides = (1,1),
        padding = 'same',
        activation = 'relu',
        kernel_regularizer = None
        )(x) # output shape = (IMGSIZE/4, IMGSIZE/4, 16)
    
    x = keras.layers.MaxPooling2D(pool_size=(4, 4))(x) # output shape = (IMGSIZE/16, IMGSIZE/16, 16) 
    
    # Third block 
    x = keras.layers.Conv2D(
        filters=32,
        kernel_size = (3,3),
        strides = (1,1),
        padding = 'same',
        activation = 'relu',
        kernel_regularizer = None
        )(x) # output shape = (IMGSIZE/16, IMGSIZE/16, 32)
    
    x = keras.layers.MaxPooling2D(pool_size=(4, 4))(x) # output shape = (IMGSIZE/64, IMGSIZE/64, 32) = (4,4,32)
    
    # Head (Flatten + Dense Layer)
    
    x = keras.layers.Flatten()(x) # output shape = 512
    x = keras.layers.Dense(10, activation="relu", kernel_regularizer = None)(x)
    
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate)(x)

    #Final Layer (Output)
    output = keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=output)
    
    return model
model = get_model(dropout_rate)

metrics = ["binary_accuracy"]

model.compile(
    loss=loss_func, 
    metrics = metrics,
    optimizer = optimizer
    )

# print model summary
model.summary()


start = time.time()

history = model.fit(
    train_data, 
    epochs = EPOCHS, 
    verbose = True,
    class_weight = class_weights_dict,
    validation_data = val_data,
    # callbacks = [lr_reduction, model_checkpoint_callback]
    )


print(f'Training time = {time.time() - start} sec')




if mlflow_logging: 
    
    ########## Get information needed for logging ##############

    # val & test accuracy
    val_accuracy =  history.history['val_binary_accuracy'][-1]
    test_data = get_test_data(BATCH_SIZE, IMGSIZE, channel_mode = "grayscale")
    _, test_accuracy = model.evaluate(test_data) ###### need to define test data and test the code line


    # get model summary as string
    model_summary_str = training_helpers.generate_model_summary_string(model)

    # create learning curve fig
    learning_curves = training_helpers.generate_plot_of_learning_curves(history)

    # take an input example out of the train data -> needed for mlflow.infer_signature()
    batch = train_data.take(1)
    
    # log custom parameters 
    custom_params_dict = {'momentum': momentum, 'class weights': class_weights_dict}

    
    # First step: run "mlflow server --host 127.0.0.1 --port 8080" in a different terminal to open the server; 
    # Make sure that both the python script and rhe mlflow server command are ran from the same folder !!!!!! 
    # When http://127.0.0.1:8080 displays nonsense, one can try to do a hard refresh while on the webpage with Crtl+Shift+R
    
    log_mlflow_run(
    model, # keras model to be logged
    run_name = 'CNN with small dense layer 5 epochs', # string that will be displayed as the run title in mlflow GUI
    epochs = EPOCHS,
	batch_size = BATCH_SIZE,
	loss_function = loss_func,
	optimizer = optimizer, # can be an optimizer object
	learning_rate = learning_rate,
	top_dropout_rate = dropout_rate,
	model_summary_string = model_summary_str, # string for model summary (comes from a helper function)
    run_tag = 'Small Self-built CNN with a dropout layer; to be used as a baseline model.', # string explaining what this run was for
    signature_batch = batch, # needed for infer_signature
    val_accuracy = val_accuracy,
    test_accuracy = test_accuracy,
	custom_params = custom_params_dict, # must be a dictionary (eg for momentum, activation functions in the top layer)
    fig = learning_curves # takes a figure object as input
    )
