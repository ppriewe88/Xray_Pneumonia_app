import io
from tensorflow import keras
import matplotlib.pyplot as plt

def generate_model_summary_string(model) -> str:
    """this function receives a keras model as input, 
    extracts model summary (object) and turns it into a string (suitable for logging).
    
    params: model -> keras model
    return: model_summary_str -> string"""

    if not isinstance(model, keras.Model):
        raise TypeError(f"Model is not a Keras model! Received type: {type(model)}")

    buffer = io.StringIO()
    model.summary(print_fn=lambda x: buffer.write(x + '\n'))
    model_summary_str = buffer.getvalue()

    return model_summary_str

def generate_plot_of_learning_curves(history_in):
    """This function receives a keras model's training history as input. 
    It returns a plot (matplotlib figure) of the learning curves. 
    It is strictly designed to return the binary accuracy, which has to be contained in the history input.
    
    params: history_in -> expected to be a model history
    return: fig -> matplotlib figure-object
    """

    if not isinstance(history_in, keras.callbacks.History):
        raise TypeError(
            f"Invalid history object. Expected keras.callbacks.History, "
            f"got {type(history_in).__name__}"
        )

    fig, ax = plt.subplots()
    ax.plot(history_in.history['binary_accuracy'], label='Train accuracy (binary)')
    ax.plot(history_in.history['val_binary_accuracy'], label='Validation accuracy (binary)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('binary accuracy')
    ax.legend()
    ax.set_title("Training and Validation binary accuracy")
    return fig