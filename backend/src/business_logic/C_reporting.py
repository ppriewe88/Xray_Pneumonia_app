import numpy as np
import os
from mlflow import MlflowClient
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./")))
from business_logic.D_modelswitch import moving_average_column


' ##############################################################################################'
' ######################### performance reporting and plotting functions #######################'

def get_performance_indicators_mlflow(num_steps_short_term):
    '''
    Function that fetches data from the mlflow client and 
    returns a dictionary summarizing the to-date performance 
    of the three pneumonia x-ray classification (aliased) models.
    
    Parameters
    ----------
    num_steps_short_term : positive int
        Size of the window used for calculating the sliding average
        of accuracy.  
        
    Returns
    -------
    performance_dictionary: dict 
        Dictionary with three keys corresponding to the three tracked
        ML classifiers. The corresponding values are also dictionaries
        with the following keys: total number of predictions, 
        average accuracy for the last {num_steps_short_term} predictions,
        pneumonia true positives, pneumonia true negatives, pneumonia 
        false positives, and pneumonia false negatives.
    
    '''

    # setting the uri 
    client = MlflowClient(tracking_uri="http://127.0.0.1:8080")
        
    # get the three experiments: performance + (baseline, challenger, champion)
    print("getting experiment list")
    experiments = list(client.search_experiments())[:3]
    
    # get experiment names and ids
    print("getting experiment names and ids")
    exp_names = [exp.name for exp in experiments]
    exp_ids = [exp.experiment_id for exp in experiments]
    
    # define an empty dictionary to hold the performance indicators for each experiment
    performance_dictionary = {}
    
    # for loop to calculate perfomance indicators for each experiment/model
    for exp_name, exp_id in zip(exp_names, exp_ids):
        print(f"{exp_name}: getting runs")
        # all runs in the experiment with exp_id, i.e. number of predictions made
        runs = list(client.search_runs(experiment_ids = exp_id))
        
        # run_ids
        run_ids = [run.info.run_id for run in runs]
        
        # extract lists of accuracies, timestamps, and correct prediction labels
        # within the given experiment (0 = no pneumonia, 1 = pneumonia)
        print(f"{exp_name}: starting extraction of accuracies")
        accuracies = [list(client.get_metric_history(run_id = run_id, key = 'accuracy'))[0].value for run_id in run_ids]
        print(f"{exp_name}: starting extraction of time stamps")
        timestamps = [list(client.get_metric_history(run_id = run_id, key = 'accuracy'))[0].timestamp for run_id in run_ids]
        print(f"{exp_name}: starting extraction of input_labels")
        y_true = [list(client.get_metric_history(run_id = run_id, key = 'y_true'))[0].value for run_id in run_ids]
        
        # 1st row is timestamps, 2nd is accuracies and so on
        values_array = np.array([timestamps, accuracies, y_true])
        # sorting according to the timestamps
        values_array = values_array[:, values_array[0].argsort()]
        # get rid of the timestamps row
        values_array = values_array[1:]
        # restrict array to latest {num_steps_short_term} runs
        values_array_short_term = values_array[:,-num_steps_short_term:]
        print(values_array_short_term.shape)

        print(f"{exp_name}: calc confusion matrix")
        # calculate confusion matrix elements
        true_positives = np.sum((values_array_short_term[0] == 1) & (values_array_short_term[1] == 1))
        true_negatives = np.sum((values_array_short_term[0] == 1) & (values_array_short_term[1] == 0))
        false_negatives = np.sum((values_array_short_term[0] == 0) & (values_array_short_term[1] == 1))
        false_positives = np.sum((values_array_short_term[0] == 0) & (values_array_short_term[1] == 0))
        
        # save the experiment information in a dictionary
        exp_dictionary ={
            'total number of predictions': str(len(accuracies)),
            f'average accuracy for the last {num_steps_short_term} predictions': str(np.mean(values_array_short_term[0])),
            'pneumonia true positives': str(true_positives),
            'pneumonia true negatives': str(true_negatives),
            'pneumonia false positives': str(false_positives), 
            'pneumonia false negatives': str(false_negatives),   
        }
        
        # update the dictionary containing the information from the other experiments
        performance_dictionary.update({exp_name: exp_dictionary})
          
    return performance_dictionary

def get_performance_indicators_csv(alias, last_n_predictions = 100):
    """
    Fetches logged performance data of model with given alias from corresponding csv-file.
    Fetches global accuracy, number of predictions, and floating average. 
    Additionally calculates confusion matrix of entire history.

    Parameters
    ----------
    alias : string
        Alias of mlflow registry model version.
    last_n_predictions: int
        Controls timeframe of confusion matrix. Only last n predictions will be used to calculate confusion matrix.
        
    Returns
    -------
    summary: dictionary
        Contains performance info of model runs. Dictionary values are strings.
    """     

    # get path of csv-files
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    tracking_path = os.path.join(project_folder, "models/performance_tracking")
    file_path = os.path.join(tracking_path, f'performance_data_{alias}.csv')

    if not os.path.exists(file_path):
        return "Error: CSV file not found."

    # read csv files
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    if not rows:
        return "Error: CSV file is empty."

    # get values of last prediction (cumulations, averages) to calculate consecutive values
    last_row = rows[-1]
    total_predictions = int(last_row['log_counter'])

    # initialize confusion matrix
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    # convert to numpy and get accuracies and true labels, restricted to last_n_predictions
    y_true = np.array([float(row['y_true']) for row in rows[-last_n_predictions:]])
    accuracy = np.array([float(row['accuracy']) for row in rows[-last_n_predictions:]])

    # calc confusion matrix
    true_positives = np.sum((y_true == 1) & (accuracy == 1))
    true_negatives = np.sum((y_true == 0) & (accuracy == 1))
    false_positives = np.sum((y_true == 0) & (accuracy == 0))
    false_negatives = np.sum((y_true == 1) & (accuracy == 0))

    # calc avg of last n predictions
    avg_last_n_predictions = np.mean(accuracy)

    # generate result dict
    summary = {
        f"performance csv {alias}": {
            "total number of predictions": str(total_predictions),
            f"average accuracy last {min(total_predictions, last_n_predictions)} predictions": f"{round(avg_last_n_predictions, 4)}",
            "pneumonia true positives": str(true_positives),
            "pneumonia true negatives": str(true_negatives),
            "pneumonia false positives": str(false_positives),
            "pneumonia false negatives": str(false_negatives)
        }
    }

    return summary

def generate_model_comparison_plot(window = 50, scaling =  "log_counter"):
    '''
    Function that generates a plot comparing the performance of
    models over time. The upper part of the plot shows the accuracy 
    of the champion and challenger models, averaged over a window
    whose lenght is specified by the {window} parameter. The lower
    part of the plot shows which of the trained ML models is the 
    challenger and the champion at a given time.
    
    Parameters
    ----------
    window : positive int
        Size of the window (= number of consecutive runs) used for 
        calculating the sliding average of the accuracy.
    scaling : "log_counter" or "timestamp"
        Controls what is shown on the x-axis of the plot. If
        scaling = "timestamp", then the x-axis shows the timestamps
        at which the api was used. Otherwise the x-axis shows the 
        run number (= number of times the api was used). 
        
    Returns
    -------
    fig: figure object 
        Figure comparing the performance of models.
    
    '''
    
    # get absolute path of the project dir
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # get path performance tracking subfolder
    tracking_path = os.path.join(project_folder ,"models/performance_tracking")

    # get file paths of model (alias) tracking
    path_champion = os.path.join(tracking_path ,"performance_data_champion.csv")
    path_challenger = os.path.join(tracking_path ,"performance_data_challenger.csv")
    path_baseline = os.path.join(tracking_path ,"performance_data_baseline.csv")

    # open files as dataframes
    df_champion = pd.read_csv(path_champion)
    df_challenger = pd.read_csv(path_challenger)
    df_baseline = pd.read_csv(path_baseline)

    # convert timestamp to datetime
    df_champion['timestamp'] = pd.to_datetime(df_champion['timestamp'])
    df_challenger['timestamp'] = pd.to_datetime(df_challenger['timestamp'])
    df_baseline['timestamp'] = pd.to_datetime(df_baseline['timestamp'])

    # get switching points from challenger csv aka. df_challenger dataframe. 
    # Result will be a pandas series containing the log_counters of the switches. 
    # The resetted index enumerates the switches.
    switch_points_log_counter = df_challenger[df_challenger["model_switch"]==True].reset_index(drop=True)["log_counter"]
    
    # define the figure and its subplots
    fig, axs = plt.subplots(2, 1, sharex=True, figsize = (16,8), height_ratios= [3,1])
    # remove horizontal space between axes
    fig.subplots_adjust(hspace=0)

    # generate plot lines
    moving_avg_challenger = moving_average_column(df_challenger["accuracy"], window = window)
    moving_avg_champion = moving_average_column(df_champion["accuracy"], window = window)
    axs[0].plot(df_champion[scaling], moving_avg_champion, label='Champion', color='blue', linestyle='-', linewidth=3)
    axs[0].plot(df_challenger[scaling], moving_avg_challenger, label='Challenger', color='orange', linestyle='--', linewidth=3)


    # vertical lines showing when the automated switches happened
    for log_counter in switch_points_log_counter:
        axs[0].axvline(
            x=log_counter,
            color='black',
            linestyle='-',
            linewidth=2,
            # generate label (legend) only for first element to avoid redundancy in legend
            label='automated model switch' if log_counter == switch_points_log_counter[0] else None
        )

    # set common axis labels and titles
    axs[0].set_ylabel(f"moving avg accuracy for the last {window} predictions", fontsize=12)
    axs[0].set_title(f'Model comparison over time', fontsize=20)

    # legend
    axs[0].legend(fontsize=15)


    # add grid
    axs[0].grid(True, linestyle='--', alpha=0.7)



    # organize all the models in a set
    models_champion = list(df_champion["model_tag"].unique())
    models_challenger = list(df_challenger["model_tag"].unique())
    models = set(models_champion + models_challenger)

    # dict for creating a new column in the df's
    # makes the y-axis ticks of lower plot easier to code
    model_mapping = {model: idx + 1 for idx, model in enumerate(models)}

    # generate plot lines
    for color, df in zip(["blue", "orange"], [df_champion, df_challenger]):
        df["plot"] = df["model_tag"].map(model_mapping)
        axs[1].plot(df["log_counter"], 
                    df["plot"], 
                    marker = "|",
                    markersize = 10, 
                    linestyle = '', 
                    color = color,
                    )

    # vertical lines showing when the automated switches happened
    for log_counter in switch_points_log_counter:
        axs[1].axvline(
            x=log_counter,
            color='black',
            linestyle='-',
            linewidth=2,
        )


    # set custom axis and title formatting according to scaling
    if scaling == "timestamp":
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        axs[1].xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        axs[1].set_xlabel("Time of run", fontsize=12)

    else:
        axs[1].xaxis.set_major_formatter(plt.ScalarFormatter())
        axs[1].xaxis.set_major_locator(plt.AutoLocator())
        axs[1].set_xlabel("Run number", fontsize=12)

    # set limits on y-axis
    axs[1].set_ylim(bottom=0.5, top=max(model_mapping.values()) + 0.5)
    # set ticks on y-axis
    axs[1].set_yticks(tuple(model_mapping.values()), labels = models)
    # make the ticks disappear from the y-axis
    axs[1].tick_params(axis='y', which='both', length=0)

    return fig    

def generate_confusion_matrix_plot(last_n_predictions = 10):
    '''
    Function that generates plot of confusion matrix of current champion.
    
    Parameters
    ----------
    last_n_predictions : positive int
        Timeframe (number of last predictions) used for confusion matrix.
        
    Returns
    -------
    fig: figure object 
        Figure of confusion matrix.
    
    '''
    # get data from csv performance report
    data = get_performance_indicators_csv(alias = "champion", last_n_predictions=last_n_predictions)

    # extract only nested dictionary of champion
    data_champion = data["performance csv champion"]

    # restructuring input for plot
    conf_matrix = np.array([
            [
                int(data_champion["pneumonia true positives"]), 
                int(data_champion["pneumonia false positives"])
                ],
            [
                int(data_champion["pneumonia false negatives"]), 
                int(data_champion["pneumonia true negatives"])
            ]
            ])
    
    # setting up plot
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Predicted Positive', 'Predicted Negative'],
        yticklabels=['Actual Positive', 'Actual Negative']
    )
    
    # get displayed predictions (get_performance_indicators_csv returns total number of available predictions)
    available_predictions = min(last_n_predictions, int(data_champion["total number of predictions"]))
    print(available_predictions)
    # now configure plot
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    accuracy = data_champion[f"average accuracy last {available_predictions} predictions"]
    main_title = f'Confusion Matrix (champion) last {available_predictions} predictions'
    plt.title(f"{main_title}\nAccuracy last {available_predictions} predictions: {accuracy}", pad=20)
    plt.tight_layout()
    
    return fig
