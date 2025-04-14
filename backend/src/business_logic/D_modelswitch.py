import numpy as np
import os
import csv

' ##############################################################################################'
' ######################## model comparison and takeover (switch) functions ####################'

def moving_average_column(column, window):
    """"
    For a given input column, the moving average of column values is calculated. 
    The window parameter (input) controls, how many consecutive values are uses for the moving average calculation.
    I.e.: 
    - For a column element's index, the last {window} predecessor values are used to calculate the moving average.
    - If there are not enough predecessors available (e.g. at low column indexes), only the avalílable values are taken.
    - This means: For low column indexes (lower than {window}), the window is shortened by definition!

    Parameters
    ----------
    column : array-like
        array-like integer containing numeric values
    window: int
        Controls how many predecessing values of a column element are used to calculate the moving average at that column element's index.
        
    Returns
    -------
    np.array: numpy array
        Numpy array containing moving averages for the input column.
    """
    column = np.array(column)
    averaged_col = [np.sum(column[max(0,i-window):i])/min(i, window) for i in range(1,len(column)+1)]
    
    return np.array(averaged_col)

def check_challenger_takeover(last_n_predictions = 20, window=50):
    """"
    Fetches data from performance logs (csv-files) of challenger and champion registry model versions. 
    Checks if the challenger's moving average accuracy has been better than the champion's moving average accuracy during last_n_predictions.
    Check is done by using accuracy column from csv-files. Moving average calculation is done with {window}.
    
    Parameters
    ----------
    last_n_predictions : integer
        Input for takeover condition (model switch). 
        Challenger has to have better moving average accuracy than champion during {last_n_predictions} to take over. 
    window: int
        Window parameter for moving average calculation. Will be passed to helper function moving_average_column.
        
    Returns
    -------
    check_if_chall_is_better: boolean
        True if challenger satisfies takeover condition (model switch).
    """

    # get relevant paths
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    tracking_path = os.path.join(project_folder, "models/performance_tracking")

    file_path_champ = os.path.join(tracking_path, f'performance_data_champion.csv')
    file_path_chall = os.path.join(tracking_path, f'performance_data_challenger.csv')

    # read csv file champion
    with open(file_path_champ, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        
    # Breaking condition nr. 1: check if there are at least {last_n_predictions + window} runs. If so, break
    if len(rows) < last_n_predictions + window:
        print(f"Initial protection phase (less than {last_n_predictions + window} runs available). No switch allowed yet.")
        return False
    
    
    # # Breaking condition nr. 2: check if switch was done in the previous {last_n_predictions} runs. If so, break
    # organize the last {last_n_predictions} model tag in a list, get unique tags (set)
    last_model_tags_unique = set([row['model_tag'] for row in rows[-(last_n_predictions+window+30):]])
    # check is switch was performed, i.e. more than one model tags in history
    switch_done = len(last_model_tags_unique) > 1
    # quit the function if the switch was done in the last {last_n_predictions + window} runs
    if switch_done:
        print(f"A switch happend during the last {last_n_predictions+window} runs. No switch allowed yet.")
        return False
        
    # If continuing here, start model comparison.
    # get last last_n_predictions, extract accuracy as integers
    last_rows_champ = rows[-(last_n_predictions + window):]
    last_acc_values_champ = [int(row['accuracy']) for row in last_rows_champ]
    # get moving average. Careful: correct calculation by extended window and capping! 
    # moving_average_column cuts window at the lower end of the column, thus the lower end has to be extended!
    moving_averages_champ = moving_average_column(last_acc_values_champ, window)[-last_n_predictions:]

    # read csv file challenger
    with open(file_path_chall, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    # get last last_n_predictions, extract accuracy as integers
    last_rows_chall = rows[-(last_n_predictions + window):]
    last_acc_values_chall = [int(row['accuracy']) for row in last_rows_chall]
    # get moving average. Careful: correct calculation by extended window and capping!
    # moving_average_column cuts window at the lower end of the column, thus the lower end has to be extended!
    moving_averages_chall = moving_average_column(last_acc_values_chall, window)[-last_n_predictions:]
    
    # compare by calculating difference
    diff = moving_averages_champ - moving_averages_chall
    
    # check if all entries negative
    check_if_chall_is_better = np.all(diff <= 0)
    print(f"Performance comparison between challenger and champion has been made. Challenger's moving average better during last {last_n_predictions} runs: ", check_if_chall_is_better)
    
    return check_if_chall_is_better

def switch_champion_and_challenger():
    """"
    Swaps mlflow registry model versions that are associated with champion and challenger model aliases.
    Swap is achieved by swapping content of alias files. 
    After swapping, function updates the "model_switch" column of the last run in the champion's and challenger's csv files (new column value = "True") 
    
    Parameters
    ----------
    No parameters
        
    Returns
    -------
    No returns
    """

    # get paths of alias files
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    unif_exp_path = os.path.join(project_folder, r"models")
    path_challenger_alias = os.path.join(unif_exp_path, r"mlruns/models/Xray_classifier/aliases/challenger")
    path_champion_alias = os.path.join(unif_exp_path, r"mlruns/models/Xray_classifier/aliases/champion")
    path_challenger_csv = os.path.join(unif_exp_path, r"performance_tracking/performance_data_challenger.csv")
    path_champion_csv = os.path.join(unif_exp_path, r"performance_tracking/performance_data_champion.csv")

    # read alias files 
    with open(path_champion_alias, 'r') as file:
        version_number_champion = file.read()
    with open(path_challenger_alias, 'r') as file:
        version_number_challenger = file.read()

    # swap content (i.e. version numbers)
    with open(path_champion_alias, 'w') as file:
        file.write(version_number_challenger)
    with open(path_challenger_alias, 'w') as file:
        file.write(version_number_champion)
        
    # update challenger csv-files of predictions: mark model_switch
    with open(path_challenger_csv, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows_chall = list(reader)
        rows_chall[-1]["model_switch"]="True"
    with open(path_challenger_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows_chall[0].keys())
        writer.writeheader()
        writer.writerows(rows_chall)
    # update champion csv-files of predictions: mark model switch
    with open(path_champion_csv, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows_champ = list(reader)
        rows_champ[-1]["model_switch"]="True"
    with open(path_champion_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows_champ[0].keys())
        writer.writeheader()
        writer.writerows(rows_champ)
    print("challenger and champion have been switched")

