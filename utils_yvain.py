from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

import pandas
import pandas as pd
from shapely import Point
from sklearn.metrics import r2_score
import matplotlib.pyplot as plot
from math import hypot
import csv

#todo : check if we need to import this because problematic
from errors import EndOfTrialNotInTarget
from myenum import TargetPosition


# criteria definition 
# beginning/end of movement
min_time_for_movement_start = 0.2
min_time_for_movement_stop = 0.15
movement_speed_threshold = 200

min_target_time = 0.01
#todo adapt to import target radius from config file
#todo check if coherent with the task
target_radius = 40
trial_feedback = True
#todo import feedback from trial-by-trial config (maybe in the main?)
#todo import screen boundaries for lost status
screen_limits = {"left": 0, "right": 1680, "top": 0, "bottom": 1050}
#toutes les vitesses sont en pixel/sec

#todo add wrong direction : create a function to say if the direction is correct or not
#first vx > 1000? in the right direction( vx pos for right target) or x > half of distance. only after trigger

#todo add 

def add_time_and_speed_to_df(df: pd.DataFrame, time_step: float = 0.01) -> pd.DataFrame:
    """
    Cette fonction ajoute une colonne nommée `t` au dataframe `df`, ainsi que les colonnes `vx` et `vy` représentant les vitesses en x et en y de l'objet.
    La colonne `t` représente le temps correspondant à chaque ligne.
    La première ligne a une valeur de t égale à 0.
    Les autres lignes ont une valeur de t égale à la valeur de t de la ligne précédente plus `time_step`.
    Les vitesses à la première ligne sont toujours nulle.

    Parameters:
      df (pd.DataFrame): Un dataframe pandas avec les colonnes `x` et `y`.
      time_step (float, optional): La durée entre chaque instant représenté par une ligne du dataframe. La valeur par défaut est 0.01.

    Returns:
      pd.DataFrame: Le dataframe `df` avec les colonnes supplémentaires `t`, `vx` et `vy`.
    """
    df['vx'] = df['X'].diff() / time_step
    df['vy'] = df['Y'].diff() / time_step
    df.loc[0, 'vx'] = 0
    df.loc[0, 'vy'] = 0
    df['t'] = df.index * time_step
    df.loc[0, 't'] = 0
    return df


def add_movement_started_column(df: pd.DataFrame, movement_speed_threshold: float, min_time_for_movement_start: float, min_time_for_movement_stop: float, time_step) -> pd.DataFrame:
    """
    Adds a 'movement' column indicating whether movement is occurring.
    Movement starts when |Vy| >= threshold for a minimum duration.
    Movement stops when both |Vx| and |Vy| are below threshold for a minimum duration.

    Movement can start and stop multiple times.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'vx' and 'vy' columns.
        movement_speed_threshold (float): Threshold to detect movement.
        min_time_for_movement_start (float): Duration above threshold to start movement.
        min_time_for_movement_stop (float): Duration below threshold to stop movement.
        time_step (float): Time step between consecutive rows.

    Returns:
        pd.DataFrame: Modified DataFrame with 'movement' column.
    """
    df["movement"] = False
    movement_started = False

    start_window = int(min_time_for_movement_start / time_step)
    stop_window = int(min_time_for_movement_stop / time_step)

    i = 0
    while i < len(df):
        if not movement_started:
            if i + start_window < len(df):
                # Check for sustained movement in Vy
                if (df["vy"].iloc[i:i + start_window].abs() >= movement_speed_threshold).all():
                    df.loc[i:i + start_window, "movement"] = True
                    movement_started = True
                    i += start_window  # skip ahead to avoid double detection
                    continue
        else:
            if i + stop_window < len(df):
                # Check for sustained stop in both Vx and Vy
                vx_ok = (df["vx"].iloc[i:i + stop_window].abs() < movement_speed_threshold).all()
                vy_ok = (df["vy"].iloc[i:i + stop_window].abs() < movement_speed_threshold).all()
                if vx_ok and vy_ok:
                    df.loc[i:i + stop_window, "movement"] = False
                    movement_started = False
                    i += stop_window  # skip ahead
                    continue
            df.loc[i, "movement"] = True  # still moving

        i += 1

    return df


def add_trigger_crossed_column(df: pd.DataFrame, trigger: int) -> pd.DataFrame:
    """
        Add a boolean column 't_crossed' that indicates if the trigger has been crossed yet
    """
    df.loc[:, "t_crossed"] = False
    lines_below_trigger = df.loc[df["Y"] < trigger]
    if lines_below_trigger.empty:
        print("Debug: No rows found where Y < trigger.")
        raise Exception("trigger not crossed")
    line_crossing_trigger = lines_below_trigger.index[0]
    df.loc[line_crossing_trigger:, "t_crossed"] = True
    return df

#problem, speed in pixel/row? todo
def end_of_movement(df: pd.DataFrame, movement_speed_threshold: float = 2, min_rows_for_stop: int = 15) -> float:
    """
    Determines the time of movement stop based on velocity thresholds.
    Movement stop is defined as both vx and vy being below the velocity threshold for a minimum number of rows.
    This condition is only checked within the last second of the trial.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        movement_speed_threshold (float): The maximum velocity to consider the cursor as stopped (default: 2 pixels/row).
        min_rows_for_stop (int): The minimum number of consecutive rows to satisfy the stop condition (default: 15).

    Returns:
        float: The time of movement stop. If no stop is detected, returns None.
    """
    # Filter the DataFrame to include only the last second of the trial
    trial_end_time = df["t"].iloc[-1]
    valid_df = df[df["t"] >= trial_end_time - 1]

    # Iterate through the filtered DataFrame to find the movement stop
    for i in range(len(valid_df) - min_rows_for_stop + 1):
        sub_df = valid_df.iloc[i:i + min_rows_for_stop]
        if (sub_df["vx"].abs() < movement_speed_threshold).all() and (sub_df["vy"].abs() < movement_speed_threshold).all():
            return sub_df["t"].iloc[0]  # Return the time when the stop condition starts

    # If no movement stop is detected, return None
    return None

####################################################
#lost_status
####################################################

# todo check if good, but need trial_feedback
def check_lost_status(df: pd.DataFrame) -> tuple[bool, float | None]:
    """
    Checks if the cursor touches the screen limit or stays on the limits for too long,
    but only after the trigger is crossed.

    Returns:
        tuple: (lost_status: bool, lost_time: float or None)
    """
    # Filter the DataFrame to include only rows after the trigger is crossed
    if "t_crossed" not in df.columns or not df["t_crossed"].any():
        return False, None  # Trigger not crossed, no lost status

    df_after_trigger = df[df["t_crossed"]]

    limit_crossings = ((df_after_trigger["X"] <= screen_limits["left"]) | 
                       (df_after_trigger["X"] >= screen_limits["right"]) | 
                       (df_after_trigger["Y"] <= screen_limits["top"]) | 
                       (df_after_trigger["Y"] >= screen_limits["bottom"])).astype(int).diff().fillna(0).abs()

    total_crossings = limit_crossings.sum()

    time_on_limits = df_after_trigger[((df_after_trigger["X"] <= screen_limits["left"]) | 
                                       (df_after_trigger["X"] >= screen_limits["right"]) | 
                                       (df_after_trigger["Y"] <= screen_limits["top"]) | 
                                       (df_after_trigger["Y"] >= screen_limits["bottom"]))]

    time_on_limits_duration = time_on_limits["t"].diff().fillna(0).sum()

    lost_status = total_crossings >= 1 or time_on_limits_duration > 3

    # Estimate the moment of "loss"
    lost_time = None
    if total_crossings >= 1:
        lost_index = limit_crossings.cumsum()[limit_crossings.cumsum() >= 1].index[0]
        lost_time = df_after_trigger.loc[lost_index, "t"]
    elif time_on_limits_duration > 3:
        time_sum = 0
        for i in time_on_limits.index:
            if i == time_on_limits.index[0]:
                continue
            time_sum += time_on_limits.loc[i, "t"] - time_on_limits.loc[i - 1, "t"]
            if time_sum > 3:
                lost_time = time_on_limits.loc[i, "t"]
                break

    return lost_status, lost_time


def get_trial_status(df: pd.DataFrame, feedback: bool, lost_status: bool, target_radius: float, min_target_time: float = 0.01, time_step: float = 0.01) -> str:
    """
    Determines the trial status based on feedback, target entry, and movement stop.
    """
    if feedback:
        t_stop = end_of_movement(df)
        if t_stop is None:
            t_stop = df["t"].iloc[-1]
        last_row = df[df["t"] == t_stop].iloc[0]
        return "success" if last_row["in_target"] else "fail"
    else:
        target_enters = get_target_enters(df, min_target_time, time_step)
        lost_status, lost_time = check_lost_status(df)
        if not lost_status:   
            return "success" if target_enters else "fail"
        else:
            if target_enters and lost_time is not None and target_enters[0] < lost_time:
                return "success"
            return "lost"



######################################
#Reaction times
######################################
#todo check, it seems shit

"""
faire un RT qui ne peux juste pas etre inférieur à 0.2s?
"""

def get_RT(df: pd.DataFrame, t_trigger: float, min_rest: float = 0.15) -> float | str:
        """
        Determines the reaction time (RT) based on the start of movement.
        The RT is valid only if there is a minimum rest period (no movement) of at least `min_rest` seconds before movement starts.
        Only considers movements occurring before the trigger time.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data.
            t_trigger (float): The trigger time.
            min_rest (float): Minimum rest duration to consider a pause before movement.

        Returns:
            float or str: The reaction time, or an error message if no valid reaction time is found.
        """
        # Filter the DataFrame to include only rows before the trigger time
        df = df[df["t"] < t_trigger]

        # Identify rows where movement is detected
        movement_rows = df[df["movement"] == True]
        if movement_rows.empty:
            return "No movement detected before trigger"

        # Iterate through movement rows to find the first valid movement after a rest period
        for i in range(len(movement_rows)):
            movement_start = movement_rows["t"].iloc[i]

            # Check if there is sufficient rest before the movement starts
            rest_df = df[(df["t"] < movement_start) & (df["t"] >= movement_start - min_rest)]
            if rest_df.empty or not rest_df["movement"].any():
                return movement_start

        return "No valid movement after sufficient rest before trigger"

#todo change to <0.2

def get_t_trigger(df):
    "return the value of the line crossing the trigger"
    ser = df[df["t_crossed"] == True]['t'].head(1)
    return float(ser.iloc[0])  


#todo make if not float, return time from first movement to trigger
def get_RtTrig(t_trigger, RT) -> float:
    # RtTrig (movement start to trigger)
    if not isinstance(RT, float):
        # If RT is not a float, compute time between first movement and t_trigger
        return round(t_trigger - RT["t"].iloc[0], 3)
    return round(t_trigger - RT, 3)


#pourquoi ne pas juste prendre le return de la fonction d'au dessus?
def get_TrigT(RT: float, RtTrig: float):
    # Temps de réaction 1= Temps à l'arrêt + Temps D-T
    return RT + RtTrig


def get_trig_distance(df: pd.DataFrame, t_trigger: float) -> float:
    """
    Computes the total distance traveled by the cursor from the beginning of the trial to the trigger.
    Returns:
        float: The total distance traveled by the cursor from the beginning of the trial to the trigger.
    """
    # Filter the DataFrame to include only rows before the trigger time
    df = df[df["t"] <= t_trigger]

    # Compute the differences between consecutive coordinates
    diffs = df.diff()

    # Compute the Euclidean distance between each pair of consecutive coordinates
    distances = np.sqrt(diffs['X'] ** 2 + diffs['Y'] ** 2)

    # Compute the sum of distances
    trigger_distance = distances.sum()

    return trigger_distance


##################################################################
#from trigger to target#
##################################################################

#need to have target size

###new target definition as circle
#on pourrait pas combiner les 2 pour pas avoir une seule fonction (pas besoin de dire TRUE ou FALSE?)
#pb target_center[0]? 
def add_in_circle_column(df: pd.DataFrame, target_center: tuple, target_radius: float, column_name: str):
    """
    Adds a boolean column to the DataFrame indicating whether each point is inside the circle.
    Parameters:
        df (pd.DataFrame): The DataFrame containing the points.
        target_center (tuple): The center of the circle as (x, y).
        target_radius (float): The radius of the circle.
        column_name (str): The name of the new column.
    """
    df[column_name] = df.apply(
        lambda row: hypot(row["X"] - target_center[0], row["Y"] - target_center[1]) <= target_radius,
        axis=1
    )
    return df



def add_in_target_column(df: pd.DataFrame, target_center: Point, target_radius: float) -> pd.DataFrame:
    """
    Adds a column indicating whether each point is inside the target circle.
    Parameters:
        df (pd.DataFrame): The DataFrame containing the points.
        target_center (Point): The center of the target circle.
        target_radius (float): The radius of the target circle.
    """
    return add_in_circle_column(df=df, target_center=target_center, target_radius=target_radius, column_name="in_target")




# faire avec un critère de déscélération pour avoir un "real" temps d'entrée dans la cible?
def get_target_enters(df: pd.DataFrame, min_target_time: float, time_step: float = 0.01):
    """
    Returns a list of times t where the subject enters the target and stays in the target for at least `min_target_time` seconds. 
    min_target_time is equal to 0.01s now (1 row).

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        min_target_time (float): The minimum time (in seconds) the subject must stay in the target.
        time_step (float): The time interval between consecutive rows in the DataFrame.

    Returns:
        list: A list of times of valid target entries.
    """
    required_rows = int(min_target_time / time_step)
    target_enters = []
    in_target = False

    for i in range(len(df)):
        # Ensure the index range is valid
        if i + required_rows - 1 >= len(df):
            break

        if df.iloc[i]["in_target"]:
            if not in_target:  # New entry detected
                # Check if the subject stays in the target for the required duration
                if df.iloc[i:i + required_rows]["in_target"].all():
                    target_enters.append(float(df.iloc[i]["t"]))
                    in_target = True
        else:
            in_target = False  # Reset when leaving the target

    return target_enters


#todo check if good
def get_trigger_to_first_target_time(df: pd.DataFrame, t_trigger: float, feedback: bool, lost_status: bool, target_radius: float, min_target_time: float = 0.01, time_step: float = 0.01) -> float | str:
    """
    Computes the time between the trigger and the first entry into the target.
    Handles different cases based on feedback and lost status.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        t_trigger (float): The time when the trigger is crossed.
        feedback (bool): Whether feedback is enabled.
        lost_status (bool): Whether the trial is marked as "lost_status".
        target_radius (float): The radius of the target.
        min_target_time (float, optional): Minimum time in the target to consider entry valid. Defaults to 0.01.
        time_step (float, optional): Time interval between consecutive rows in the DataFrame. Defaults to 0.01.

    Returns:
        float or str: The computed time or a status string.
    """
    target_enters = get_target_enters(df, min_target_time, time_step)

    if not target_enters or t_trigger is None:
        return "no target enter"

    lost_status, lost_time = check_lost_status(df)

    if feedback or not lost_status:
        return target_enters[0] - t_trigger

    if lost_time and target_enters[0] < lost_time:
        return target_enters[0] - t_trigger

    return "no target enter"
#add edga case if movement stops?


def get_trigger_to_first_target_distance(df: pd.DataFrame, t_trigger: float, feedback: bool, lost_status: bool, time_step: float = 0.01) -> float | str:
    """
    Computes the distance traveled by the cursor from the trigger to the first entry into the target.
    Handles different cases based on feedback and lost status.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        t_trigger (float): The time when the trigger is crossed.
        feedback (bool): Whether feedback is enabled.
        lost_status (bool): Whether the trial is marked as "lost_status".
        time_step (float, optional): The time interval between consecutive rows in the DataFrame. Defaults to 0.01.

    Returns:
        float or str: The computed distance or a status string.
    """
    target_enters = get_target_enters(df, min_target_time, time_step)

    if not target_enters or t_trigger is None:
        return "no target enter"

    lost_status, lost_time = check_lost_status(df)

    if feedback or not lost_status:
        t_first_target_enter = target_enters[0]
        df = df[(df["t"] >= t_trigger) & (df["t"] <= t_first_target_enter)]
        return compute_distance(df)

    if lost_time and target_enters[0] < lost_time:
        t_first_target_enter = target_enters[0]
        df = df[(df["t"] >= t_trigger) & (df["t"] <= t_first_target_enter)]
        return compute_distance(df)

    return "no target enter"


def compute_distance(df: pd.DataFrame) -> float:
    """
    Computes the total Euclidean distance traveled based on the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        float: The total distance traveled.
    """
    diffs = df.diff()
    distances = np.sqrt(diffs['X'] ** 2 + diffs['Y'] ** 2)
    return distances.sum()


################################################################################
#from target to stop#
################################################################################


def get_target_to_stop_time(df: pd.DataFrame, t_first_target_enter: float, movement_speed_threshold: float, min_time_for_movement: float, time_step: float, feedback: bool) -> float:
    """
    Computes the time between the first target entry and the stop of the cursor, only if feedback is enabled.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        t_first_target_enter (float): The time of the first entry into the target.
        movement_speed_threshold (float): The maximum velocity to consider the cursor as stopped.
        min_time_for_movement (float): The minimum duration for which the velocity must remain below movement_speed_threshold to consider the movement stopped.
        time_step (float): The time interval between consecutive rows in the DataFrame.
        feedback (bool): Whether feedback is enabled.

    Returns:
        float or str: The time between the first target entry and the stop of the cursor, or a status string.
    """
    if not feedback:
        return "feedback disabled"

    if not t_first_target_enter:
        return "no target enter"

    # Filter the DataFrame to start from the first target entry
    df = df[df["t"] >= t_first_target_enter]

    # Use the end_of_movement function to determine movement stop
    t_stop = end_of_movement(df, movement_speed_threshold=movement_speed_threshold, min_rows_for_stop=int(min_time_for_movement / time_step))

    if t_stop is not None:
        return t_stop - t_first_target_enter

    # If the cursor never stops, return the time until the end of the trial
    t_end_trial = df["t"].iloc[-1]
    return t_end_trial - t_first_target_enter


def get_target_to_stop_distance(df: pd.DataFrame, t_first_target_enter: float, movement_speed_threshold: float, min_time_for_movement: float, time_step: float, target_center: Point) -> float:
    """
    Computes the traveled distance from the first target entry to the movement stop.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        t_first_target_enter (float): The time of the first entry into the target.
        movement_speed_threshold (float): The maximum velocity to consider the cursor as stopped.
        min_time_for_movement (float): The minimum duration for which the velocity must remain below movement_speed_threshold to consider the movement stopped.
        time_step (float): The time interval between consecutive rows in the DataFrame.
        target_center (Point): The center of the target circle.

    Returns:
        float: The traveled distance from the first target entry to movement stop.
    """
    # Filter the DataFrame to start from the first target entry
    df = df[df["t"] >= t_first_target_enter]

    # Calculate the number of consecutive rows required to satisfy min_time_for_movement
    required_rows = int(min_time_for_movement / time_step)

    # Iterate through the DataFrame to check for movement stop
    for i in range(len(df) - required_rows + 1):
        sub_df = df.iloc[i:i + required_rows]
        if (sub_df["vx"].abs() <= movement_speed_threshold).all() and (sub_df["vy"].abs() <= movement_speed_threshold).all():
            # Filter the DataFrame up to the stop time
            df = df[df["t"] <= sub_df["t"].iloc[0]]
            break
    # Compute the differences between consecutive coordinates
    diffs = df.diff()
    # Compute the Euclidean distance between each pair of consecutive coordinates
    distances = np.sqrt(diffs['X'] ** 2 + diffs['Y'] ** 2)
    # Compute the sum of distances
    return distances.sum()




##########################################################################
#total movement#
##########################################################################



def get_total_movement_time(df: pd.DataFrame, RT: float) -> float:
    """
    Computes the total duration of the movement from movement onset to movement stop.
    Movement stop is determined using the end_of_movement function.
    """
    movement_start = RT if isinstance(RT, float) else df[df["movement"] == True]["t"].iloc[0]

    # Determine the movement stop using the end_of_movement function
    movement_stop = end_of_movement(df)

    # If no movement stop is detected, use the trial end time
    if movement_stop is None:
        movement_stop = df["t"].iloc[-1]

    return movement_stop - movement_start

def get_total_movement_distance(df: pd.DataFrame, RT: float) -> float:
    """
    Computes the total Euclidean distance traveled by the cursor during the movement.
    Movement is considered from the reaction time (RT) until the movement stops or the trial ends.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        RT (float): The reaction time indicating the start of the movement.

    Returns:
        float: The total distance traveled during the movement.
    """
    # Filter the DataFrame to include only rows from the reaction time onward
    df = df[df["t"] >= RT]

    # Determine the time of movement stop using the end_of_movement function
    t_stop = end_of_movement(df)

    # If movement stop is detected, filter the DataFrame up to the stop time
    if t_stop is not None:
        df = df[df["t"] <= t_stop]

    # Compute the differences between consecutive coordinates
    diffs = df.diff()

    # Compute the Euclidean distance between each pair of consecutive coordinates
    distances = np.sqrt(diffs['X'] ** 2 + diffs['Y'] ** 2)

    # Compute the sum of distances
    return distances.sum()




def get_total_trial_distance(df: pd.DataFrame) -> float:
    """
    Computes the total Euclidean distance traveled by the cursor until movement stops or until the end of the trial if no stop is detected.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        float: The total distance traveled.
    """
    # Determine the time of movement stop using the end_of_movement function
    t_stop = end_of_movement(df)

    # If movement stop is detected, filter the DataFrame up to the stop time
    if t_stop is not None:
        df = df[df["t"] <= t_stop]

    # Compute the differences between consecutive coordinates
    diffs = df.diff()

    # Compute the Euclidean distance between each pair of consecutive coordinates
    distances = np.sqrt(diffs['X'] ** 2 + diffs['Y'] ** 2)

    # Compute the sum of distances
    return distances.sum()



def get_total_trial_time(df: pd.DataFrame) -> float:
    """
    Computes the total duration of the trial.
    """
    if "t" not in df.columns:
        raise ValueError("The DataFrame must contain a 't' column representing time.")
    total_time = df["t"].iloc[-1]
    return total_time



def compute_final_distance(x, y, target_center):
    return hypot(x - target_center[0], y - target_center[1])


def get_cursor_final_distance(df, movement_speed_threshold: float, trial_data, target_center: Point):
    """
    Computes the final distance of the cursor from the target center.
    If the cursor stops moving (velocity below movement_speed_threshold), the distance is computed at the end of movement.
    If the cursor never stops moving, the distance is computed at the end of the trial.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        movement_speed_threshold (float): The maximum velocity to consider the cursor as stopped.
        trial_data (dict): The trial data.
        target_center (Point): The center of the target circle.

    Returns:
        tuple: The final distance and the time it was computed.
    """
    # Determine the time of movement stop using the end_of_movement function
    t_stop = end_of_movement(df, movement_speed_threshold=movement_speed_threshold)

    if t_stop is not None:
        # Compute distance at the end of movement
        stop_row = df[df["t"] == t_stop].iloc[0]
        return compute_final_distance(stop_row["X"], stop_row["Y"], target_center), t_stop

    # If no end of movement is detected, compute distance at the end of the trial
    last_row = df.iloc[-1]
    return compute_final_distance(last_row["X"], last_row["Y"], target_center), last_row["t"]


##################################
#RT movement correction
##################################

def get_TtA(t_trigger: float, t_max_vx):
    # TtA: Temps de réaction 2 = Temps entre le passage du trigger et l 'extremum de vitesse en X (pixel/s)
    if t_max_vx is None or t_trigger is None:
        return "Invalid input: t_max_vx or t_trigger is None"
    return t_max_vx - t_trigger


#todo : demander à ce que Vmax soit entre Ttrig et TTrig+1s
def get_t_max_vx(df: pd.DataFrame, target_position: TargetPosition, t_trigger: float, t_trigger_buffer: float):
    my_df = df[df["t"] >= t_trigger + t_trigger_buffer]
    if target_position == TargetPosition.C:
        return "centre", "centre"

    if target_position == TargetPosition.D:
        max_vx = my_df[my_df["vx"] == my_df["vx"].max()].head(1)
        max_vx_vx = float(max_vx["vx"].iloc[0])
        return max_vx_vx, float(max_vx["t"].iloc[0])

    if target_position == TargetPosition.G:
        min_vx = my_df[my_df["vx"] == my_df["vx"].min()].head(1)
        min_vx_vx = float(min_vx["vx"].iloc[0])
        return min_vx_vx, float(min_vx["t"].iloc[0])



#######################################
#initial direction of movement
#######################################

def get_initial_direction(df: pd.DataFrame, RT: float, time_window: float = 0.1) -> float:
    """
    Computes the initial direction of movement based on the velocity vector within a time window after the trigger.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        t_trigger (float): The time when the trigger is crossed.
        time_window (float, optional): The time window after the trigger to compute the direction. Defaults to 0.1.

    Returns:
        float: The angle of the initial movement direction in degrees.
    """
    # Filter the DataFrame to include rows within the time window after the trigger
    df_window = df[(df["t"] >= RT) & (df["t"] <= RT + time_window)]

    # Compute the average velocity in x and y directions
    avg_vx = df_window["vx"].mean()
    avg_vy = df_window["vy"].mean()

    # Compute the angle of the velocity vector in degrees (0 degree on top, 90 degrees to the right)
    angle = np.degrees(np.arctan2(-avg_vx, -avg_vy))
    return angle


def get_target_center(trial_data, trial_number):
    target_position = trial_data["target_positions"][trial_number]
    if target_position == "gauche":
        return trial_data["centre_cible_gauche"]
    elif target_position == "centre":
        return trial_data["centre_cible_centre"]
    elif target_position == "droite":
        return trial_data["centre_cible_droite"]
    else:
        raise Exception("Unknown target position")




def compute_trial(result_file: Path, trial_number: int, trial_data: dict, trigger: int, df=None, time_step=0.01,
                  min_target_time=0.01, movement_speed_threshold=200, period_min=0.01):
    """
    Computes trial variables and writes them to a CSV file.

    Parameters:
        result_file (Path): Path to the result file.
        trial_number (int): The trial number.
        trial_data (dict): Trial-specific data.
        trigger (int): Trigger value.
        df (pd.DataFrame): DataFrame containing trial data.
        time_step (float): Time step between rows in the DataFrame.
        min_target_time (float): Minimum time in the target to consider entry valid.
        movement_speed_threshold (float): Minimum velocity threshold for movement detection.
        period_min (float): Minimum duration for movement detection.
        max_final_speed (float): Maximum velocity to consider the cursor as stopped.
    """
    try:
        print(f"Starting computation for trial {trial_number}")
        # Initialize variables
        RT = t_trigger = RtTrig = t_trigger_computed = None
        t_first_target_enter = None
        trigger_to_target_time = None
        trigger_to_target_distance = None
        t_max_vx = None
        TtA = None
        initial_direction_angle = None
        target_enters = None
        target_to_stop_time = None 
        target_to_stop_distance = None
        total_movement_time = total_movement_distance = total_trial_time = total_distance_travelled = None
        finale_distance_to_center = None
        finale_distance_to_center_time = None

        # Add time and speed columns
        print("Adding time and speed columns")
        df = add_time_and_speed_to_df(df, time_step=time_step)

        # Add movement and trigger columns
        print("Adding movement and trigger columns")
        df = add_movement_started_column(df, movement_speed_threshold=movement_speed_threshold, min_time_for_movement_start=min_time_for_movement_start, min_time_for_movement_stop=min_time_for_movement_stop, time_step=time_step)
        df = add_trigger_crossed_column(df, trigger=trigger)

        # Add target-related columns
        print("Adding target-related columns")
        target_center = get_target_center(trial_data, trial_number)
        df = add_in_target_column(df, target_center=target_center, target_radius=target_radius)

        lost_status = check_lost_status(df)

        # Compute variables
        print("Computing variables")

        # Unpack the tuple returned by check_lost_status
        lost_status, lost_time = check_lost_status(df)

        # Pass only the boolean part (lost_status) to the lost_status parameter
        trial_status = get_trial_status(df, feedback=trial_feedback, lost_status=lost_status, target_radius=target_radius, time_step=time_step)
        print(f"trial_status: {trial_status}")

        t_trigger = get_t_trigger(df)
        if t_trigger is None:
            raise ValueError("t_trigger could not be computed. No trigger crossing detected.")

        RT = get_RT(df, t_trigger, min_rest=0.15)
        if RT is None or isinstance(RT, str):
            raise ValueError(f"RT could not be computed. Reason: {RT}")

        RtTrig = get_RtTrig(t_trigger, RT)
        print(f"RtTrig: {RtTrig}")
      
        t_trigger_computed = get_TrigT(RT, RtTrig)
        print(f"t_trigger_computed: {t_trigger_computed}")
        distance_to_trigger = get_trig_distance(df, t_trigger)
        print(f"distance_to_trigger: {distance_to_trigger}")
        target_enters = get_target_enters(df, min_target_time, time_step)
        print(f"target_enters: {target_enters}")
        t_first_target_enter = target_enters[0] if target_enters else None
        print(f"t_first_target_enter: {t_first_target_enter}")
        trigger_to_target_time = get_trigger_to_first_target_time(df, t_trigger, feedback=trial_feedback, lost_status=lost_status, target_radius=target_radius, min_target_time=min_target_time, time_step=time_step)
        print(f"trigger_to_target_time: {trigger_to_target_time}")
        trigger_to_target_distance = get_trigger_to_first_target_distance(df, t_trigger, feedback=trial_feedback, lost_status=lost_status, time_step=time_step)
        print(f"trigger_to_target_distance: {trigger_to_target_distance}")
        target_to_stop_time = get_target_to_stop_time(df, t_first_target_enter, movement_speed_threshold=movement_speed_threshold, min_time_for_movement=min_time_for_movement_stop, time_step=time_step, feedback=trial_feedback)
        print(f"target_to_stop_time: {target_to_stop_time}")
        target_to_stop_distance = get_target_to_stop_distance(df, t_first_target_enter, movement_speed_threshold=movement_speed_threshold, min_time_for_movement=min_time_for_movement_stop, time_step=time_step, target_center=target_center)
        print(f"target_to_stop_distance: {target_to_stop_distance}")
        total_movement_time = get_total_movement_time(df, RT)
        print(f"total_movement_time: {total_movement_time}")
        total_movement_distance = get_total_movement_distance(df, RT)
        total_distance_travelled = get_total_trial_distance(df)
        print(f"total_distance_travelled: {total_distance_travelled}")
        total_trial_time = get_total_trial_time(df)
        print(f"total_trial_time: {total_trial_time}")
        finale_distance_to_center, finale_distance_to_center_time = get_cursor_final_distance(df, movement_speed_threshold, trial_data, target_center)
        print(f"finale_distance_to_center: {finale_distance_to_center}, finale_distance_to_center_time: {finale_distance_to_center_time}")

        print("Calculating max_vx and t_max_vx")
        max_vx, t_max_vx = get_t_max_vx(df, trial_data["target_positions"][trial_number], t_trigger, t_trigger_buffer=0.12)
        print(f"max_vx: {max_vx}, t_max_vx: {t_max_vx}")
        if t_max_vx == "centre":
            TtA = "centre"
        elif t_max_vx is None:
            raise ValueError("t_max_vx could not be computed.")
        else:
            TtA = get_TtA(t_trigger, t_max_vx)
        print(f"TtA: {TtA}")
        initial_direction_angle = get_initial_direction(df, RT, time_window=0.1)
        print(f"initial_direction_angle: {initial_direction_angle}")


        # Write results to CSV
        with open('resume_resultats.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                result_file, t_trigger, RT, RtTrig, t_trigger_computed, distance_to_trigger, 
                target_enters, t_first_target_enter, trigger_to_target_time, trigger_to_target_distance, 
                target_to_stop_time, target_to_stop_distance, 
                total_movement_time, total_movement_distance, total_distance_travelled, total_trial_time, 
                finale_distance_to_center, finale_distance_to_center_time,
                max_vx, t_max_vx, TtA,
                initial_direction_angle, 
                trial_status,
                trial_data["target_positions"][trial_number],
            ])
        print(f"Trial computation completed for {result_file}")

    except Exception as e:
        print(f"Error processing trial {trial_number}: {e}")
        print("An error occurred. Check the log file for details.")
        import traceback
        with open('error_log.txt', 'a') as log_file:
            log_file.write(f"Error processing trial {trial_number}:\n")
            traceback.print_exc(file=log_file)
        with open(result_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([result_file, f"Error: {e}"])
