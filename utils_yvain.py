from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

import pandas
import pandas as pd
from shapely import Point
from sklearn.metrics import r2_score
import matplotlib.pyplot as plot # test
from math import hypot

#todo : check if we need to import this because problematic
from errors import EndOfTrialNotInTarget
from myenum import TargetPosition


# criteria definition 
# beginning/end of movement
min_time_for_movement_start = 0.2
min_time_for_movement_stop = 0.15
movement_speed_threshold = 200


min_target_time = 0.01
#todo adapt to import from config file
target_radius = 80
trial_feedback = True
#todo to import from trial-by-trial config
#todo import screen boundaries for lost status
screen_limits = {"left": 0, "right": 1680, "top": 0, "bottom": 1050}
#toutes les vitesses sont en pixel/sec

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


def add_movement_started_column(df: pd.DataFrame, vy_min: float, min_time_for_movement_start: float, timestep) -> pd.DataFrame:
    """
    Adds a 'movement' column to the DataFrame indicating whether movement has started or stopped.
    Movement starts when Vy >= vy_min for a minimum duration.
    Movement stops when both Vx and Vy are below vy_min for a minimum duration.

    """
    df.loc[:, 'movement'] = False
    movement_started = False

    for i in range(len(df)):
        if not movement_started:
            # Check if movement starts
            if abs(df['vy'][i:i + int(min_time_for_movement_start / timestep)]).ge(movement_speed_threshold).all():
                df.loc[i:, 'movement'] = True
                movement_started = True
        else:
            # Check if movement stops
            if (abs(df['vx'][i:i + int(min_time_for_movement_start / timestep)]).lt(movement_speed_threshold).all() and
                abs(df['vy'][i:i + int(min_time_for_movement_start / timestep)]).lt(movement_speed_threshold).all()):
                df.loc[i:, 'movement'] = False
                break

    return df


def add_trigger_crossed_column(df: pd.DataFrame, trigger: int) -> pd.DataFrame:
    """
        Add a boolean column 't_crossed' that indicates if the trigger has been crossed yet
    """
    # df["t_crossed"] = False
    df.loc[:, "t_crossed"] = False
    lines_below_trigger = df.loc[df["Y"] < trigger]
    if lines_below_trigger.empty:
        raise Exception("trigger not crossed")
    line_crossing_trigger = lines_below_trigger.index[0]
    # df["t_crossed"][line_crossing_trigger:] = True
    df.loc[line_crossing_trigger:, "t_crossed"] = True
    return df


def end_of_movement(df: pd.DataFrame, velocity_threshold: float = 2, min_rows_for_stop: int = 15) -> float:
    """
    Determines the time of movement stop based on velocity thresholds.
    Movement stop is defined as both vx and vy being below the velocity threshold for a minimum number of rows.
    This condition is only checked within the last second of the trial.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        velocity_threshold (float): The maximum velocity to consider the cursor as stopped (default: 2 pixels/row).
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
        if (sub_df["vx"].abs() < velocity_threshold).all() and (sub_df["vy"].abs() < velocity_threshold).all():
            return sub_df["t"].iloc[0]  # Return the time when the stop condition starts

    # If no movement stop is detected, return None
    return None

####################################################
#lost_status
####################################################

# Check if the mouse touches the limits 4 different times
def check_lost_status(df: pd.DataFrame) -> bool:
    """
    Checks if the cursor crosses the screen limits multiple times or stays on the limits for too long.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the cursor data.

    Returns:
        bool: True if the lost status condition is met, False otherwise.
    """
    limit_crossings = ((df["X"] <= screen_limits["left"]) | 
                       (df["X"] >= screen_limits["right"]) | 
                       (df["Y"] <= screen_limits["top"]) | 
                       (df["Y"] >= screen_limits["bottom"])).astype(int).diff().fillna(0).abs().sum()

    time_on_limits = df[((df["X"] <= screen_limits["left"]) | 
                         (df["X"] >= screen_limits["right"]) | 
                         (df["Y"] <= screen_limits["top"]) | 
                         (df["Y"] >= screen_limits["bottom"]))]["t"].diff().fillna(0).sum()

    lost_status = limit_crossings >= 4 or time_on_limits > 3
    return lost_status




######################################
#Reaction times
######################################
def get_RT(df: pd.DataFrame) -> float:
    """
    Determines the reaction time (RT) based on the start of movement.
    If RT is less than 0.2 seconds, it returns the time of movement after a period of 0.15 seconds of no movement.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        float: The reaction time (RT).
    """
    # Find the first time movement starts
    movement_rows = df[df["movement"] == True]
    if movement_rows.empty:
        raise ValueError("No movement detected in the DataFrame.")
    movement_start = movement_rows["t"].iloc[0]

    # Check if RT is less than 0.2 seconds
    if movement_start < 0.2:
        # Find the time after 0.15 seconds of no movement
        no_movement_period = df[(df["t"] < movement_start) & (df["movement"] == False)]
        if not no_movement_period.empty:
            last_no_movement_time = no_movement_period["t"].iloc[-1]
            if movement_start - last_no_movement_time >= 0.15:
                return movement_start

    return movement_start

def get_RtTrig(t_trigger, RT) -> float:
    #RtTrig (movement start to trigger)
    return round(t_trigger - RT, 3)


def get_t_trigger(df):
    "return the value of the line crossing the trigger"
    ser = df[df["t_crossed"] == True]['t'].head(1)
    return float(ser.iloc[0])  

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
                    target_enters.append(df.iloc[i]["t"])
                    in_target = True
        else:
            in_target = False  # Reset when leaving the target

    return target_enters



def get_trigger_to_first_target_time(df: pd.DataFrame, t_trigger: float, feedback: bool, perdu: bool, target_radius: float, time_step: float = 0.01) -> float:
    """
    Computes the time between the trigger and the first entry into the target.
    Handles different cases based on feedback and perdu status.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        t_trigger (float): The time when the trigger is crossed.
        feedback (bool): Whether feedback is enabled.
        perdu (bool): Whether the trial is marked as "perdu".
        target_radius (float): The radius of the target.
        time_step (float, optional): The time interval between consecutive rows in the DataFrame. Defaults to 0.01.

    Returns:
        float or str: The computed time or a status string.
    """
    # Case 1: If feedback is enabled
    if feedback:
        target_enters = get_target_enters(df, min_target_time, time_step)
        if target_enters and t_trigger:
            return target_enters[0] - t_trigger
        return "no target enter"

    # Case 2: If feedback is disabled and perdu is True
    if not feedback and perdu:
        return "perdu"

    # Case 3: If feedback is disabled and perdu is False
    if not feedback and not perdu:
        target_enters = get_target_enters(df, min_target_time, time_step)
        if target_enters and t_trigger:
            return target_enters[0] - t_trigger

        # Use the end_of_movement function to determine movement stop
        movement_stopped = end_of_movement(df)

        if movement_stopped is not None and movement_stopped - df["t"].iloc[-1] <= 1:
            return movement_stopped - t_trigger

        return "no target no stop"


def get_trigger_to_first_target_distance(df: pd.DataFrame, t_trigger: float, feedback: bool, perdu: bool, time_step: float = 0.01) -> float:
    """
    Computes the distance traveled by the cursor from the trigger to the first entry into the target.
    Handles different cases based on feedback and perdu status.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        t_trigger (float): The time when the trigger is crossed.
        feedback (bool): Whether feedback is enabled.
        perdu (bool): Whether the trial is marked as "perdu".
        target_radius (float): The radius of the target.
        time_step (float, optional): The time interval between consecutive rows in the DataFrame. Defaults to 0.01.

    Returns:
        float or str: The computed distance or a status string.
    """
    # Filter the DataFrame to start from the trigger time
    df = df[df["t"] >= t_trigger]

    # Case 1: If feedback is enabled
    if feedback:
        target_enters = get_target_enters(df, min_target_time, time_step)
        if target_enters:
            t_first_target_enter = target_enters[0]
            df = df[df["t"] <= t_first_target_enter]
            if not df.empty:
                return compute_distance(df)
            return "empty DataFrame"
        return "no target enter"

    # Case 2: If feedback is disabled and perdu is True
    if not feedback and perdu:
        return "perdu"

    # Case 3: If feedback is disabled and perdu is False
    if not feedback and not perdu:
        target_enters = get_target_enters(df, min_target_time, time_step)
        if target_enters:
            t_first_target_enter = target_enters[0]
            df = df[df["t"] <= t_first_target_enter]
            return compute_distance(df)

        # Use the end_of_movement function to determine movement stop
        movement_stopped = end_of_movement(df)

        if movement_stopped is not None and movement_stopped - df["t"].iloc[-1] <= 1:
            df = df[df["t"] <= movement_stopped]
            return compute_distance(df)

        return "no target no stop"


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


def get_target_to_stop_time(df: pd.DataFrame, t_first_target_enter: float, v_max: float, min_time_for_movement: float, timestep: float, feedback: bool) -> float:
    """
    Computes the time between the first target entry and the stop of the cursor, only if feedback is enabled.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        t_first_target_enter (float): The time of the first entry into the target.
        v_max (float): The maximum velocity to consider the cursor as stopped.
        min_time_for_movement (float): The minimum duration for which the velocity must remain below v_max to consider the movement stopped.
        timestep (float): The time interval between consecutive rows in the DataFrame.
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
    t_stop = end_of_movement(df, velocity_threshold=v_max, min_rows_for_stop=int(min_time_for_movement / timestep))

    if t_stop is not None:
        return t_stop - t_first_target_enter

    # If the cursor never stops, return the time until the end of the trial
    t_end_trial = df["t"].iloc[-1]
    return t_end_trial - t_first_target_enter


def get_target_to_stop_distance(df: pd.DataFrame, t_first_target_enter: float, v_max: float, min_time_for_movement: float, timestep: float, target_center: Point) -> float:
    """
    Computes the traveled distance from the first target entry to the movement stop.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        t_first_target_enter (float): The time of the first entry into the target.
        v_max (float): The maximum velocity to consider the cursor as stopped.
        min_time_for_movement (float): The minimum duration for which the velocity must remain below v_max to consider the movement stopped.
        timestep (float): The time interval between consecutive rows in the DataFrame.
        target_center (Point): The center of the target circle.

    Returns:
        float: The traveled distance from the first target entry to movement stop.
    """
    # Filter the DataFrame to start from the first target entry
    df = df[df["t"] >= t_first_target_enter]

    # Calculate the number of consecutive rows required to satisfy min_time_for_movement
    required_rows = int(min_time_for_movement / timestep)

    # Iterate through the DataFrame to check for movement stop
    for i in range(len(df) - required_rows + 1):
        sub_df = df.iloc[i:i + required_rows]
        if (sub_df["vx"].abs() <= v_max).all() and (sub_df["vy"].abs() <= v_max).all():
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



def get_total_movement_time(df: pd.DataFrame) -> float:
    """
    Computes the total duration of the movement from movement onset to movement stop.
    Movement stop is determined using the end_of_movement function.
    """
    # Find the movement onset #todo change to take the first real movement
    movement_start = df[df["movement"] == True]["t"].iloc[0]

    # Determine the movement stop using the end_of_movement function
    movement_stop = end_of_movement(df)

    # If no movement stop is detected, use the trial end time
    if movement_stop is None:
        movement_stop = df["t"].iloc[-1]

    return movement_stop - movement_start

def get_total_distance(df: pd.DataFrame) -> float:
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

#todo add get_total_distance function?


def compute_final_distance(x, y, target_center):
    return hypot(x - target_center[0], y - target_center[1])


def get_cursor_final_distance(df, v_max: float, trial_data, target_center: Point):
    """
    Computes the final distance of the cursor from the target center.
    If the cursor stops moving (velocity below v_max), the distance is computed at the end of movement.
    If the cursor never stops moving, the distance is computed at the end of the trial.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        v_max (float): The maximum velocity to consider the cursor as stopped.
        trial_data (dict): The trial data.
        target_center (Point): The center of the target circle.

    Returns:
        tuple: The final distance and the time it was computed.
    """
    # Determine the time of movement stop using the end_of_movement function
    t_stop = end_of_movement(df, velocity_threshold=v_max)

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
#todo y'a l'air d'avoir un pb tuple/qu'est ce qui est utilisé?

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
#prendre scalaire? de vit instantannée à 100ms après trigger?

def get_initial_direction(df: pd.DataFrame, t_trigger: float, time_window: float = 0.1) -> float:
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
    df_window = df[(df["t"] >= t_trigger) & (df["t"] <= t_trigger + time_window)]

    # Compute the average velocity in x and y directions
    avg_vx = df_window["vx"].mean()
    avg_vy = df_window["vy"].mean()

    # Compute the angle of the velocity vector in degrees
    angle = np.degrees(np.arctan2(avg_vy, avg_vx))
    return angle

def draw_my_plot(rx, ry, regressor, df):
    plot.scatter(df["X"], df["Y"], color='green')
    plot.scatter(rx, ry, color='red')
    plot.plot(rx, regressor.predict(rx), color='blue')
    plot.title("My plot")
    plot.xlabel('x')
    plot.ylabel('y')
    plot.show()


def get_linear_regression(df: pandas.DataFrame, t_start, t_end) -> tuple:
    regressor = LinearRegression()
    sub_df = df[(df["t"] >= t_start) & (df["t"] <= t_end)]

    rx = sub_df["X"].values.reshape(-1, 1)
    ry = sub_df["Y"].values
    regressor.fit(rx, ry)
    r2score = regressor.score(rx, ry)
    # plot = draw_my_plot(rx=rx, ry=ry, regressor=regressor, df=df)
    # coefficient y = a*x +b   : a=regressor.coef_  b=regressor.intercept_
    return regressor.coef_, regressor.intercept_, r2score






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

#todo improve

def get_trial_status(df):
    try:
        if get_target_enters(df, min_target_time) is not None:
            return "Success"
        elif get_target_enters(df, min_target_time) is None:
            return "Fail"
        else:
            return "Unknown"
    except Exception as e:
        return f"Error: {e}"


#add function to indicate if feedback is on or off


"""
def compute_trial(result_file: Path, trial_number: int, trial_data: dict, trigger: int, df=None, timestep=0.01,
                  minimum_target_time=min_target_time):
    RT = t_trigger = RtTrig = t_final_target_enter = t_final_core_enter = None
    TTTarg = ca = t_max_vx = TTrig = TtA = equation_a = pente_droite = r2score = None

    try:
        # définit des paramètres
        # period_min = 0.01  # temps d'activité minimum pour considérer un début de mouvement - before set to 0.05 - check if reasonable to change TODO
        # vy_min = 300  # 3 pixel/sec est la vitesse minimale pour dire qu'il y un début de mouvement
        # charger un fichier
        # df = load_trial_data()

        print(f"Processing {result_file}")
        print(trial_data)
        print(trial_number)

        target_position = trial_data["target_positions"][trial_number]
        target_polygon = get_target_polygon(trial_data=trial_data, trial_number=trial_number)
        target_center = get_target_center(trial_data, trial_number)
        target_core_polygon = get_target_core_polygon(trial_data, trial_number=trial_number)

        # ajouter les vitesses au dataframe
        print("Adding time and speed to dataframe")
        df = add_time_and_speed_to_df(df=df, time_step=timestep)
        print("Adding movement started column")
        df = add_movement_started_column(df=df, vy_min=vy_min, min_move_time=period_min, timestep=timestep)
        print("Adding trigger crossed column")
        df = add_trigger_crossed_column(df=df, trigger=trigger)
        print("Adding in target column")
        df = add_in_target_column(df=df, target_center=target_center, target_radius=target_radius)


        # calculer le temps de début de mouvement
        print("Calculating RT")
        RT = get_RT(df)
        print("Calculating t_trigger")
        t_trigger = get_t_trigger(df)
        print("Calculating RtTrig")
        RtTrig = get_RtTrig(t_trigger=t_trigger, RT=RT)
        print("Calculating time_first_target_enter")
        t_first_target_enter = get_t_first_target_enter(df=df)
        #print("Calculating t_final_target_enter")
        #t_final_target_enter = get_t_final_target_enter(df=df, minimum_time_in=minimum_target_time)

        print("Calculating TTTarg")
        TTTarg = get_TTTarg(df=df, t_trigger=t_trigger, t_final_target_enter=t_final_target_enter)
        print("Calculating ca")
        ca = get_ca(t_final_target_enter=t_final_target_enter, t_final_core_enter=t_final_core_enter)

        print("Calculating max_vx and t_max_vx")
        max_vx, t_max_vx = get_t_max_vx(df=df, target_position=trial_data["target_positions"][trial_number],
                                        t_trigger=t_trigger, t_trigger_buffer=0.12)  # todo : corriger

        print("Calculating TTrig")
        TTrig = get_TTrig(RT=RT, RtTrig=RtTrig)
        print("Calculating TtA")
        TtA = get_TtA(t_trigger, t_max_vx)

        print("Calculating finale_distance_to_center and finale_distance_to_center_time")
        finale_distance_to_center, finale_distance_to_center_time = get_cursor_final_distance(df=df, v_max=max_final_speed, trial_data=trial_data,
                                                            target_center=target_center)

        print("Calculating total_distance_travelled")
        total_distance_travelled = get_total_distance(df=df)

        TStop = None
        if not finale_distance_to_center_time and t_final_target_enter:
            TStop = df.iloc[-1]["t"] - t_final_target_enter
        elif t_final_target_enter:
            TStop = finale_distance_to_center_time - t_final_target_enter
        else:
            TStop = None

        # y = a * x + b
        print("Calculating linear regression")
        equation_a, pente_droite, r2score = get_linear_regression(df=df, t_start=RT, t_end=RT + 0.1)
        df.to_csv(result_file)
        print(f"Trial computation completed for {result_file}")

    except EndOfTrialNotInTarget:
        print(f"End of trial not in target : {trial_number} {result_file}")
    except Exception as e:
        with open(result_file, 'w') as fd:
            fd.write(f"Error : {e}")
            print(f"Error :{result_file} {e}")

    with open('resume_resultats.csv', 'a') as fd:
        if t_max_vx is not None:
            if t_max_vx != "centre":
                t_max_vx = float(t_max_vx)
        if TTrig is not None:
            TTrig = float(TTrig)
        if TtA is not None and TtA != "centre":
            TtA = float(TtA)
        if equation_a is not None:
            equation_a = float(equation_a)
        fd.write(
            f"{result_file}, {RT}, {t_trigger}, {RtTrig}, {t_final_target_enter}, {t_final_core_enter}, {TTTarg}, {ca}, "
            f"{t_max_vx}, {max_vx}, {TTrig}, {TtA}, {equation_a}, {pente_droite}, {r2score}, {finale_distance_to_center}, {TStop}, "
            f"{target_position}, {total_distance_travelled}\n"
        )
"""


import csv

def compute_trial(result_file: Path, trial_number: int, trial_data: dict, trigger: int, df=None, timestep=0.01,
                  minimum_target_time=0.01, vy_min=200, period_min=0.01, max_final_speed=200):
    """
    Computes trial variables and writes them to a CSV file.

    Parameters:
        result_file (Path): Path to the result file.
        trial_number (int): The trial number.
        trial_data (dict): Trial-specific data.
        trigger (int): Trigger value.
        df (pd.DataFrame): DataFrame containing trial data.
        timestep (float): Time step between rows in the DataFrame.
        minimum_target_time (float): Minimum time in the target to consider entry valid.
        vy_min (float): Minimum velocity threshold for movement detection.
        period_min (float): Minimum duration for movement detection.
        max_final_speed (float): Maximum velocity to consider the cursor as stopped.
    """
    try:
        print(f"Starting computation for trial {trial_number}")
        # Initialize variables
        RT = t_trigger = RtTrig = t_trigger_computed = None
        t_first_target_enter = t_max_vx = TtA = equation_a = pente_droite = r2score = None
        finale_distance_to_center = finale_distance_to_center_time = total_distance_travelled = TStop = None

        # Add time and speed columns
        print("Adding time and speed columns")
        df = add_time_and_speed_to_df(df, time_step=timestep)

        # Add movement and trigger columns
        print("Adding movement and trigger columns")
        df = add_movement_started_column(df, vy_min=vy_min, min_time_for_movement_start=period_min, timestep=timestep)
        df = add_trigger_crossed_column(df, trigger=trigger)

        # Add target-related columns
        print("Adding target-related columns")
        target_center = get_target_center(trial_data, trial_number)
        df = add_in_target_column(df, target_center=target_center, target_radius=target_radius)

        lost_status = check_lost_status(df)

        # Compute variables
        print("Computing variables")
        RT = get_RT(df)
        print(f"RT: {RT}")
        t_trigger = get_t_trigger(df)
        print(f"t_trigger: {t_trigger}")
        if t_trigger is None:
            raise ValueError("t_trigger could not be computed.")

        RtTrig = get_RtTrig(t_trigger, RT)
        print(f"RtTrig: {RtTrig}")
        t_trigger_computed = get_TrigT(RT, RtTrig)
        print(f"t_trigger_computed: {t_trigger_computed}")
        distance_to_trigger = get_trig_distance(df, t_trigger)
        print(f"distance_to_trigger: {distance_to_trigger}")
        target_enters = get_target_enters(df, minimum_target_time, timestep)
        print(f"target_enters: {target_enters}")
        t_first_target_enter = target_enters[0] if target_enters else None
        print(f"t_first_target_enter: {t_first_target_enter}")
        trigger_to_target_time = get_trigger_to_first_target_time(df, t_trigger, feedback=trial_feedback, perdu=lost_status, target_radius=target_radius, time_step=timestep)
        print(f"trigger_to_target_time: {trigger_to_target_time}")
        trigger_to_target_distance = get_trigger_to_first_target_distance(df, t_trigger, feedback=trial_feedback, perdu=lost_status, time_step=timestep)
        print(f"trigger_to_target_distance: {trigger_to_target_distance}")
        target_to_stop_time = get_target_to_stop_time(df, t_first_target_enter, v_max=movement_speed_threshold, min_time_for_movement=min_time_for_movement_stop, timestep=timestep, feedback=trial_feedback)
        print(f"target_to_stop_time: {target_to_stop_time}")
        target_to_stop_distance = get_target_to_stop_distance(df, t_first_target_enter, v_max=movement_speed_threshold, min_time_for_movement=min_time_for_movement_stop, timestep=timestep, target_center=target_center)
        print(f"target_to_stop_distance: {target_to_stop_distance}")
        total_movement_time = get_total_movement_time(df)
        print(f"total_movement_time: {total_movement_time}")
        total_distance_travelled = get_total_distance(df)
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
        equation_a, pente_droite, r2score = get_linear_regression(df, t_start=RT, t_end=RT + 0.1)
        print(f"equation_a: {equation_a}, pente_droite: {pente_droite}, r2score: {r2score}")

        # Write results to CSV
        with open(result_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                result_file, RT, RtTrig, t_trigger, t_trigger_computed, distance_to_trigger, 
                target_enters, t_first_target_enter, trigger_to_target_time, trigger_to_target_distance, 
                target_to_stop_time, target_to_stop_distance, 
                total_movement_time, total_distance_travelled, total_trial_time, 
                finale_distance_to_center, finale_distance_to_center_time,
                max_vx, t_max_vx, TtA,
                equation_a, pente_droite, r2score, 
                trial_data["target_positions"][trial_number],
            ])
        print(f"Trial computation completed for {result_file}")

    except Exception as e:
        print(f"Error processing trial {trial_number}: {e}")
        print("Detailed error traceback:")
        import traceback
        traceback.print_exc()
        with open(result_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([result_file, f"Error: {e}"])
