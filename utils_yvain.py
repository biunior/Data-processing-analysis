from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

import pandas
import pandas as pd
from shapely import Point, Polygon
from sklearn.metrics import r2_score
import matplotlib.pyplot as plot

#todo : check if we need to import this because problematic
from errors import EndOfTrialNotInTarget
from myenum import TargetPosition


# criteria definition 
# beginning/end of movement
min_time_for_movement_start = 0.2
min_time_for_movement_stop = 0.15
speed_criteria_movement = 200


min_time_in_target = 0.01
#add radius of target circle from config file, 
#todo adapt to import from config file
target_radius = 80

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


def add_movement_started_column(df: pd.DataFrame, vy_min: float, min_move_time: float, timestep) -> pd.DataFrame:
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
            if abs(df['vy'][i:i + int(min_time_for_movement / timestep)]).ge(min_speed_movement).all():
                df.loc[i:, 'movement'] = True
                movement_started = True
        else:
            # Check if movement stops
            if (abs(df['vx'][i:i + int(min_time_for_movement / timestep)]).lt(min_speed_movement).all() and
                abs(df['vy'][i:i + int(min_time_for_movement / timestep)]).lt(min_speed_movement).all()):
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


##################################
#from beginning to trigger#
##################################

#todo : if < 0.2s return "movement start before beginning". Si possible faire en sorte de regarder le premier départ après min 0.15s d'arrêt
def get_RT(df: pd.DataFrame) -> float:
    #RT: Temps jusqu'au début du mouvement
    res = df[df["movement"] == True]['t'].head(1)
    return float(res.iloc[0])



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
def add_in_circle_column(df: pd.DataFrame, target_center: Point, target_radius: float, column_name: str):
    """
    Adds a boolean column to the DataFrame indicating whether each point is inside the circle.
    Parameters:
        df (pd.DataFrame): The DataFrame containing the points.
        center (Point): The center of the circle.
        radius (float): The radius of the circle.
        column_name (str): The name of the new column.
    """
    df[column_name] = df.apply(lambda row: target_center.distance(Point(row["X"], row["Y"])) <= target_radius, axis=1)
    return df

def add_in_target_column(df: pd.DataFrame, target_center: Point, target_radius: float) -> pd.DataFrame:
    """
    Adds a column indicating whether each point is inside the target circle.
    Parameters:
        df (pd.DataFrame): The DataFrame containing the points.
        target_center (Point): The center of the target circle.
        target_radius (float): The radius of the target circle.
    """
    return add_in_circle_column(df=df, center=target_center, radius=target_radius, column_name="in_target")


#todo check if need to changed for cases of entering for the real times happens only 0,1s before end of trial
# faire avec un critère de déscélération pour avoir un "real" temps d'entrée dans la cible?
def get_first_target_enter(df: pd.DataFrame, min_time_in_target: float, time_step: float = 0.01):
    """
    Returns the time t where the subject first enters the target and stays in the target for at least `min_time_in_target` seconds. 
    min_time_in_target is equal to 0.01s now (1 row)

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        min_time_in_target (float): The minimum time (in seconds) the subject must stay in the target.
        time_step (float): The time interval between consecutive rows in the DataFrame.

    Returns:
        float or None: The time of the first valid target entry, or None if no such entry exists.
    """
    required_rows = int(min_time_in_target / time_step)

    # Iterate through rows where 'in_target' is True
    for i in range(len(df)):
        if df.loc[i, "in_target"]:
            # Check if the subject stays in the target for the required duration
            if df.loc[i:i + required_rows - 1, "in_target"].all():
                return df.loc[i, "t"]
    return None




def get_trigger_to_first_target_time(df: pd.DataFrame, t_trigger: float, feedback: bool, target_radius: float, time_step: float = 0.01) -> float:
    """
    todo: revoir pour rajouter "perdu"? et vérifier que ça correspond au word.

    Computes the time between the trigger and the first entry into the target.
    If feedback is False and there is no target entry, computes the time between the trigger and the end of movement.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        t_trigger (float): The time when the trigger is crossed.
        feedback (bool): Whether feedback is enabled.
        target_radius (float): The radius of the target.
        time_step (float, optional): The time interval between consecutive rows in the DataFrame. Defaults to 0.01.

    Returns:
        float: The computed time.
    """
    # Case 1: Compute time to first  target entry
    t_first_target_enter = get_first_target_enter(df)
    if t_first_target_enter is not None:
        return t_first_target_enter - t_trigger

    # Case 2: If feedback is False and no target entry, compute time to end of movement
    if not feedback:
        required_rows = int(15)  # 15 rows for 1 second
        movement_stopped = None

        for i in range(len(df) - required_rows + 1):
            sub_df = df.iloc[i:i + required_rows]
            if (sub_df["vx"].abs() < min_speed_movement).all() and (sub_df["vy"].abs() < min_speed_movement).all():
                movement_stopped = sub_df["t"].iloc[0]
                break

        if movement_stopped is not None and movement_stopped - df["t"].iloc[-1] <= 1:
            return movement_stopped - t_trigger





def get_trigger_to_first_target_distance(df: pd.DataFrame, t_trigger: float) -> float:
    """
    Computes the  distance traveled by the cursor from the trigger to the final target entry.
    If the cursor does not enter the target, the distance is computed from the trigger to the end of the trial.
    #todo change to if no target enter-->return "no target enter"
    todo follow same logic as get_trigger_to_first_target_time
    Returns:
        float: The distance traveled by the cursor.
    """
    # Filter the DataFrame to start from the trigger time
    df = df[df["t"] >= t_trigger]

    # Check if the subject ever enters the target
    if "in_target" in df.columns and df["in_target"].any():
        # Get the time of the final target entry
        t_final_target_enter = df[df["in_target"] == True]["t"].iloc[-1]
        # Filter the DataFrame up to the final target entry
        df = df[df["t"] <= t_final_target_enter]

    # Compute the differences between consecutive coordinates
    diffs = df.diff()
    # Compute the Euclidean distance between each pair of consecutive coordinates
    distances = np.sqrt(diffs['X'] ** 2 + diffs['Y'] ** 2)
    # Compute the sum of distances
    sum_distance = distances.sum()
    return sum_distance



################################################################################
#from target to stop#
################################################################################


def get_target_to_stop_time(df: pd.DataFrame, t_first_target_enter: float, v_max: float, min_time_for_movement: float, timestep: float) -> float:
    """
    Computes the time between the first target entry and the stop of the cursor.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        t_first_target_enter (float): The time of the first entry into the target.
        v_max (float): The maximum velocity to consider the cursor as stopped.
        min_time_for_movement (float): The minimum duration for which the velocity must remain below v_max to consider the movement stopped.
        timestep (float): The time interval between consecutive rows in the DataFrame.

    Returns:
        float: The time between the first target entry and the stop of the cursor.
    """
    # Filter the DataFrame to start from the first target entry
    df = df[df["t"] >= t_first_target_enter]

    # Calculate the number of consecutive rows required to satisfy end movement : todo 15 rows at < 200pixel/sec
    required_rows = int(min_time_for_movement / timestep)

    if not t_first_target_enter:
        return "not in target"

    # Iterate through the DataFrame to check for movement stop
    for i in range(len(df) - required_rows + 1):
        sub_df = df.iloc[i:i + required_rows]
        if (sub_df["vx"].abs() <= v_max).all() and (sub_df["vy"].abs() <= v_max).all():
            t_stop = sub_df["t"].iloc[0]  # Time when the stop condition starts
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
    Movement stop is defined as both vx and vy being < 2 pixels/row for at least 15 consecutive rows.
    This criteria is only valid within the last second of the trial.
    """
    # Define the velocity threshold and the minimum number of rows for movement stop
    velocity_threshold = 200
    min_rows_for_stop = 15

    # Find the movement onset
    movement_start = df[df["movement"] == True]["t"].iloc[0]

    # Filter the DataFrame to include only the last second of the trial
    trial_end_time = df["t"].iloc[-1]
    valid_df = df[df["t"] >= trial_end_time - 1]

    # Iterate through the filtered DataFrame to find the movement stop
    for i in range(len(valid_df) - min_rows_for_stop + 1):
        sub_df = valid_df.iloc[i:i + min_rows_for_stop]
        if (sub_df["vx"].abs() < velocity_threshold).all() and (sub_df["vy"].abs() < velocity_threshold).all():
            movement_stop = sub_df["t"].iloc[0]
            return movement_stop - movement_start

    # If no movement stop is detected, return the duration up to the trial end
    return trial_end_time - movement_start



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
    cursor_point = Point(x, y)
    return cursor_point.distance(target_center)




#TOdo : modify to compute get_final_distance at the time where movement stop (final time) or at the end of the trial if never stops
def get_cursor_final_distance(df, v_max: float, trial_data, target_center: Point):
    """
    Computes the final distance of the cursor from the target center.
    see get_total_movement_time for the definition of movement stop.
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
    # Filter rows where the cursor is in the target and velocity is below the threshold
    my_df = df[(df["in_target"]) & (abs(df["vx"]) <= v_max) & (abs(df["vy"]) <= v_max)]

    # Check for the end of movement
    for row in my_df.iterrows():
        sub_df = df[(df["t"] >= row[1]["t"]) & (df["t"] <= row[1]["t"] + 0.2)]
        vy_ok = abs(sub_df["vy"]).le(v_max).all()
        vx_ok = abs(sub_df["vx"]).le(v_max).all()
        in_target_ok = sub_df["in_target"].all()
        if vy_ok and vx_ok and in_target_ok:
            # Compute distance at the end of movement
            return compute_final_distance(row[1]["X"], row[1]["Y"], target_center), row[1]["t"]

    # If no end of movement is detected, compute distance at the end of the trial
    last_row = df.iloc[-1]
    return compute_final_distance(last_row["X"], last_row["Y"], target_center), last_row["t"]


##################################
#RT movement correction
##################################

def get_TtA(t_trigger: float, t_max_vx):
    # TtA: Temps de réaction 2 = Temps entre le passage du trigger et l 'extremum de vitesse en X (pixel/s)
    if t_max_vx == "centre":
        return "centre"
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

# change to vecteur instantanné de vitesse à TTrig+100ms
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
"""
def get_trial_status(df):
    if get_t_final_target_enter(df, min_time_in_target) is not None:
        return "Success"
    elif Space:
        return "Space"
    elif get_t_final_target_enter(df, min_time_in_target) is None:
        return "Fail"
    else:
        return "Unknown"

"""
#add function to indicate if feedback is on or off




def compute_trial(result_file: Path, trial_number: int, trial_data: dict, trigger: int, df=None, timestep=0.01,
                  minimum_target_time=min_time_in_target):
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
        print("Adding in target core column")
        df = add_in_target_core_column(df, target_core=target_core_polygon)

        # calculer le temps de début de mouvement
        print("Calculating RT")
        RT = get_RT(df)
        print("Calculating t_trigger")
        t_trigger = get_t_trigger(df)
        print("Calculating RtTrig")
        RtTrig = get_RtTrig(t_trigger=t_trigger, RT=RT)
        t_first_target_enter = get_t_first_target_enter(df=df)
        print("Calculating t_final_target_enter")
        t_final_target_enter = get_t_final_target_enter(df=df, minimum_time_in=minimum_target_time)

        print("Calculating t_final_core_enter")
        t_final_core_enter = get_t_final_core_enter(df=df, minimum_time_in=minimum_target_time)
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

        print("Calculating dist_final and t_dist_final")
        dist_final, t_dist_final = get_cursor_final_distance(df=df, v_max=max_final_speed, trial_data=trial_data,
                                                            target_center=target_center)

        print("Calculating total_distance_travelled")
        total_distance_travelled = get_total_distance(df=df)

        TStop = None
        if not t_dist_final and t_final_target_enter:
            TStop = df.iloc[-1]["t"] - t_final_target_enter
        elif t_final_target_enter:
            TStop = t_dist_final - t_final_target_enter
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
            f"{result_file}, {RT}, {t_trigger}, {RtTrig}, {t_final_target_enter}, {t_final_core_enter}, {TTTarg}, {ca}, {t_max_vx}, {max_vx}, {TTrig}, {TtA}, {equation_a}, {pente_droite}, {r2score}, {dist_final}, {TStop}, {target_position}, {total_distance_travelled}\n")





















#unused functions (target core)
"""
def add_in_target_core_column(df: pd.DataFrame, target_core: Polygon) -> pd.DataFrame:
    return add_in_polygon_colon(df=df, polygon=target_core, column_name="in_target_core")



def get_target_core_polygon(trial_data: dict, trial_number: int):
    target_position = trial_data["target_positions"][trial_number]
    if target_position == "gauche":
        x = trial_data["centre_cible_gauche"].x
        y = trial_data["centre_cible_gauche"].y
        return Polygon([(x - 5, y - 5), (x + 5, y - 5), (x + 5, y + 5), (x - 5, y + 5)])
    elif target_position == "centre":
        x = trial_data["centre_cible_centre"].x
        y = trial_data["centre_cible_centre"].y
        return Polygon([(x - 5, y - 5), (x + 5, y - 5), (x + 5, y + 5), (x - 5, y + 5)])
    elif target_position == "droite":
        x = trial_data["centre_cible_droite"].x
        y = trial_data["centre_cible_droite"].y
        return Polygon([(x - 5, y - 5), (x + 5, y - 5), (x + 5, y + 5), (x - 5, y + 5)])
    else:
        raise Exception("Unknown target position")


def get_t_final_core_enter(df: pandas.DataFrame, minimum_time_in: float = 0.5):
    
    #L'entrée finale dans le cœur de cible est valide si le sujet reste dans le cœur pendant 0,5 secondes minimum

    

    for row in df[df["in_target_core"] == True].iterrows():
        # get all rows starting between row.t and row.t+minimum time. Check if they are all true
        # if the final enter is less than 0.5 seconds before the end. It is accepted
        sub_df = df[(df["t"] >= row[1]["t"]) & (df["t"] <= row[1]["t"] + minimum_time_in)]
        if sub_df["in_target_core"].all():
            return row[1]["t"]
    return None


def get_ca(t_final_target_enter: float, t_final_core_enter: float):
    
    #temps entre l'entrée finale dans la cible et l'entrée finale dans le cœur de cible
    #si la souris n'entre pas dans le cœur de cible ou reste moins de 0,5s, C-A= 1 seconde
    
    if t_final_target_enter is None:
        return "Echec"
    if t_final_core_enter is None:
        # si t_final_core_enter is None, c'est qu'il n'est pas entré assez longtemps dans le coeur,
        return 1
    return t_final_core_enter - t_final_target_enter
"""


###old functions to define in target with polygon
"""
def add_in_polygon_colon(df: pd.DataFrame, polygon: Polygon, column_name: str):
    df[column_name] = False
    for index, row in df.iterrows():
        df.loc[index, column_name] = polygon.covers(Point(row["X"], row["Y"]))
        # df[column_name][index] = polygon.contains(Point(row["X"], row["Y"]))
    return df


def add_in_target_column(df: pd.DataFrame, target_polygon: Polygon) -> pd.DataFrame:
    return add_in_polygon_colon(df=df, polygon=target_polygon, column_name="in_target")


def get_target_polygon(trial_data: dict, trial_number: int):
    target_position = trial_data["target_positions"][trial_number]
    if target_position == "gauche":
        return Polygon([(c.x, c.y) for c in trial_data["cible_gauche"]])
    elif target_position == "centre":
        return Polygon([(c.x, c.y) for c in trial_data["cible_centre"]])
    elif target_position == "droite":
        return Polygon([(c.x, c.y) for c in trial_data["cible_droite"]])
    else:
        raise Exception("Unknown target position")

"""
#old get_t_final_target_enter function
"""
def get_t_final_target_enter(df: pd.DataFrame, minimum_time_in: float):
     return the time where it enters the target and don't get out
     - the time where it enters the target core and don't get out

    L'entrée finale dans le cœur de cible est valide si le sujet reste dans le cœur pendant 0,5 secondes minimum


    

    for row in df[df["in_target"] == True].iterrows():
        # get all rows starting between row.t and row.t+minimum time. Check if they are all true
        # if the final enter is less than "min_time_in_target" before the end. It is accepted
        sub_df = df[(df["t"] >= row[1]["t"]) & (df["t"] <= row[1]["t"] + minimum_time_in)]
        if sub_df["in_target"].all():
            return row[1]["t"]
    return None


def get_final_target_entry(df: pd.DataFrame, return_full_row: bool = False):
    
    Returns the time of the final target entry or the first row of the last period where the cursor is in the target.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        return_full_row (bool): If True, returns the first row of the last period where the cursor is in the target.
                                If False, returns the time of the final target entry. 
                                Is there a difference between the two? TODo

    Returns:
        float, pd.Series, or None: The time of the final target entry (if return_full_row is False),
                                   the first row of the last period (if return_full_row is True),
                                   or None if no target entry is found.
    
    if "in_target" not in df.columns:
        raise ValueError("The DataFrame must contain an 'in_target' column.")

    # Reverse the DataFrame to find the last period where 'in_target' is True
    reversed_df = df.iloc[::-1]
    last_target_block = reversed_df[reversed_df["in_target"] == True]

    if last_target_block.empty:
        return None  # No period where the cursor is in the target

    if return_full_row:
        # Return the first row of the last block
        return last_target_block["t"].iloc[-1]
    else:
        # Return the time of the final target entry
        return float(last_target_block["t"].iloc[0])
    
"""
