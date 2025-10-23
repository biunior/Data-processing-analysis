from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

import pandas
import pandas as pd
from shapely import Point, Polygon
from sklearn.metrics import r2_score
import matplotlib.pyplot as plot

from errors import EndOfTrialNotInTarget
from myenum import TargetPosition


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
    # df['movement'] = False
    df.loc[:, 'movement'] = False
    for i in range(len(df)):
        if abs(df['vy'][i:i + int(min_move_time / timestep)]).ge(vy_min).all():
            # df['movement'][i:] = True
            df.loc[i:, 'movement'] = True
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


def get_dt(t_trigger, ta) -> float:
    "D-T : Temps début de mouvement --> passage du trigger"
    return round(t_trigger - ta, 3)


def get_ta(df: pd.DataFrame) -> float:
    "Ta: Temps à l 'arrêt (avant le début du mouvement)"
    res = df[df["movement"] == True]['t'].head(1)
    return float(res.iloc[0])


def get_t_trigger(df):
    "return the valuet of the line crossing the trigger"
    ser = df[df["t_crossed"] == True]['t'].head(1)
    return float(ser.iloc[0])  


def add_in_polygon_colon(df: pd.DataFrame, polygon: Polygon, column_name: str):
    df[column_name] = False
    for index, row in df.iterrows():
        df.loc[index, column_name] = polygon.covers(Point(row["X"], row["Y"]))
        # df[column_name][index] = polygon.contains(Point(row["X"], row["Y"]))
    return df


def add_in_target_column(df: pd.DataFrame, target_polygon: Polygon) -> pd.DataFrame:
    return add_in_polygon_colon(df=df, polygon=target_polygon, column_name="in_target")


def add_in_target_core_column(df: pd.DataFrame, target_core: Polygon) -> pd.DataFrame:
    return add_in_polygon_colon(df=df, polygon=target_core, column_name="in_target_core")


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
    """
    L'entrée finale dans le cœur de cible est valide si le sujet reste dans le cœur pendant 0,5 secondes minimum

    """

    for row in df[df["in_target_core"] == True].iterrows():
        # get all rows starting between row.t and row.t+minimum time. Check if they are all true
        # if the final enter is less than 0.5 seconds before the end. It is accepted
        sub_df = df[(df["t"] >= row[1]["t"]) & (df["t"] <= row[1]["t"] + minimum_time_in)]
        if sub_df["in_target_core"].all():
            return row[1]["t"]
    return None


def get_t_first_target_enter(df: pd.DataFrame):
    """
    returns the time t where the subject first enters the target.
    It is not the same as the final target enter, as the subject can get out of the target.
    """
    # get first row in target
    first_row_in_target = df.loc[df["in_target"] == True].head(1)
    return first_row_in_target["t"]


def get_t_final_target_enter(df: pd.DataFrame, minimum_time_in: float):
    """ return the time where it enters the target and don't get out
     - the time where it enters the target core and don't get out

    L'entrée finale dans le cœur de cible est valide si le sujet reste dans le cœur pendant 0,5 secondes minimum


    """

    for row in df[df["in_target"] == True].iterrows():
        # get all rows starting between row.t and row.t+minimum time. Check if they are all true
        # if the final enter is less than 0.5 seconds before the end. It is accepted
        sub_df = df[(df["t"] >= row[1]["t"]) & (df["t"] <= row[1]["t"] + minimum_time_in)]
        if sub_df["in_target"].all():
            return row[1]["t"]
    return None


def get_tc(df: pd.DataFrame, t_trigger: float, t_final_target_enter: float):
    """
    Temps Trigger-->rentrée "finale" dans cible (la souris rentre et n'en ressort plus)
    """
    if not t_final_target_enter:
        return "Echec"
    return t_final_target_enter - t_trigger


def get_ca(t_final_target_enter: float, t_final_core_enter: float):
    """
    temps entre l'entrée finale dans la cible et l'entrée finale dans le cœur de cible
    si la souris n'entre pas dans le cœur de cible ou reste moins de 0,5s, C-A= 1 seconde
    """
    if t_final_target_enter is None:
        return "Echec"
    if t_final_core_enter is None:
        # si t_final_core_enter is None, c'est qu'il n'est pas entré assez longtemps dans le coeur,
        return 1
    return t_final_core_enter - t_final_target_enter


def get_tr1(ta: float, dt: float):
    # Temps de réaction 1= Temps à l'arrêt + Temps D-T
    return ta + dt


def get_tr2(t_trigger: float, t_max_vx):
    # Tr2: Temps de réaction 2 = Temps entre le passage du trigger et l 'extremum de vitesse en X (pixel/s)
    if t_max_vx == "centre":
        return "centre"
    return t_max_vx - t_trigger


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


def compute_distance(x, y, target_core):
    return ((x - target_core.x) ** 2 + (y - target_core.y) ** 2) ** 0.5


def get_cursor_core_distance(df, v_max: float, trial_data, target_core):
    my_df = df[(df["in_target"]) & (abs(df["vx"]) <= v_max) & (abs(df["vy"]) <= v_max)]
    for row in my_df.iterrows():
        sub_df = df[(df["t"] >= row[1]["t"]) & (df["t"] <= row[1]["t"] + 0.2)]
        vy_ok = abs(sub_df["vy"]).le(v_max).all()
        vx_ok = abs(sub_df["vx"]).le(v_max).all()
        in_target_ok = sub_df["in_target"].all()
        if vy_ok and vx_ok and in_target_ok:
            return compute_distance(row[1]["X"], row[1]["Y"], target_core), row[1]["t"]
    return None, None


def get_target_core(trial_data, trial_number):
    target_position = trial_data["target_positions"][trial_number]
    if target_position == "gauche":
        return trial_data["centre_cible_gauche"]
    elif target_position == "centre":
        return trial_data["centre_cible_centre"]
    elif target_position == "droite":
        return trial_data["centre_cible_droite"]
    else:
        raise Exception("Unknown target position")


def get_total_distance(df):
    # Compute the differences between consecutive coordinates
    diffs = df.diff()
    # Compute the Euclidean distance between each pair of consecutive coordinates
    distances = np.sqrt(diffs['X'] ** 2 + diffs['Y'] ** 2)
    # Compute the sum of distances
    sum_distance = distances.sum()
    return sum_distance


def compute_trial(result_file: Path, trial_number: int, trial_data: dict, trigger: int, df=None, timestep=0.01,
                  minimum_target_time=0.4):
    ta = t_trigger = dt = t_final_target_enter = t_final_core_enter = None
    tc = ca = t_max_vx = tr1 = tr2 = equation_a = pente_droite = r2score = None

    try:
        # définit des paramètres
        period_min = 0.01  # temps d'activité minimum pour considérer un début de mouvement - before set to 0.05 - check if reasonable to change TODO
        vy_min = 300  # 3 pixel/sec est la vitesse minimale pour dire qu'il y un début de mouvement
        # charger un fichier
        # df = load_trial_data()

        print(f"Processing {result_file}")
        print(trial_data)
        print(trial_number)

        target_position = trial_data["target_positions"][trial_number]
        target_polygon = get_target_polygon(trial_data=trial_data, trial_number=trial_number)
        target_core = get_target_core(trial_data, trial_number)
        target_core_polygon = get_target_core_polygon(trial_data, trial_number=trial_number)

        # ajouter les vitesses au dataframe
        print("Adding time and speed to dataframe")
        df = add_time_and_speed_to_df(df=df, time_step=timestep)
        print("Adding movement started column")
        df = add_movement_started_column(df=df, vy_min=vy_min, min_move_time=period_min, timestep=timestep)
        print("Adding trigger crossed column")
        df = add_trigger_crossed_column(df=df, trigger=trigger)
        print("Adding in target column")
        df = add_in_target_column(df=df, target_polygon=target_polygon)
        print("Adding in target core column")
        df = add_in_target_core_column(df, target_core=target_core_polygon)

        # calculer le temps de début de mouvement
        print("Calculating ta")
        ta = get_ta(df)
        print("Calculating t_trigger")
        t_trigger = get_t_trigger(df)
        print("Calculating dt")
        dt = get_dt(t_trigger=t_trigger, ta=ta)
        # t_first_target_enter = get_t_first_target_enter(df=df)
        print("Calculating t_final_target_enter")
        t_final_target_enter = get_t_final_target_enter(df=df, minimum_time_in=minimum_target_time)

        print("Calculating t_final_core_enter")
        t_final_core_enter = get_t_final_core_enter(df=df, minimum_time_in=minimum_target_time)
        print("Calculating tc")
        tc = get_tc(df=df, t_trigger=t_trigger, t_final_target_enter=t_final_target_enter)
        print("Calculating ca")
        ca = get_ca(t_final_target_enter=t_final_target_enter, t_final_core_enter=t_final_core_enter)

        print("Calculating max_vx and t_max_vx")
        max_vx, t_max_vx = get_t_max_vx(df=df, target_position=trial_data["target_positions"][trial_number],
                                        t_trigger=t_trigger, t_trigger_buffer=0.12)  # todo : corriger

        print("Calculating tr1")
        tr1 = get_tr1(ta=ta, dt=dt)
        print("Calculating tr2")
        tr2 = get_tr2(t_trigger, t_max_vx)

        print("Calculating dist_final and t_dist_final")
        dist_final, t_dist_final = get_cursor_core_distance(df=df, v_max=100, trial_data=trial_data,
                                                            target_core=target_core)

        print("Calculating total_distance_travelled")
        total_distance_travelled = get_total_distance(df=df)

        ts = None
        if not t_dist_final and t_final_target_enter:
            ts = df.iloc[-1]["t"] - t_final_target_enter
        elif t_final_target_enter:
            ts = t_dist_final - t_final_target_enter
        else:
            ts = None

        # y = a * x + b
        print("Calculating linear regression")
        equation_a, pente_droite, r2score = get_linear_regression(df=df, t_start=ta, t_end=ta + 0.1)
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
        if tr1 is not None:
            tr1 = float(tr1)
        if tr2 is not None and tr2 != "centre":
            tr2 = float(tr2)
        if equation_a is not None:
            equation_a = float(equation_a)
        fd.write(
            f"{result_file}, {ta}, {t_trigger}, {dt}, {t_final_target_enter}, {t_final_core_enter}, {tc}, {ca}, {t_max_vx}, {max_vx}, {tr1}, {tr2}, {equation_a}, {pente_droite}, {r2score}, {dist_final}, {ts}, {target_position}, {total_distance_travelled}\n")