import pathlib
import csv
import pandas as pd
import pandas
from pathlib import Path

import re
import subprocess
import sys

import os
import stat

from collections import namedtuple, deque

from myenum import TargetPosition
from utils import compute_trial
import numpy as np
import math

from target_positions_rw import *

Coordinates = namedtuple('Coordinates', ['x', 'y'])


def read_config_target_pos(path, gen_config_file): 
        # Lists to hold the data
        target_pos = []
        
        # Open the CSV file for reading
        with open(gen_config_file, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                target_pos.append(int(row[0]))
        
        # create a new txt file that will write for each line the value of target_pos as (gauche for 0, centre for 1 and droite for 2)
        # the name of the file will be the same as the csv file but with a .txt extension
        new_file = path/"mouse_trajectory_coord-resNEW.txt"
        with open(new_file, 'w') as file:
            for pos in target_pos:
                if pos == 0:
                    file.write("gauche\n")
                elif pos == 1:
                    file.write("centre\n")
                elif pos == 2:
                    file.write("droite\n")
                else:
                    file.write("unknown\n")

        # save the file
        print(f"File {gen_config_file} has been processed")


def explore_directory_for_copy(path):
    for child in pathlib.Path(path).iterdir():
        if str(child.name).endswith("trial_by_trial_config.csv"):
            # Define paths
            absolute_pos_file = path / "absolute_positions.txt"
            new_file = path / "mouse_trajectory_coord-resNEW.txt"

             # Ensure new_file exists (create it if not)
            if not new_file.exists():
                new_file.touch()  # Create an empty file
            
            read_config_target_pos(path, child)
            
            save_config_absolute_positions(absolute_pos_file, new_file)  # Pass file paths, not objects

        if child.is_dir():
            explore_directory_for_copy(child)


def analyse_trial(trial_file: pathlib.Path, trial_data, trial_number):
    try:
        print(f"Analyzing trial: {trial_file}")
        compute_trial(
            result_file=trial_file.with_name(trial_file.stem + "_DF.xls"),
            df=pandas.read_csv(trial_file, delimiter=";"), timestep=0.01, trigger=trial_data["trigger"], trial_data=trial_data, trial_number=trial_number-1, minimum_target_time=0.4) # change to -1 trial_number since we start at 1 
        print(f"Analysis successful for trial: {trial_file}")
    except Exception as e:
        print(f"Analysis failed for trial: {trial_file}")
        print(e)
    return None


def get_trigger(l: str):
    y = None
    for i in l.split():
        if i.isnumeric() and not y:
            y = int(i)
        elif i.isnumeric() and y:
            raise Exception("Bad format to specify trigger: multiple numbers found")
    return y


def get_target_position(l: str):
    numbers = []
    for i in l.split():
        if i.isnumeric():
            numbers.append(int(i))
    if len(numbers) != 8:
        raise Exception(f"Bad format to specify target coordinates: {l}")
    return [
        Coordinates(x=numbers[0], y=numbers[1]),
        Coordinates(x=numbers[2], y=numbers[3]),
        Coordinates(x=numbers[4], y=numbers[5]),
        Coordinates(x=numbers[6], y=numbers[7]),
    ]


def get_single_coord(l: str):
    numbers = []
    for i in l.split():
        if i.isnumeric():
            numbers.append(int(i))
    if len(numbers) != 2:
        raise Exception(f"Bad format to specify target coordinates: {l}")
    return Coordinates(x=numbers[0], y=numbers[1])


def get_trial_data(child: pathlib.Path) -> dict:
    try:
        print(f"Getting trial data from file: {child}")
        trials_description_file = pathlib.Path(child.with_name(re.sub(r"trial\d+.csv", "resNEW.txt", child.name)))
        trials_description_file.is_file()

        trials_description = dict()
        trials_description["target_positions"] = []
        if not trials_description_file.is_file():
            raise Exception(f"file not found: {trials_description_file}")
        if trials_description_file.is_file():
            with open(trials_description_file, "r") as f:
                for l in reversed(list(f)):
                    if l.startswith("gauche"):
                        trials_description["target_positions"].insert(0, TargetPosition.G)
                    elif l.startswith("droite"):
                        trials_description["target_positions"].insert(0, TargetPosition.D)
                    elif l.startswith("centre") and len(l.split()) == 1:
                        trials_description["target_positions"].insert(0, TargetPosition.C)
                    elif l.startswith("Départ"):
                        trials_description["Depart"] = get_single_coord(l)
                    elif l.startswith("cible_gauche"):
                        trials_description["cible_gauche"] = get_target_position(l)
                    elif l.startswith("cible_droite"):
                        trials_description["cible_droite"] = get_target_position(l)
                    elif l.startswith("centre_cible_gauche"):
                        trials_description["centre_cible_gauche"] = get_single_coord(l)
                    elif l.startswith("centre_cible_droite"):
                        trials_description["centre_cible_droite"] = get_single_coord(l)
                    elif l.startswith("cible_centre"):
                        trials_description["cible_centre"] = get_target_position(l)
                    elif l.startswith("centre_cible_centre"):
                        trials_description["centre_cible_centre"] = get_single_coord(l)
                    elif l.startswith("Trigger"):
                        trials_description["trigger"] = get_trigger(l)
                    else:
                        raise Exception(f"Unexpected line: {trials_description_file} : {l} :")
        return trials_description
    except Exception as e:
        print(f"Failed to obtain trial data from file: {child}")
        print(e)
        return None


def explore_directory(path: pathlib.Path):
    trial_file_pattern = r'^.+-trial\d+\.csv$'
    print(f"Exploring directory: {path}")
    for child in path.iterdir():
        if child.is_dir():
            explore_directory(child)
        elif re.match(trial_file_pattern, child.name):
            trial_data = None
            trial_number = re.search(r"trial(\d+)", child.stem)
            trial_number = int(trial_number.group(1))
            try:
                trial_data = get_trial_data(child)
                if trial_data:
                    analyse_trial(trial_file=child, trial_data=trial_data, trial_number=trial_number)
            except Exception as e:
                print(e)


def group_indexes_by_second_parent(paths):
    # Initialize a dictionary to hold the second parent folder names and their associated indexes
    folder_indexes = {}
    
    for i, path in enumerate(paths):
        # Extract the second parent folder name from the path
        second_parent_folder = path.split('/')[1]
        
        # If the folder name is not in the dictionary, initialize it with the current index in a list
        if second_parent_folder not in folder_indexes:
            folder_indexes[second_parent_folder] = [i]
        else:
            # If the folder name is already in the dictionary, append the current index to its list
            folder_indexes[second_parent_folder].append(i)
            
    return folder_indexes


def match_rows_df(files_paths, breaks_dict):

    num_breaks_list = []
    break_timing_list = []
    break_position_list = []

    cwd = os.getcwd()
    for file_path in files_paths:
        
        output_dir = os.path.dirname(file_path)
        match = re.search(r'^(.*?)(trial(\d+)_DF)', file_path)
        if match:
            number = int(match.group(3))
            # NOTE: TODO if statement pour legacy data
            if format_version == "legacy":
                beginning_file_path = match.group(1)
        
                new_file_path = beginning_file_path + "trial" + str(number) + ".csv"
                full_file_path = os.path.join(cwd, new_file_path)
        
        if not (format_version == "legacy"):
            # NOTE: TODO Handle For New File Format (resNEW.txt)   
            new_file_path = f"mouse_trajectory_coord_{number:03}.csv"
            traject_dir = "Trajectories"
            full_file_path = os.path.join(cwd, output_dir, traject_dir, new_file_path)

        num_breaks, break_timings, break_positions = breaks_dict[Path(full_file_path)] 
        num_breaks_list.append(num_breaks)
        break_timing_list.append(break_timings)
        break_position_list.append(break_positions)

    return num_breaks_list, break_timing_list, break_position_list


def modify_resume_resultats(data_path, vmin, angle_threshold, time_interval, time_thresh, spatial_thresh, angle_window):
    # Load the CSV file
    df = pd.read_csv('resume_resultats.csv', sep=',') 

    # Define a function to convert 'Echec' and None to NaN for numerical computation
    def to_numeric(value):
        if value in ['Echec', 'None', None]:
            return float('nan')
        return float(value)

    # Apply the conversion to numeric for relevant columns
    for column in [' tc', ' total_distance_travelled', ' ts', ' dist_final']:
        df[column] = df[column].apply(to_numeric)

    # Extract the result_file
    files_paths = df["result_file"]

    # Compute the new columns
    df["SA"] = df['trigger_to_target_time'] * df['trigger_to_target_distance']
    df['precision'] = df['target_to_stop_time'] * df['target_to_stop_distance']

    folder_indexes = group_indexes_by_second_parent(files_paths)
    # construct a list with the SA_std depending on the indices for trials from the same participant
    SA_std_list = np.zeros((df.shape[0])) 
    for key in folder_indexes:
        indices = folder_indexes[key]
        SAs_std = df[" SA"][indices].std()
        SA_std_list[indices] = SAs_std

    df['SA_variability'] = SA_std_list

    # Drop unnecessary columns if they exist
    columns_to_drop = [
        'equation_a', 'ca', 't_final_target_enter', 't_trigger', 't_max_vx',
        'r2score', 'ta', 'dt', 't_final_core_enter'
    ]
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)


    # remove previous summary csv file
    #cwd = os.getcwd()
    #summary_csv_file = 'resume_resultats.csv'
    #summary_csv_file_path = os.path.join(cwd, summary_csv_file)
    #os.remove(summary_csv_file_path) 
    #print(f"File {summary_csv_file_path} has been deleted.")

    # add information about breaks 
    breaks_dict = explore_directory_for_trajectories(data_path, vmin, angle_threshold, time_interval, time_thresh, spatial_thresh, angle_window)
    
    # function returns the elements of the dictionary in the order of the rows in the dataframe
    num_breaks_arr, break_timing_arr, break_position_arr = match_rows_df(files_paths, breaks_dict)
    df[" num_breaks"] = num_breaks_arr 
    df[" break_timings"] = break_timing_arr
    df[" break_positions"] = break_position_arr

    # Save the updated DataFrame to a new CSV file
    df.to_csv('updated_resume_resultats.csv', index=False)
    print("Updated csv has been saved")


def explore_directory_for_trajectories(data_path, vmin, angle_threshold, time_interval, time_thresh, spatial_thresh, angle_window):
    """
    Returns a dict with key the subdirectory of the trajectory and value 
    """
    breaks_dict = {}
    for child in pathlib.Path(data_path).iterdir():
        print("****** Exploring directory in explore_directory_for_trajectories:")
        print(data_path)
        print(child)
        
        # condition to check whether the child file endswith -trial#.csv
        if format_version == "legacy":
            cond = re.search(r'-trial[0-9]+\.csv$', str(child))
        else:
            cond = (str(child.name).startswith("mouse_trajectory_coord_") and not str(child.name).endswith("_DF.xls"))
        if cond:
            num_breaks, break_timing, break_position = trajectory_breaks_processing(data_path, vmin, angle_threshold, time_interval, str(child), time_thresh, spatial_thresh, angle_window)
            breaks_dict[child] = num_breaks, break_timing, break_position        
        elif child.is_dir():
            sub_breaks_dict = explore_directory_for_trajectories(child, vmin, angle_threshold, time_interval, time_thresh, spatial_thresh, angle_window)
            breaks_dict.update(sub_breaks_dict) 
    return breaks_dict 


def trajectory_breaks_processing(output_trajectory_path, vmin, angle_threshold, time_interval, path_trajectory, time_thresh=0.1, spatial_thresh=30, angle_window=2):
    """
    Main function to read .csv file and detect trajectory breaks.
    """
    try:
        data = pd.read_csv(path_trajectory, sep=';')
        data['Time'] = generate_timestamps(data, time_interval)
        breaks = detect_trajectory_breaks(data, vmin, angle_threshold, time_thresh, spatial_thresh, output_trajectory_path, angle_window)
        break_timing = []
        break_position = []

        if breaks:
            print("Trajectory breaks detected:")
            for t, angle_change, velocity, x, y in breaks:
                print(f"Time: {t}, Average Angle: {angle_change}, Average Velocity: {velocity}, Average Position: ({x}, {y})")
                break_timing.append(float(t))
                break_position.append((float(x), float(y)))
        else:
            print("No trajectory breaks detected.")
                
        return len(breaks), break_timing, break_position
    
    except FileNotFoundError:
        print(f"File '{path_trajectory}' not found.")
    except Exception as e:
        print("An error occurred:", str(e))


def calculate_angle(v1, v2):
    """
    Calculate the angle change between two vectors.
    """
    # Dot product of v1 and v2
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    
    # Magnitude of vector v1 and v2
    mag_v1 = math.sqrt(v1[0]**2 + v1[0]**2)
    mag_v2 = math.sqrt(v2[1]**2 + v2[1]**2)
    
    # Cosine of the angle between v1 and v2
    if mag_v1 == 0 or mag_v2 == 0:
        return 0  # Degenerate case where one of the vectors has no length
    cos_theta = dot_product / (mag_v1 * mag_v2)
    
    # Ensure the cosine value does not exceed the range of -1 to 1 due to floating-point precision issues
    cos_theta = max(-1, min(1, cos_theta))
    
    # Angle in degrees
    angle = math.degrees(math.acos(cos_theta))
    
    return angle


def is_mouse_in_target(output_trajectory_path, x, y):
    try:
        # getting the parent directory of the subdir (Trajectories)
        output_subdir_path = os.path.dirname(output_trajectory_path)
        # param: path to the output subfolder
        absolute_pos_file = os.path.join(output_subdir_path, "absolute_positions.txt") 
        
        # NOTE: TODO: In reality add a flag to check whether the data is in legacy or new format
        # Find the file path in output_subdir_path that ends with -resNEW.txt (legacy format)
        if format_version == "legacy":
            for file in pathlib.Path(output_trajectory_path).iterdir():
                if file.name.endswith("-resNEW.txt"):
                    legacy_positions_path = file
                    break
            save_config_relative_positions(legacy_positions_path, absolute_pos_file)
        
        starting_pos, target_pos_boundaries, target_centers, trigger_pos = read_config_absolute_positions(open(absolute_pos_file, 'r'))

        in_left_target = (target_pos_boundaries[0][0][0] < x and target_pos_boundaries[0][1][0] > x) and (target_pos_boundaries[0][0][1] < y and target_pos_boundaries[0][2][1] > y)
        in_center_target = (target_pos_boundaries[1][0][0] < x and target_pos_boundaries[1][1][0] > x) and (target_pos_boundaries[1][0][1] < y and target_pos_boundaries[1][2][1] > y)
        in_right_target = (target_pos_boundaries[2][0][0] < x and target_pos_boundaries[2][1][0] > x) and (target_pos_boundaries[2][0][1] < y and target_pos_boundaries[2][2][1] > y)
        
        return in_left_target or in_center_target or in_right_target
    except FileNotFoundError:
        print(f"File '{output_subdir_path}' not found for is_mouse_in_target.")
    except Exception as e:
        print("An error occurred:", str(e)) 


def generate_timestamps(data, time_interval):
    """
    Generate timestamps based on a constant time interval.
    """
    num_samples = len(data)
    # Only letting 3 decimal places for num_samples * time_interval
    max_time = round((num_samples * time_interval)/time_interval, 1)
    timestamps = time_interval*np.arange(0, max_time, 1)

    return timestamps


def detect_trajectory_breaks(data, vmin, angle_threshold, time_threshold, spatial_threshold, output_trajectory_path, angle_window):
    """
    Detect trajectory breaks based on velocity and angle change threshold,
    and average multiple breaks within a specified time window.
    """
    breaks = []
    # stores the position and time points of the 2*angle_window last timesteps
    # (x, y, t)
    prev_values = deque(maxlen=2*angle_window + 1) 
    for _ in range(2*angle_window):
        prev_values.append((None, None, None))
    velocity = None

    for index, row in data.iterrows():

        prev_values.append((row['X'], row['Y'], row['Time']))

        if prev_values[0][0] is not None and prev_values[0][1] is not None:
            v1 = np.array([prev_values[angle_window][0] - prev_values[0][0], prev_values[angle_window][1] - prev_values[0][1]])
            v2 = np.array([prev_values[-1][0] - prev_values[angle_window][0], prev_values[-1][1] - prev_values[angle_window][1]])
            distance = np.sqrt(v1[0]**2 + v1[1]**2) + np.sqrt(v2[0]**2 + v2[1]**2)
            time_diff = prev_values[-1][2] - prev_values[0][2]
            velocity = distance / time_diff
            
            # computing spatial threshold
            if breaks:
                dx = breaks[-1][3] - prev_values[-2][0]
                dy = breaks[-1][4] - prev_values[-2][1]
                dist_prev_break = np.sqrt(dx**2 + dy**2)
            
            if velocity > vmin and not is_mouse_in_target(output_trajectory_path, prev_values[-1][0], prev_values[-1][1]):
                angle = calculate_angle(v1, v2)
                if angle > angle_threshold:
                    if breaks and breaks[-1][0] + time_threshold < prev_values[-1][2] and dist_prev_break > spatial_threshold:
                       breaks.append((prev_values[-1][2], angle, velocity, prev_values[angle_window][0], prev_values[angle_window][1]))
                    elif not breaks: 
                       breaks.append((prev_values[-1][2], angle, velocity, prev_values[angle_window][0], prev_values[angle_window][1]))
            else:
                # consider the case when the mouse leaves the screen basically when the y is negative or outisde of the x frame
                if prev_values[-1][0] < 0 or prev_values[-1][1] < 0: # Add the right x limit (based on the sarting point x coordinate * 2) TODO (maybe not necessary if using a single screen)
                    if breaks and breaks[-1][0] + time_threshold < prev_values[-1][2] and dist_prev_break > spatial_threshold:
                        breaks.append((prev_values[-1][2], angle, velocity, prev_values[angle_window][0], prev_values[angle_window][1]))
        prev_values.popleft()
        
    return breaks


if __name__ == "__main__":
    """
    Ce script parcours les dossiers et sous dossiers du dossier donné en entrée 
    Pour chaque fichier finissant par "res.txt"
    Il nettoie un peu le fichier
    """
    if len(sys.argv) != 3:
        print("This script except only two command line argument : the path to the directory containing the file to process\n and whether the data is in 'legacy' format or 'new' format")
    else:
        data_path = pathlib.Path(sys.argv[1])
        format_version = str(sys.argv[2])
        # vmin = float(input("Enter the velocity threshold (vmin): "))
        # angle_threshold = float(input("Enter the angle change threshold (in degrees): "))
        # time_interval = float(input("Enter the time interval between samples (in seconds): "))
        # time_thresh = float(input("Enter the minimum time between breaks (in seconds): "))
        # spatial_thresh = float(input("Enter the minimum distance between breaks (in screen units): "))
        # angle_window = int(input("Enter the window for computing the angle (number of time intervals): "))
        
        vmin = 10
        angle_threshold = 60
        time_interval = 0.01
        time_thresh = 0.1
        spatial_thresh = 10
        angle_window = 2
        
        explore_directory_for_copy(data_path)

        # Rename and move the trajectory coordinate files to the subject's folder
        cwd = os.getcwd()
        rename_script = 'rename_csv_total.sh'
        # Append the filename to the cwd to get the full path
        rename_script_path = os.path.join(cwd, rename_script)
        full_data_path = os.path.join(cwd, data_path)
        # Read the current permissions
        current_permissions = os.stat(rename_script_path).st_mode
        # Add the executable bit for the owner, group, and others
        new_permissions = current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        # Change the mode of the file to make it executable
        os.chmod(rename_script_path, new_permissions)
        # Call the shell script
        subprocess.run([rename_script_path])

        with open('resume_resultats.csv', 'w') as fd:
            fd.write("result_file, RT, t_trigger, RtTrig, t_trigger_computed, distance_to_trigger, "
                     "target_enters, t_first_target_enter, trigger_to_target_time, trigger_to_target_distance, "
                     "target_to_stop_time, target_to_stop_distance, total_movement_time, total_distance_travelled, "
                     "total_trial_time, finale_distance_to_center, finale_distance_to_center_time, "
                     "max_vx, t_max_vx, TtA, equation_a, pente_droite, r2score, target_position\n")
        explore_directory(data_path)

        # ajoute les nouvelles variables SA et precision
        modify_resume_resultats(full_data_path, vmin, angle_threshold, time_interval, time_thresh, spatial_thresh, angle_window)


