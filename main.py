import pathlib
import csv
import pandas as pd
import pandas
from pathlib import Path
from PIL import Image, ImageDraw

import re
import subprocess
import sys
import traceback

import os
import stat

from collections import namedtuple, deque

from myenum import TargetPosition
from utils import compute_trial, get_target_enters
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


def analyse_trial(trial_file: pathlib.Path, trial_data, trial_number, trial_feedback=None):
    try:
        print(f"Analyzing trial: {trial_file}")

        # Étape 1 : lire le config AVANT compute_trial
        parent_folder = trial_file.parent
        config_file = parent_folder / "trial_by_trial_config.csv"
        config_df = pd.read_csv(config_file, header=None)
        config_df = pd.read_csv(config_file, header=None)
        print(f"Config DataFrame shape: {config_df.shape}")
        trial_feedback = config_df.iloc[trial_number-1, 1]
        trial_feedback = trial_feedback == 1
        
        
        # Ensuite seulement : appel de compute_trial
        compute_trial(
            result_file=trial_file.with_name(trial_file.stem + "_DF.xls"),
            df=pd.read_csv(trial_file, delimiter=";"), 
            time_step=0.01, 
            trigger=trial_data["trigger"], 
            trial_data=trial_data, 
            trial_number=trial_number - 1,
            min_target_time=0.01,
            trial_feedback=trial_feedback
        )

        if trial_feedback:
            print(f"Trial {trial_number} has feedback enabled.")
        else:
            print(f"Trial {trial_number} has feedback disabled.")
        
        print(f"Analysis successful for trial: {trial_file}")

    except Exception as e:
        print(f"Analysis failed for trial: {trial_file}")
        print(e)

    return None

"""
def analyse_trial(trial_file: pathlib.Path, trial_data, trial_number):
    try:
        print(f"Analyzing trial: {trial_file}")
        compute_trial(
            result_file=trial_file.with_name(trial_file.stem + "_DF.xls"),
            df=pandas.read_csv(trial_file, delimiter=";"), time_step=0.01, trigger=trial_data["trigger"], trial_data=trial_data, trial_number=trial_number-1, min_target_time=0.01) # change to -1 trial_number since we start at 1 
        print(f"Analysis successful for trial: {trial_file}")


    except Exception as e:
        print(f"Analysis failed for trial: {trial_file}")
        print(e)
    return None
"""

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
                    print(f"Filename: {child.name}, extracted trial_number: {trial_number}")

            except Exception as e:
                print(e)


def group_indexes_by_second_parent(paths):
    # Initialize a dictionary to hold the second parent folder names and their associated indexes
    folder_indexes = {}
    
    for i, path in enumerate(paths):
        #debug
        if not isinstance(path, str):
            print(f"[{i}] Problème : type = {type(path)}, valeur = {path}")
            #fin debug
        # Extract the second parent folder name from the path
        second_parent_folder = path.split('\\')[1]
        
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
    print("Loaded DataFrame from resume_resultats.csv:")
    print(df)  # Debug: Print the first few rows 

    # Strip leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Define a function to convert 'Echec' and None to NaN for numerical computation
    def to_numeric(value):
        if value in ['Echec', 'None', None]:
            return float('nan')
        return float(value)

    # Apply the conversion to numeric for relevant columns
    for column in ['tc', 'total_distance_travelled', 'ts', 'dist_final']:
        if column in df.columns:
            df[column] = df[column].apply(to_numeric)
        else:
            print(f"Warning: Column '{column}' not found in DataFrame.")

    # Ensure relevant columns are numeric
    for column in ['trigger_to_target_time', 'trigger_to_target_distance', 'target_to_stop_time', 'target_to_stop_distance', 'total_movement_time', 'total_trial_time', 'finale_distance_to_center', 'finale_distance_to_center_time', 'total_distance_travelled']:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        else:
            print(f"Warning: Column '{column}' not found in DataFrame.")

    # Debug: Print DataFrame after processing
    print("Processed DataFrame:")
    print(df.head())

    # Extract the result_file
    files_paths = df['result_file']
    print(df.head())
    print(df.dtypes)


    # Compute the new columns
    df['SA'] = df['trigger_to_target_time'] * df['trigger_to_target_distance']
    df['precision'] = df['target_to_stop_time'] * df['target_to_stop_distance']

    folder_indexes = group_indexes_by_second_parent(files_paths)
    # construct a list with the SA_std depending on the indices for trials from the same participant
    SA_std_list = np.zeros((df.shape[0])) 
    for key in folder_indexes:
        indices = folder_indexes[key]
        SAs_std = df["SA"][indices].std()
        SA_std_list[indices] = SAs_std

    df['SA_variability'] = SA_std_list

    # Drop unnecessary columns if they exist
    columns_to_drop = [
        'equation_a', 'ca', 't_final_target_enter',
        'r2score', 'ta', 'dt', 't_final_core_enter'
    ]
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    # add information about breaks BEFORE removing the CSV file
    trigger_time_df = df[["result_file", "t_trigger"]] # to exclude the breaks before the trigger is passed
    target_entry_df = df[["result_file", "t_first_target_enter"]] # to stop breaks after target entry
    breaks_dict = explore_directory_for_trajectories(data_path, vmin, angle_threshold, time_interval, time_thresh, spatial_thresh, angle_window, trigger_time_df, target_entry_df)
    
    #remove previous summary csv file
    cwd = os.getcwd()
    summary_csv_file = 'resume_resultats.csv'
    summary_csv_file_path = os.path.join(cwd, summary_csv_file)
    os.remove(summary_csv_file_path) 
    print(f"File {summary_csv_file_path} has been deleted.")
    
    # function returns the elements of the dictionary in the order of the rows in the dataframe
    num_breaks_arr, break_timing_arr, break_position_arr = match_rows_df(files_paths, breaks_dict)
    df['num_breaks'] = num_breaks_arr 
    df['break_timings'] = break_timing_arr
    df['break_positions'] = break_position_arr

    # Save the updated DataFrame to a new CSV file
    updated_csv_path = os.path.join(data_path, 'updated_resume_resultats.csv')
    df.to_csv(updated_csv_path, index=False)
    updated_csv_path = os.path.join(data_path, 'updated_resume_resultats.csv')
    df.to_csv(updated_csv_path, index=False)
    print("Updated csv has been saved")


def explore_directory_for_trajectories(data_path, vmin, angle_threshold, time_interval, time_thresh, spatial_thresh, angle_window, trigger_time_df, target_entry_df):
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
            num_breaks, break_timing, break_position = trajectory_breaks_processing(data_path, vmin, angle_threshold, time_interval, str(child), format_version, time_thresh, spatial_thresh, angle_window, trigger_time_df, target_entry_df)
            breaks_dict[child] = num_breaks, break_timing, break_position        
        elif child.is_dir():
            sub_breaks_dict = explore_directory_for_trajectories(child, vmin, angle_threshold, time_interval, time_thresh, spatial_thresh, angle_window, trigger_time_df, target_entry_df)
            breaks_dict.update(sub_breaks_dict) 
    return breaks_dict 


def transform_coord2image_fname(s):
    match = re.match(r"mouse_trajectory_coord_(\d+)", s)
    if match:
        number = int(match.group(1))
        return f"mouse_trajectory_{number:03d}.png"
    return s  # return original if pattern doesn't match


def convert_to_DF_fname(filename):
    # Extract the 3-digit number using regex
    match = re.search(r'_(\d{3})\.csv$', filename)
    if not match:
        raise ValueError("Filename must end with a 3-digit number followed by .csv")
    
    num_str = match.group(1)
    num_int = int(num_str)  # Remove leading zeros

    # Create new filename
    new_filename = f"mouse_trajectory_coord-trial{num_int}_DF.xls"
    return new_filename



def trajectory_breaks_processing(output_trajectory_path, vmin, angle_threshold, time_interval, path_trajectory, format_version, time_thresh, spatial_thresh, angle_window, trigger_time_df=None, target_entry_df=None):
    """
    Main function to read .csv file and detect trajectory breaks.
    """
    try:        
        # get only the filename of path trajectory 
        path_trajectory = pathlib.Path(path_trajectory)
        trajectory_fname = path_trajectory.name
        result_key_fname = convert_to_DF_fname(str(trajectory_fname))
        result_key_path = pathlib.Path(output_trajectory_path).parent / result_key_fname
        
        time_2_trigger = trigger_time_df.loc[trigger_time_df['result_file'] == str(result_key_path), 't_trigger'].values[0]
        
        first_target_entry_time = None
        try:
            # Get target entry time from the passed dataframe
            if target_entry_df is not None:
                result_row = target_entry_df.loc[target_entry_df['result_file'] == str(result_key_path)]
                if not result_row.empty:
                    first_target_entry_time = result_row['t_first_target_enter'].values[0]
                    if pd.isna(first_target_entry_time) or first_target_entry_time == 'Echec' or str(first_target_entry_time).lower() == 'none':
                        first_target_entry_time = None
                    else:
                        first_target_entry_time = float(first_target_entry_time)
                        print(f"Found first target entry time: {first_target_entry_time:.3f}")
        except Exception as e:
            print(f"Could not read target entry time for {result_key_fname}: {e}")
            first_target_entry_time = None


        data = pd.read_csv(path_trajectory, sep=';')
        print(f"Data loaded, shape: {data.shape}")
        data['Time'] = generate_timestamps(data, time_interval)
        
        breaks = detect_trajectory_breaks(data, vmin, angle_threshold, time_thresh, spatial_thresh, output_trajectory_path, angle_window, time_2_trigger, accel_threshold, first_target_entry_time) 
        print(f"Breaks detection completed, found {len(breaks)} breaks")
        
        break_timing = []
        break_position = []

        print(f"Format version: {format_version}, PNG processing starting...")

        if breaks:
            print("Trajectory breaks detected:")
            for t, angle_change, velocity, x, y, acceleration in breaks:
                print(f" BREAK Time: {t}, Average Angle: {angle_change}, Average Velocity: {velocity}, Average Position: ({x}, {y}), Acceleration: {acceleration}")
                break_timing.append(float(t))
                break_position.append((float(x), float(y)))

            # Save png with breaks visualization
            if format_version == "new":
                fname_coord = pathlib.Path(path_trajectory).name
                trajectory_dir = pathlib.Path(path_trajectory).parent
                fname_png = transform_coord2image_fname(fname_coord)
                trajectory_png_path = trajectory_dir / fname_png
                image = Image.open(trajectory_png_path)

                draw = ImageDraw.Draw(image)
                for x, y in break_position:
                    r = 5
                    draw.ellipse((x - r, y - r, x + r, y + r), fill="red", outline="red")
                
                marked_fname_png = f"marked_{fname_png}"
                new_image_path = os.path.join(trajectory_dir, marked_fname_png)
                
                image.save(new_image_path)
                print(f"Saved marked image to: {new_image_path}")

            # if format_version == "legacy": # TODO 
            #     raise ImplementationError("Legacy format not implemented yet")
            #     #pathlib.Path(output_path_trajectory) / "breaks.png"

            # Save png with breaks visualization
            if format_version == "new":
                fname_coord = pathlib.Path(path_trajectory).name
                trajectory_dir = pathlib.Path(path_trajectory).parent
                fname_png = transform_coord2image_fname(fname_coord)
                trajectory_png_path = trajectory_dir / fname_png
                image = Image.open(trajectory_png_path)

                draw = ImageDraw.Draw(image)
                for x, y in break_position:
                    r = 5
                    draw.ellipse((x - r, y - r, x + r, y + r), fill="red", outline="red")
                
                marked_fname_png = f"marked_{fname_png}"
                new_image_path = os.path.join(trajectory_dir, marked_fname_png)
                
                image.save(new_image_path)
                print(f"Saved marked image to: {new_image_path}")

            # if format_version == "legacy": # TODO 
            #     raise ImplementationError("Legacy format not implemented yet")
            #     #pathlib.Path(output_path_trajectory) / "breaks.png"
        else:
            print("No trajectory breaks detected.")
            if format_version == "new": # added marked png even when no breaks found
                fname_coord = pathlib.Path(path_trajectory).name
                trajectory_dir = pathlib.Path(path_trajectory).parent
                fname_png = transform_coord2image_fname(fname_coord)
                trajectory_png_path = trajectory_dir / fname_png
                image = Image.open(trajectory_png_path)

                marked_fname_png = f"marked_{fname_png}"
                new_image_path = os.path.join(trajectory_dir, marked_fname_png)
                
                image.save(new_image_path)
                
        return len(breaks), break_timing, break_position
    
    except FileNotFoundError:
        print(f"File '{path_trajectory}' not found.")
        return 0, [], []
    except Exception as e:
        print("An error occurred:", str(e))
        import traceback
        traceback.print_exc()
        return 0, [], []


def calculate_angle(v1, v2):
    """
    Calculate the angle change between two vectors.
    """
    # Dot product of v1 and v2
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    
    # Magnitude of vector v1 and v2
    mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    # Cosine of the angle between v1 and v2
    if mag_v1 == 0 or mag_v2 == 0:
        return 0  # Degenerate case where one of the vectors has no length
    cos_theta = dot_product / (mag_v1 * mag_v2)
    
    # Ensure the cosine value does not exceed the range of -1 to 1 due to floating-point precision issues
    cos_theta = max(-1, min(1, cos_theta))
    
    # Angle in degrees
    angle = math.degrees(math.acos(cos_theta))
    
    return angle

"""
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

        # Read target radius from the absolute_positions.txt file
        target_radius = None
        with open(absolute_pos_file, 'r') as f:
            for line in f:
                if line.startswith("Target radius:"):
                    target_radius = float(line.split(":")[1].strip())
                    break
        
        if target_radius is None:
            #print("Warning: Could not find target radius, using default value of 40")
            target_radius = 40.0  # Default fallback
        
        # Check if mouse is within circular targets using distance from center
        def distance_to_center(center_x, center_y):
            return math.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Check each target center
        for center_x, center_y in target_centers:
            if distance_to_center(center_x, center_y) <= target_radius:
                return True
        
        return False
        
    except FileNotFoundError:
        print(f"File '{output_subdir_path}' not found for is_mouse_in_target. Assuming mouse is NOT in target.")
        return False  # Assume not in target if file not found
    except Exception as e:
        print(f"Error in is_mouse_in_target: {str(e)}. Assuming mouse is NOT in target.")
        return False  # Assume not in target if error occurs 
"""

def generate_timestamps(data, time_interval):
    """
    Generate timestamps based on a constant time interval.
    """
    num_samples = len(data)
    # Only letting 3 decimal places for num_samples * time_interval
    max_time = round((num_samples * time_interval)/time_interval, 1)
    timestamps = time_interval*np.arange(0, max_time, 1)

    return timestamps


def detect_trajectory_breaks(data, vmin, angle_threshold, time_threshold, spatial_threshold, output_trajectory_path, angle_window, time_2_trigger, accel_threshold, first_target_entry_time=None): 
    """
todo : do not count breaks at the end of trajectory when no FB. Do not count 1st break when mouse touches the border of the screen
    """

    breaks = []
    # stores the position and time points of the 2*angle_window last timesteps
    # (x, y, t)
    extended_history = deque(maxlen=25)  # Pour fenêtre étendue
    prev_values = deque(maxlen=2*angle_window + 1) 
    for _ in range(2*angle_window):
        prev_values.append((None, None, None))
    velocity = None
    acceleration = None

    # Get the final trajectory point coordinates
    final_x = data.iloc[-1]['X']
    final_y = data.iloc[-1]['Y']
    print(f"Final trajectory coordinates: ({final_x}, {final_y})")

    if first_target_entry_time is not None:
        print(f"First target entry detected at time {first_target_entry_time:.3f}. Break detection will stop after this point.")
    else:
        print("No target entries detected. Break detection will run for the entire trajectory.")


    # Initialize a counter for low velocity duration
    low_velocity_duration = 0
    no_velocity_duration = 0

    for index, row in data.iterrows():
        # Stop break detection after first target entry
        if first_target_entry_time is not None and row['Time'] >= first_target_entry_time:
            print(f"Stopping break detection at time {row['Time']:.3f} - first target entry reached at {first_target_entry_time:.3f}")
            break

        prev_values.append((row['X'], row['Y'], row['Time']))
        extended_history.append((row['X'], row['Y'], row['Time']))

        # TODO of ignoring the break inside of the targets 
        if prev_values[0][0] is not None and prev_values[0][1] is not None:
            v1 = np.array([prev_values[angle_window][0] - prev_values[0][0], prev_values[angle_window][1] - prev_values[0][1]])
            v2 = np.array([prev_values[-1][0] - prev_values[angle_window][0], prev_values[-1][1] - prev_values[angle_window][1]])

            distance = np.sqrt(v1[0]**2 + v1[1]**2) + np.sqrt(v2[0]**2 + v2[1]**2)
            time_diff = prev_values[-1][2] - prev_values[0][2]
            velocity = distance / time_diff

            #Compute instantaneous velocity
            dx = prev_values[angle_window][0] - prev_values[angle_window - 1][0]
            dy = prev_values[angle_window][1] - prev_values[angle_window - 1][1]
            dt = prev_values[angle_window][2] - prev_values[angle_window - 1][2]
            velocity_instant = np.sqrt(dx**2 + dy**2) / dt if dt > 0 else 0

            # Debug: Print instantaneous velocity
            # print(f"Time {row['Time']:.3f}: Instantaneous velocity = {velocity:.2f}")


             # Check if conditions are met for break detection
            is_fast_enough = velocity > vmin
            is_after_trigger = row['Time'] > time_2_trigger + 0.1 # add 100ms buffer after trigger
            
            # Calculate angle once for both debugging and break detection
            angle = calculate_angle(v1, v2)
            
            
            #new acceleration criteria
            # Calculate acceleration
            if len(prev_values) >= 3:
                dt = prev_values[-1][2] - prev_values[-3][2]
                dv = velocity - (np.sqrt((prev_values[-3][0] - prev_values[-2][0])**2 + (prev_values[-3][1] - prev_values[-2][1])**2) / dt if dt > 0 else 0)
                acceleration = dv / dt if dt > 0 else 0
            else:
                acceleration = 0

            # Check if conditions are met for break detection
            is_large_acceleration = accel_threshold is not None and abs(acceleration) >= accel_threshold

                       
            def append_break(breaks, prev_values, angle, velocity, acceleration, angle_window):
                """Append a break or update an existing one if it's within thresholds."""
                break_x = prev_values[angle_window][0]
                break_y = prev_values[angle_window][1]
                break_time = prev_values[angle_window][2]

                # Check if break is within 5 pixels of the final trajectory coordinates - if so, ignore it
                distance_to_final = np.sqrt((break_x - final_x)**2 + (break_y - final_y)**2)
                if distance_to_final <= 30:
                    print(f"Ignoring break within 5px of final trajectory coordinates ({final_x}, {final_y}) at time {break_time:.3f}")
                    return

                # Check if break is at screen borders - if so, ignore it
                screen_limits = {"left": 0, "right": 1680, "top": 0, "bottom": 1050}
                border_buffer = 2
                is_at_border = (break_x <= (screen_limits["left"] + border_buffer) or
                                break_x >= (screen_limits["right"] - border_buffer) or
                                break_y <= (screen_limits["top"] + border_buffer) or
                                break_y >= (screen_limits["bottom"] - border_buffer))
                if is_at_border:
                    print(f"Ignoring break at screen border ({break_x}, {break_y}) at time {break_time:.3f}")
                    return

                # Check if there is an existing break within spatial and time thresholds
                for i, (t, a, v, x, y, acc) in enumerate(breaks):
                    if abs(break_time - t) < time_threshold or np.sqrt((break_x - x)**2 + (break_y - y)**2) < spatial_threshold:
                        # Update the break if the new one has a larger angle
                        if angle > a:
                            breaks[i] = (break_time, angle, velocity, break_x, break_y, acceleration)
                        return

                # If no nearby break exists, append the new break
                breaks.append((break_time, angle, velocity, break_x, break_y, acceleration))

            # Check if mouse is off screen first
            screen_limits = {"left": 0, "right": 1680, "top": 0, "bottom": 1050}
            mouse_off_screen = (prev_values[-1][0] <= screen_limits["left"] or prev_values[-1][0] >= screen_limits["right"] or 
                              prev_values[-1][1] <= screen_limits["top"] or prev_values[-1][1] >= screen_limits["bottom"])
            
            # if mouse_off_screen:
                # Only check for low velocity breaks when mouse is off screen (regardless of trigger)
                # if velocity_instant <= 100:
                    # low_velocity_duration += time_interval
                    # if low_velocity_duration >= 0.2:
                        # append_break(breaks, prev_values, angle, velocity, acceleration, angle_window)
                        # print(f" BREAK detected due to low velocity (off-screen) at time {prev_values[-1][2]:.3f} (velocity={velocity:.2f}px/s)")
                        # low_velocity_duration = 0  # Reset after detecting a break
                # else:
                    # low_velocity_duration = 0
            if is_after_trigger and not mouse_off_screen:
                # Normal break detection logic when on-screen and after trigger
                #print(f"Time {row['Time']:.3f}: velocity={velocity:.2f}, angle={angle:.2f}, acceleration={acceleration:.2f}, after_trigger={is_after_trigger}")
                #this first if is a try
                if (velocity >= 1000 and angle >= 80):
                    append_break(breaks, prev_values, angle, velocity, acceleration, angle_window)
                    print(f" Fast BREAK detected at time {prev_values[angle_window][2]:.3f} (velocity={velocity:.2f}px/s, angle={angle:.2f}°, acceleration={acceleration:.2f}px/s²)")
                elif (is_fast_enough and velocity < 1000 and angle >= angle_threshold and is_large_acceleration) or angle > 135:
                    append_break(breaks, prev_values, angle, velocity, acceleration, angle_window)
                    print(f" BREAK detected at time {prev_values[angle_window][2]:.3f} (velocity={velocity:.2f}px/s, angle={angle:.2f}°, acceleration={acceleration:.2f}px/s²)")

                #elif np.allclose(v1, [0, 0], atol=1e-6):  # Détection d'arrêt via v1, it was already very good
                elif velocity_instant == 0:
                    no_velocity_duration += time_interval
                    if no_velocity_duration >= 0.5:
                        append_break(breaks, prev_values, angle, velocity, acceleration, angle_window)
                        print(f" BREAK detected due to stop (velocity=0) at time {prev_values[angle_window][2]:.3f} (velocity={velocity:.2f}px/s)")
                        no_velocity_duration = 0  # Reset after detecting a break
                    elif no_velocity_duration >= 0.02: #TODO remove detection just before the target?
                        # Look ahead to find moving points after the stop

                        pause_time = prev_values[angle_window][2]  # Time of the stationary point
                        pause_index = None
                        for i, (x, y, t) in enumerate(extended_history):
                            if abs(t - pause_time) < 0.005:  # Small tolerance for floating point comparison
                                pause_index = i
                                break

                        # Find moving points before the stop (from extended_history)
                        moving_points_before = []
                        for i in range(pause_index - 1, -1, -1):
                            if len(moving_points_before) >= angle_window:
                                break
                            if i > 0:
                                dx_check = extended_history[i][0] - extended_history[i-1][0]
                                dy_check = extended_history[i][1] - extended_history[i-1][1]
                                dt_check = extended_history[i][2] - extended_history[i-1][2]
                                vel_check = np.sqrt(dx_check**2 + dy_check**2) / dt_check if dt_check > 0 else 0
                                
                                if vel_check > 0:  # Only include moving points
                                    moving_points_before.insert(0, extended_history[i])  # Insert at beginning to maintain order
                        
                        # Look ahead in the data to find moving points after the stop
                        moving_points_after = []

                        for i in range(pause_index + 1, len(extended_history)):
                            if len(moving_points_after) >= angle_window:
                                break
                            if i > 0:
                                dx_check = extended_history[i][0] - extended_history[i-1][0]
                                dy_check = extended_history[i][1] - extended_history[i-1][1]
                                dt_check = extended_history[i][2] - extended_history[i-1][2]
                                vel_check = np.sqrt(dx_check**2 + dy_check**2) / dt_check if dt_check > 0 else 0
                                
                                if vel_check > 0:  # Only include moving points
                                    moving_points_after.append(extended_history[i])

                        # Also look ahead in future data if we don't have enough points after the pause
                        if len(moving_points_after) < angle_window:
                            look_ahead_idx = index + 1
                            while len(moving_points_after) < angle_window and look_ahead_idx < len(data):
                                future_row = data.iloc[look_ahead_idx]
                                
                                if look_ahead_idx > index:
                                    dx_future = future_row['X'] - data.iloc[look_ahead_idx-1]['X']
                                    dy_future = future_row['Y'] - data.iloc[look_ahead_idx-1]['Y']
                                    dt_future = time_interval
                                    vel_future = np.sqrt(dx_future**2 + dy_future**2) / dt_future if dt_future > 0 else 0
                                    
                                    if vel_future > 0:
                                        moving_points_after.append((future_row['X'], future_row['Y'], future_row['Time']))
                                
                                look_ahead_idx += 1
                        
                        # Compute angle if we have enough moving points before and after
                        if len(moving_points_before) >= angle_window and len(moving_points_after) >= angle_window:
                    
                            # Create vectors from the moving points
                            v1_before = np.array([
                                moving_points_before[-1][0] - moving_points_before[0][0],  # From first to last
                                moving_points_before[-1][1] - moving_points_before[0][1]
                            ])
                            v2_after = np.array([
                                moving_points_after[-1][0] - moving_points_after[0][0],   # From first to last
                                moving_points_after[-1][1] - moving_points_after[0][1]
                            ])
                            
                            angle_extended = calculate_angle(v1_before, v2_after)
                            if angle_extended > angle_threshold:
                                append_break(breaks, prev_values, angle_extended, velocity, acceleration, angle_window)
                                print(f" BREAK detected due to stop (moving points only) at time {row['Time']:.3f}, angle={angle_extended:.2f}°")
                                no_velocity_duration = 0  # Reset after detecting a break

                # elif np.allclose(v1, [0, 0], atol=1e-6):  # Détection d'arrêt via v1, faire en sorte daugmenter angle window à 10 si y'a un arret?
                    # append_break(breaks, prev_values, angle, velocity, acceleration, angle_window)
                    # print(f" BREAK detected due to stop (v1=0) at time {prev_values[angle_window][2]:.3f} (velocity={velocity:.2f}px/s)")

                elif not mouse_off_screen and not is_fast_enough and angle > 85: #put to 60? remettre?
                    # no_velocity_duration += time_interval
                    # if no_velocity_duration >= 0.04: #4/5?
                    append_break(breaks, prev_values, angle, velocity, acceleration, angle_window)
                    print(f" BREAK low velocity at time {prev_values[angle_window][2]:.3f} (velocity={velocity:.2f}px/s)")
                        # no_velocity_duration = 0  # Reset after detecting a break
                else:
                    no_velocity_duration = 0
            else:
                # Before trigger and on-screen: no break detection
                no_velocity_duration = 0
                low_velocity_duration = 0

        #prev_values.popleft()
        
    print(f"Break detection finished. Total breaks found: {len(breaks)}")
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
        


#remettre un critère low velocity? à 70? ou plus?
        vmin = 200 #put to 200?
        angle_threshold = 62.5
        time_interval = 0.01
        time_thresh = 0.1
        spatial_thresh = 35
        angle_window = 3 #3? faire 3 sur copie new et comparer avec 4 de test
        accel_threshold = 10000

        data_path_abs = pathlib.Path(os.path.abspath(data_path))
        explore_directory_for_copy(data_path_abs)
        print("data_path_abs", data_path_abs)

        script_dir = Path(__file__).resolve().parent
        os.chdir(script_dir)

        # Rename and move the trajectory coordinate files to the subject's folder
        second_cwd = os.getcwd()     

        # rename_script = 'rename_csv_total.sh'
        # rename_script_path = os.path.join(second_cwd, rename_script)
        # full_data_path = os.path.join(second_cwd, data_path)

        # # Read the current permissions
        # current_permissions = os.stat(rename_script_path).st_mode
        # # Add the executable bit for the owner, group, and others
        # new_permissions = current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        # # Change the mode of the file to make it executable
        # os.chmod(rename_script_path, new_permissions)
        # Call the shell script
        # subprocess.run([rename_script_path])
        subprocess.run(["python3", "rename_csv_total.py", data_path_abs])

        # subprocess.run([rename_script_path])
        subprocess.run(["python3", "rename_csv_total.py", data_path_abs])


        with open('resume_resultats.csv', 'w') as fd:
            fd.write("result_file, t_trigger, RT , RtTrig, t_trigger_computed, distance_to_trigger, "
                     "target_enters, t_first_target_enter, trigger_to_target_time, trigger_to_target_distance, "
                     "target_to_stop_time, target_to_stop_distance, total_movement_time, total_movement_distance, total_distance_travelled, "
                     "total_trial_time, finale_distance_to_center, finale_distance_to_center_time, "
                     "max_vx, t_max_vx, TtA, initial_movement_direction, movement_smoothness, trial_status, trial_feedback, target_position\n")
        explore_directory(data_path_abs)

        # ajoute les nouvelles variables SA et precision
        modify_resume_resultats(data_path_abs, vmin, angle_threshold, time_interval, time_thresh, spatial_thresh, angle_window)