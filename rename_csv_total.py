import os
import re
import shutil
from pathlib import Path
import sys

if len(sys.argv) != 2:
    print("This script except only one command line argument : the path to the directory containing the Outputs of the task.")
else:
    base_dir = Path(sys.argv[1])

# Get current working directory
#base_dir = Path.cwd() / "Task_output_analysis"

# Find all directories named "Trajectories"
trajectories_dirs = list(base_dir.rglob('Trajectories'))

for traj_dir in trajectories_dirs:
    print(f"Processing directory: {traj_dir}")

    # Get the parent directory of "Trajectories"
    parent_dir = traj_dir.parent

    # Loop through each matching file in the Trajectories directory
    for file in traj_dir.glob("*_[0-9]*.csv"):
        # Extract the number from the filename
        match = re.search(r'_(\d+)\.csv$', file.name)
        if match:
            num_str = match.group(1)
            new_num = str(int(num_str))  # Removes leading zeros

            # Build the new filename
            new_name = re.sub(rf"_{num_str}\.csv$", f"-trial{new_num}.csv", file.name)

            # Copy the file to the parent directory with the new name
            target_path = parent_dir / new_name
            shutil.copy(file, target_path)