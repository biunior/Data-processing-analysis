#!/bin/bash

# Define the base directory
current_dir=$(pwd)
base_dir="${current_dir}/Output"

# Use find to list all directories containing "Trajectories"
find "$base_dir" -type d -name "Trajectories" | while IFS= read -r traj_dir; do
    echo "Processing directory: $traj_dir"

    # Extract the parent directory containing "Trajectories"
    parent_dir=$(dirname "$traj_dir")

    # Loop through each file matching the pattern *_{num}.csv
    for file in "$traj_dir"/*_[0-9]*.csv; do
        # Extract the number from the filename
        num=$(basename "$file" | grep -oE '[0-9]+') # Remove leading zeros
        new_num=$(echo "$num" | sed 's/^0*//') # without leading zeros

        # Construct the new filename
        new_filename=$(basename "$file" | sed "s/_${num}.csv/-trial${new_num}.csv/")

        # Print the old and new filenames for debugging
        # echo "Copying file: $file to $parent_dir/$new_filename"

        # Copy the file to the parent directory
        cp "$file" "$parent_dir/$new_filename"
    done
done

