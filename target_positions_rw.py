import re

# Reads from legacy format -resNEW.txt files with target and trigger coordinates 
def read_config_relative_positions(new_file):
    
    with open(new_file, 'r') as file:
        for line in file:
            if line.startswith("Départ"):
                # Extract the starting position
                starting_pos = line.split("(")[-1].split(")")[0]
                starting_pos = [int(x) for x in starting_pos.split(" , ")]

            if line.startswith("centre_cible_gauche"):
                # Extract left target center
                left_target_center = line.split("(")[-1].split(")")[0]
                left_target_center = [int(x) for x in left_target_center.split(" , ")]

            if line.startswith("centre_cible_droite"):
                # Extract right target center
                right_target_center = line.split("(")[-1].split(")")[0]
                right_target_center = [int(x) for x in right_target_center.split(" , ")]

            if line.startswith("centre_cible_centre"):
                # Extract center target center
                center_target_center = line.split("(")[-1].split(")")[0]
                center_target_center = [int(x) for x in center_target_center.split(" , ")]

            if line.startswith("cible_gauche"):
                # Extract left target boundaries
                left_target_boundaries = extract_boundaries(line)

            if line.startswith("cible_droite"):
                # Extract right target boundaries
                right_target_boundaries = extract_boundaries(line)

            if line.startswith("cible_centre"):
                # Extract center target boundaries
                center_target_boundaries = extract_boundaries(line)

            if line.startswith("Trigger"):
                # Extract trigger position
                trigger_pos = int(line.split("=")[-1].strip())

    # Combine results
    target_centers = [
        tuple(left_target_center),
        tuple(center_target_center),
        tuple(right_target_center),
    ]
    target_pos_boundaries = [
        left_target_boundaries,
        center_target_boundaries,
        right_target_boundaries,
    ]
    return starting_pos, target_pos_boundaries, target_centers, trigger_pos

# Helper function to extract target boundaries called by read_config_relative_positions
def extract_boundaries(line):
    # Regular expression to handle spaces and match the points
    pattern = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)"
    matches = re.findall(pattern, line)
    # Convert the matches into a list of tuples of integers
    return [(int(x[0]), int(x[1])) for x in matches]

# Writes info from legacy format -resNEW.txt files to the new format -absolute_positions.txt
def save_config_relative_positions(new_file, original_file):
    starting_pos, target_pos_boundaries, target_centers, trigger_pos = read_config_relative_positions(new_file)
    
    with open(original_file, 'w') as file:
        # Write absolute positions header
        file.write("ABSOLUTE POSITIONS:\n")
        
        # Write trigger positions
        file.write(f"Trigger positions used (y-axis): ({trigger_pos})\n")
        
        # Write target centers
        file.write("Target centers: ")
        target_centers_str = ",  ".join([f"{i}: ({center[0]}, {center[1]})" for i, center in enumerate(target_centers)])
        file.write(target_centers_str + "\n")
        
        # Write target center boundaries
        file.write("Target center boundaries (UPLEFT, UPRIGHT, DOWNRIGHT, DOWNLEFT): ")
        boundaries_str = ",  ".join(
            [f"{i}: ((" + "), (".join([f"{point[0]}, {point[1]}" for point in boundaries]) + "))"
             for i, boundaries in enumerate(target_pos_boundaries)]
        )
        file.write(boundaries_str + "\n")
        
        # Write starting position
        file.write(f"Starting point: ({starting_pos[0]}, {starting_pos[1]})\n")

# Reads from new format -absolute_positions.txt files with target and trigger coordinates
def read_config_absolute_positions(original_file):
    for line in original_file:
        if line.startswith("Starting point"):
            # extract the first tupple in paratheses in the line
            starting_pos = line.split("(")[-1].split(")")[0]
            # convert the string to a list of two integers
            starting_pos = starting_pos.split(", ")
            starting_pos = [int(x) for x in starting_pos]

        if line.startswith("Target centers"):
            # Regular expression pattern to match integers
            pattern = r"\d+: \((\d+), (\d+)\)"
            # Find all matches using the pattern
            matches = re.findall(pattern, line)
            # Create a list of tuples of integers
            target_centers = [(int(match[0]), int(match[1])) for match in matches]

        if line.startswith("Target center boundaries"):
            # Regular expression pattern to match nested tuples of integers
            pattern = r"\d+: \(\((\d+), (\d+)\), \((\d+), (\d+)\), \((\d+), (\d+)\), \((\d+), (\d+)\)\)"
            # Find all matches using the pattern
            matches = re.findall(pattern, line)
            # Create a list of tuples of tuples of integers
            target_pos_boundaries = [[(int(match[i]), int(match[i+1])) for i in range(0, len(match), 2)] for match in matches]

        if line.startswith("Trigger"):
            # extract the last value between parantheses in the line
            trigger_pos = int(line.split("(")[-1].split(")")[0])

    return starting_pos, target_pos_boundaries, target_centers, trigger_pos

# Writes info from new format -absolute_positions.txt files to the legacy format -resNEW.txt
def save_config_absolute_positions(original_file, new_file):

    starting_pos, target_pos_boundaries, target_centers, trigger_pos = read_config_absolute_positions(open(original_file, 'r'))
    
    with open(new_file, 'a') as new_file:
        new_file.write("Départ " + "( " + str(starting_pos[0]) + " , " + str(starting_pos[1]) + " )" +  "\n")
        new_file.write("cible_gauche " + "( " + str(target_pos_boundaries[0][0][0]) + " , " + str(target_pos_boundaries[0][0][1]) + " )" + " "  +  "( " + str(target_pos_boundaries[0][1][0]) + " , " + str(target_pos_boundaries[0][1][1]) + " )" +  " " +  "( " + str(target_pos_boundaries[0][2][0]) + " , " + str(target_pos_boundaries[0][2][1]) + " )" +  " " +  "( " + str(target_pos_boundaries[0][3][0]) + " , " + str(target_pos_boundaries[0][3][1]) + " )" + "\n")
        new_file.write("centre_cible_gauche " + "( " + str(target_centers[0][0]) + " , " + str(target_centers[0][1]) + " )" + "\n")
        new_file.write("cible_droite " + "( " + str(target_pos_boundaries[2][0][0]) + " , " + str(target_pos_boundaries[2][0][1]) + " )" + " "  +  "( " + str(target_pos_boundaries[2][1][0]) + " , " + str(target_pos_boundaries[2][1][1]) + " )" +  " " +  "( " + str(target_pos_boundaries[2][2][0]) + " , " + str(target_pos_boundaries[2][2][1]) + " )" +  " " +  "( " + str(target_pos_boundaries[2][3][0]) + " , " + str(target_pos_boundaries[2][3][1]) + " )" + "\n")
        new_file.write("centre_cible_droite " + "( " + str(target_centers[2][0]) + " , " + str(target_centers[2][1]) + " )" + "\n")
        new_file.write("cible_centre " + "( " + str(target_pos_boundaries[1][0][0]) + " , " + str(target_pos_boundaries[1][0][1]) + " )" + " "  +  "( " + str(target_pos_boundaries[1][1][0]) + " , " + str(target_pos_boundaries[1][1][1]) + " )" +  " " +  "( " + str(target_pos_boundaries[1][2][0]) + " , " + str(target_pos_boundaries[1][2][1]) + " )" +  " " +  "( " + str(target_pos_boundaries[1][3][0]) + " , " + str(target_pos_boundaries[1][3][1]) + " )" + "\n")
        new_file.write("centre_cible_centre " + "( " + str(target_centers[1][0]) + " , " + str(target_centers[1][1]) + " )" + "\n")
        new_file.write("Trigger  y = " + str(trigger_pos) + "\n")
