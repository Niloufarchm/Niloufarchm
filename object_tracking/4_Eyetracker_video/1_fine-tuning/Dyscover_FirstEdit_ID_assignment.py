import pandas as pd
import os

base_path = '/.../screen_recording_videos'
object_file = 'File_yolov8s8000.csv'
file_path = os.path.join(base_path, object_file)

df = pd.read_csv(file_path, low_memory=False)

num_objects = 8  
id_columns = [f'Object_ID_{i}' for i in range(num_objects)]
class_columns = [f'Class_{i}' for i in range(num_objects)]
x_columns = [f'Center_X_{i}' for i in range(num_objects)]
y_columns = [f'Center_Y_{i}' for i in range(num_objects)]
width_columns = [f'Width_{i}' for i in range(num_objects)]
height_columns = [f'Height_{i}' for i in range(num_objects)]
conf_columns = [f'Confidence_{i}' for i in range(num_objects)]

x_threshold = 40.0  # Threshold for x-axis
y_threshold = 40.0  # Threshold for y-axis

# Iterate through the DataFrame to find objects with IDs
for index in range(len(df)):
    for i in range(num_objects):
        if not pd.isna(df.at[index, id_columns[i]]):  # If an ID is detected
            current_id = df.at[index, id_columns[i]]
            current_position = (df.at[index, x_columns[i]], df.at[index, y_columns[i]])

            # Look back through all previous frames to find related objects without IDs
            lookback_index = index - 1
            while lookback_index >= 0:  # Continue until we reach the beginning of the DataFrame
                found_any_related = False

                for j in range(num_objects):
                    if pd.isna(df.at[lookback_index, id_columns[j]]):  # No ID
                        prev_position = (df.at[lookback_index, x_columns[j]], df.at[lookback_index, y_columns[j]])

                        # Calculate the differences in x and y coordinates
                        x_diff = abs(current_position[0] - prev_position[0])
                        y_diff = abs(current_position[1] - prev_position[1])

                        if x_diff < x_threshold and y_diff < y_threshold:
                            # Assign the ID if the object is closely related based on x and y thresholds
                            df.at[lookback_index, id_columns[j]] = current_id
                            current_position = prev_position  # Update the current position to continue looking back
                            found_any_related = True

                if not found_any_related:
                    break  # Stop looking further back if no related object was found in this frame

                lookback_index -= 1  # Move to the previous frame

# Remove objects that exist in fewer than 5 timestamps
object_counts = {col: df[col].value_counts() for col in id_columns}
for i, col in enumerate(id_columns):
    # Find IDs that appear in fewer than 5 timestamps
    ids_to_remove = object_counts[col][object_counts[col] < 5].index
    for obj_id in ids_to_remove:
        # Clear all features for these IDs
        indices_to_clear = df[df[col] == obj_id].index
        for index in indices_to_clear:
            df.at[index, id_columns[i]] = None
            df.at[index, class_columns[i]] = None
            df.at[index, x_columns[i]] = None
            df.at[index, y_columns[i]] = None
            df.at[index, width_columns[i]] = None
            df.at[index, height_columns[i]] = None
            df.at[index, conf_columns[i]] = None

# Additional Step: Remove objects with no ID (clear their data)
for index in range(len(df)):
    for i in range(num_objects):
        if pd.isna(df.at[index, id_columns[i]]):  # No ID
            # Clear all features related to this object
            df.at[index, class_columns[i]] = None
            df.at[index, x_columns[i]] = None
            df.at[index, y_columns[i]] = None
            df.at[index, width_columns[i]] = None
            df.at[index, height_columns[i]] = None
            df.at[index, conf_columns[i]] = None

df.to_csv(file_path, index=False)

print(f"Assigned IDs to related objects across multiple frames, cleared data for objects with no ID, and removed objects existing in fewer than 5 timestamps in {object_file}.")
