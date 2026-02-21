import pandas as pd
import numpy as np
import os


def add_object_positions_to_files(base_path, files, object_file, threshold_ns=42e6):
    # Load the object positions file
    object_positions = pd.read_csv(os.path.join(base_path, object_file))

    for file in files:
        file_path = os.path.join(base_path, file)
        annotated_data = pd.read_csv(file_path)

        # Use the 'timestamp [ns]' field from annotated data directly
        updated_data = add_object_positions(annotated_data, object_positions, 'timestamp [ns]', 'timestamp [ns]',
                                            threshold_ns)

        # Define the path for saving the updated file
        updated_file_path = os.path.join(base_path, f"updated_{file}")
        # Save the updated data to a new CSV file
        updated_data.to_csv(updated_file_path, index=False)
        print(f"Updated data saved to {updated_file_path}")


def add_object_positions(annotated_data, object_positions, annotated_timestamp_field, object_timestamp_field,
                         threshold):
    # Sort the object positions by timestamp for more efficient searching
    object_positions = object_positions.sort_values(by=object_timestamp_field).reset_index(drop=True)

    # Initialize new columns in the annotated data for each object position attribute
    object_columns = object_positions.columns.drop(object_timestamp_field)
    for col in object_columns:
        annotated_data[col] = np.nan if object_positions[col].dtype.kind in 'ifc' else None

    # Function to find the nearest object position based on timestamp proximity
    def find_nearest(row):
        timestamp = row[annotated_timestamp_field]
        pos = np.searchsorted(object_positions[object_timestamp_field].values, timestamp, side='left')
        if pos == 0:
            nearest_idx = 0
        elif pos == len(object_positions):
            nearest_idx = len(object_positions) - 1
        else:
            before = object_positions.iloc[pos - 1][object_timestamp_field]
            after = object_positions.iloc[pos][object_timestamp_field]
            nearest_idx = pos if abs(after - timestamp) < abs(before - timestamp) else pos - 1

        nearest_timestamp = object_positions.iloc[nearest_idx][object_timestamp_field]
        if abs(nearest_timestamp - timestamp) <= threshold:
            return nearest_idx
        return -1  # Return -1 if no suitable object is found within the threshold

    # Apply the find_nearest function to each row in the annotated data
    nearest_indices = annotated_data.apply(find_nearest, axis=1)

    # Assign object position data based on the nearest indices found
    for col in object_columns:
        valid_indices = nearest_indices != -1
        if valid_indices.any():
            annotated_data.loc[valid_indices, col] = object_positions.iloc[nearest_indices[valid_indices]][col].values

    return annotated_data


base_path = ''
files = ['annotated_gaze.csv', 'annotated_imu.csv']
object_file = 'Filtered_File_yolov8s8000.csv'
add_object_positions_to_files(base_path, files, object_file)
