import pandas as pd
import os

base_path = '.../screen_recording_videos/'
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

output_file = os.path.join(base_path, "Filtered_" + object_file)
df.to_csv(output_file, index=False)

print(f"Removed objects existing in fewer than 5 timestamps and saved the filtered file to {output_file}.")
