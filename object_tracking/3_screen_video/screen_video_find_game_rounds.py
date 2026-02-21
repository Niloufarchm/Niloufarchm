import pandas as pd
import os

base_path = '.../screen_recording_videos'
object_file = 'Processed_File.csv'
file_path = os.path.join(base_path, object_file)
num_objects = 8  # Number of object columns
id_columns = [f'Object_ID_{i}' for i in range(num_objects)]
class_columns = [f'Class_{i}' for i in range(num_objects)]
x_columns = [f'Center_X_{i}' for i in range(num_objects)]
y_columns = [f'Center_Y_{i}' for i in range(num_objects)]
width_columns = [f'Width_{i}' for i in range(num_objects)]
height_columns = [f'Height_{i}' for i in range(num_objects)]
conf_columns = [f'Confidence_{i}' for i in range(num_objects)]

df = pd.read_csv(file_path, low_memory=False)

# Define start and end mappings from the notes
game_rounds = [
    {"start_class": "Fruit", "start_id": 19, "end_class": "Fruit", "end_id": 99},
    {"start_class": "Fruit", "start_id": 138, "end_class": "Fruit", "end_id": 222},
    {"start_class": "Fruit", "start_id": 260, "end_class": "Bomb", "end_id": 301},
    {"start_class": "Fruit", "start_id": 336, "end_class": "Fruit", "end_id": 495},
    {"start_class": "Fruit", "start_id": 570, "end_class": "Fruit", "end_id": 714},
    {"start_class": "Fruit", "start_id": 751, "end_class": "Bomb", "end_id": 797},
    {"start_class": "Fruit", "start_id": 825, "end_class": "Fruit", "end_id": 938},
    {"start_class": "Fruit", "start_id": 981, "end_class": "Bomb", "end_id": 1438},
    {"start_class": "Fruit", "start_id": 1626, "end_class": "Fruit", "end_id": 1672},
    {"start_class": "Fruit", "start_id": 1714, "end_class": "Bomb", "end_id": 1759},
    {"start_class": "Fruit", "start_id": 1852, "end_class": "Bomb", "end_id": 1909},
    {"start_class": "Fruit", "start_id": 2000, "end_class": "Fruit", "end_id": 2370},
    {"start_class": "Fruit", "start_id": 2446, "end_class": "Fruit", "end_id": 2839},
    {"start_class": "Fruit", "start_id": 2943, "end_class": "Fruit", "end_id": 3120},
    {"start_class": "Fruit", "start_id": 3156, "end_class": "Fruit", "end_id": 3484},
    {"start_class": "Fruit", "start_id": 3520, "end_class": "Fruit", "end_id": 3588},
    {"start_class": "Fruit", "start_id": 3629, "end_class": "Bomb", "end_id": 3684},
    {"start_class": "Fruit", "start_id": 3706, "end_class": "Bomb", "end_id": 3873},
]

valid_frame_ranges = []

for round_info in game_rounds:
    start_class = round_info["start_class"]
    start_id = round_info["start_id"]
    end_class = round_info["end_class"]
    end_id = round_info["end_id"]

    start_frames = df[
        (df[class_columns].isin([start_class]).any(axis=1)) &
        (df[id_columns].isin([start_id]).any(axis=1))
    ].index
    end_frames = df[
        (df[class_columns].isin([end_class]).any(axis=1)) &
        (df[id_columns].isin([end_id]).any(axis=1))
    ].index

    if len(start_frames) > 0 and len(end_frames) > 0:
        start_frame = start_frames.min()
        end_frame = end_frames.max()
        valid_frame_ranges.append((start_frame, end_frame))

# Collect all valid frames across all game rounds
valid_frames = set()
for start_frame, end_frame in valid_frame_ranges:
    valid_frames.update(range(start_frame, end_frame + 1))

# Filter out rows outside valid frames and clear object data
for index in df.index:
    if index not in valid_frames:
        for i in range(num_objects):
            df.at[index, id_columns[i]] = None
            df.at[index, class_columns[i]] = None
            df.at[index, x_columns[i]] = None
            df.at[index, y_columns[i]] = None
            df.at[index, width_columns[i]] = None
            df.at[index, height_columns[i]] = None
            df.at[index, conf_columns[i]] = None

output_file = os.path.join(base_path, "Filtered_" + object_file)
df.to_csv(output_file, index=False)

print(f"Filtered data saved to {output_file}.")
