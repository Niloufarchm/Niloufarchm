import pandas as pd

file_path = '.../updated_annotated_gaze.csv'
filtered_file = pd.read_csv(file_path)

# Filter rows containing "FN" and "Start" in the event_name column
filtered_file = filtered_file[filtered_file['event_name'].str.contains('FN', na=False)]
filtered_file = filtered_file[filtered_file['event_name'].str.contains('Start', na=False)]

# Save the filtered file to a new CSV
filtered_output_path = 'updated_annotated_gaze.csv'
filtered_file.to_csv(filtered_output_path, index=False)

unique_fruit_ids = set()
unique_bomb_ids = set()

num_objects = 8

# Loop each row in filtered_file
for _, row in filtered_file.iterrows():
    # Loop through each object slot
    for j in range(num_objects):
        class_col = f'Class_{j}'
        id_col = f'Object_ID_{j}'

        # Check if the ID column is not NaN
        current_id = row[id_col]
        if pd.notna(current_id):  # Ensure the ID is valid
            current_class = row[class_col]  # Get the object class

            if current_class == 'Bomb':
                unique_bomb_ids.add(current_id)

            if current_class == 'Fruit':
                unique_fruit_ids.add(current_id)

num_fruit = len(unique_fruit_ids)
num_bomb = len(unique_bomb_ids)

print(f'Total unique Bombs: {num_bomb}')
print(f'Total unique Fruits: {num_fruit}')
