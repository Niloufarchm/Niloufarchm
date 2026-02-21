import pandas as pd

dyscover_path = '/Filtered_File_yolov8s8000.csv'
worldtimestamps_path = '.../world_timestamps.csv'
output_path = '.../Filtered_File_yolov8s8000.csv'

dyscover = pd.read_csv(dyscover_path)
worldtimestamps = pd.read_csv(worldtimestamps_path)

dyscover['timestamp [ns]'] = worldtimestamps['timestamp [ns]']

dyscover.to_csv(output_path, index=False)

print(f"Updated data saved to {output_path}")
