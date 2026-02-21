import pandas as pd


def annotate_filtered_data():
    events = pd.read_csv('.../events.csv')
    filtered_data = pd.read_csv('.../nonFiltered_File_yolov8s8000.csv')

    # Ensure events are sorted by 'timestamp [ns]'
    events_sorted = events.sort_values(by='timestamp [ns]')

    # Annotate the filtered data with event names
    filtered_data['event_name'] = filtered_data['timestamp [ns]'].apply(
        lambda x: find_event_for_timestamp(x, events_sorted))

    # Write the annotated data back to a CSV file
    output_file = 'Annotated_nonFiltered_File_yolov8s8000.csv'
    filtered_data.to_csv(output_file, index=False)

    print(f"Filtered data has been annotated and saved to {output_file}")


def find_event_for_timestamp(timestamp, events_sorted):
    """
    Find the name of the event occurring at or just after a given timestamp.
    """
    # Find the first event where the timestamp is greater than or equal to the given timestamp
    idx = (events_sorted['timestamp [ns]'] >= timestamp).idxmax()

    if idx == 0 or pd.isna(idx):
        return events_sorted.iloc[idx]['name'] if len(events_sorted) > 0 else ''

    # Check if the previous event's timestamp is less than or equal to the given timestamp
    if idx > 0 and events_sorted.iloc[idx - 1]['timestamp [ns]'] <= timestamp:
        return events_sorted.iloc[idx - 1]['name']

    return events_sorted.iloc[idx]['name'] if len(events_sorted) > 0 else ''


# Run the function to annotate the filtered data
if __name__ == "__main__":
    annotate_filtered_data()
