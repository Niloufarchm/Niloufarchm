import pandas as pd
import scipy.io as sio
import os

def csv_to_mat_as_table(file_path, save_directory):
    data = pd.read_csv(file_path)

    # Convert columns to appropriate types (numerical columns to numbers, others to strings)
    for col in data.columns:
        # Try to convert the column to numeric, errors='coerce' will convert non-convertible to NaN
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(data[col])

    # Convert remaining non-numeric data to string to ensure MATLAB compatibility
    data = data.astype({col: 'str' for col in data.columns if data[col].dtype == 'object'})

    # Prepare the data for MATLAB
    mat_data = {
        'updated_annotated_imu': data.to_numpy(), 
        'column_names': data.columns.to_list()  
    }

    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Create the .mat file path
    mat_file_name = os.path.splitext(os.path.basename(file_path))[0] + '.mat'
    mat_file_path = os.path.join(save_directory, mat_file_name)

    sio.savemat(mat_file_path, mat_data)

    print(f"File saved to {mat_file_path}")
    return mat_file_path

file_path = '.../updated_annotated_imu.csv'
save_directory = ''
csv_to_mat_as_table(file_path, save_directory)
