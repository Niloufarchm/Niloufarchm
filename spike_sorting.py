import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import spikeinterface.sorters as ss
import numpy as np

file_path = "/Volumes/ExtremeSSD/Mac/matlab-probability-class/Data/change_detection/Monkey/Camelot/electrophysiology/20260330/cam_20260330_26350_LEFTEYEFF_030.ns6"

recording = se.BlackrockRecordingExtractor(file_path)

print("Num channels:", recording.get_num_channels())
print("Channel IDs:", recording.get_channel_ids())

recording = sp.bandpass_filter(recording, freq_min=300, freq_max=6000)

# Do NOT assign back unless you've verified this method returns a recording
recording.set_dummy_probe_from_locations(np.array([[0.0, 0.0]]))

sorting = ss.run_sorter(
    sorter_name="tridesclous2",
    recording=recording,
    folder="sorting_output",
    remove_existing_folder=True
)

print("Number of units found:", len(sorting.get_unit_ids()))
print("Unit IDs:", sorting.get_unit_ids())

