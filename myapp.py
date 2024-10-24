import streamlit as st
import pandas as pd
import numpy as np
import wfdb
import neurokit2 as nk
import plotly.graph_objects as go

from read_record import Record, RecordReader
from scanning_window import scan_without_interval
from local_af_detection import find_r_peaks, calculate_rr_intervals, detect_af_in_window


st.title("Simple ECG Visualizer App")
record_selection = st.selectbox("Select a record", ["100","101","102","103","104","105","106","107","108","109","111","112","113","114","115","116","117","118","119","121","122","123","124","200","201","202","203","204","205","206","207","208","209","210", "211", "212", "213", "214", "215", "216", "217", "218", "219"])
record_name = int(record_selection)
current_record = RecordReader.read("D:\ECG DB\MIT_BIH_ARR",f"{record_name}",0,0,None)
current_signal = current_record['signal']
st.write(f"This is a simple app to visualize ECG signals.\n Here's the record we're using is {record_name} from MIT-BIH Arrhythmia Database.")



# Calculate the number of samples in one minute
sampling_rate = current_record['sampling_frequency']
samples_per_minute = sampling_rate * 60  

# Create a slider to select the starting minute
total_minutes = len(current_signal) // samples_per_minute
start_minute = st.slider("Select starting minute", 0, total_minutes - 1, 0)

# Calculate the start and end indices for the selected minute
start_index = start_minute * samples_per_minute
end_index = start_index + samples_per_minute

# Create the main ECG plot
fig_ecg = go.Figure()
fig_ecg.add_trace(go.Scatter(y=current_signal[start_index:end_index], mode='lines', name='ECG Signal'))

# Update layout for the main ECG plot
fig_ecg.update_layout(
    title='ECG Signal Visualization',
    xaxis_title='Sample',
    yaxis_title='Amplitude',
    height=500,
)
# Display the main ECG plot
st.plotly_chart(fig_ecg, use_container_width=True)
# Display additional information
st.write(f"Showing  {start_minute + 1}-minute out of {total_minutes} minutes")
st.write(f"Total signal length: {len(current_signal)} samples")



# Segment the data for the current minute
segment_length = 2  # in seconds
samples_per_segment = int(segment_length * sampling_rate)
num_segments = samples_per_minute // samples_per_segment

# Initialize session state for segment index if it doesn't exist
if 'segment_index' not in st.session_state:
    st.session_state.segment_index = 0

# Create next and previous buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Previous Segment"):
        st.session_state.segment_index = (st.session_state.segment_index - 1) % num_segments
with col2:
    if st.button("Next Segment"):
        st.session_state.segment_index = (st.session_state.segment_index + 1) % num_segments

# Get the current segment data
segment_start = start_index + st.session_state.segment_index * samples_per_segment
segment_end = segment_start + samples_per_segment
segment_data = current_signal[segment_start:segment_end]

# Perform AF detection
r_peaks = find_r_peaks(segment_data, sampling_rate)
rr_intervals = calculate_rr_intervals(r_peaks)
af_onset, af_offset = detect_af_in_window(r_peaks, rr_intervals, segment_data, 0, sampling_rate)

# Create the segment plot
fig_segment = go.Figure()

# Plot the ECG data
fig_segment.add_trace(go.Scatter(y=segment_data, mode='lines', name='ECG Segment'))

# Add R-peaks
fig_segment.add_trace(go.Scatter(
    x=r_peaks,
    y=segment_data[r_peaks],
    mode='markers',
    marker=dict(symbol='circle', size=8, color='red'),
    name='R-peaks'
))

# Add AF onset and offset if detected
if af_onset is not None:
    fig_segment.add_trace(go.Scatter(
        x=[af_onset],
        y=[segment_data[af_onset]],
        mode='markers',
        marker=dict(symbol='star', size=12, color='green'),
        name='AF Onset'
    ))

if af_offset is not None:
    fig_segment.add_trace(go.Scatter(
        x=[af_offset],
        y=[segment_data[af_offset]],
        mode='markers',
        marker=dict(symbol='star', size=12, color='blue'),
        name='AF Offset'
    ))

# Update layout for the segment plot
fig_segment.update_layout(
    title=f'2-Second Segment Visualization (Segment {st.session_state.segment_index + 1}/{num_segments})',
    xaxis_title='Sample',
    yaxis_title='Amplitude',
    height=400,
    showlegend=True
)

# Display the segment plot
st.plotly_chart(fig_segment, use_container_width=True)

# Display segment information
st.write(f"Showing segment {st.session_state.segment_index + 1} out of {num_segments}")
st.write(f"Number of R-peaks detected: {len(r_peaks)}")

# if af_onset is not None and af_offset is not None:
#     st.write(f"Atrial Fibrillation detected from sample {af_onset} to {af_offset}")
# elif af_onset is not None:
#     st.write(f"Atrial Fibrillation onset detected at sample {af_onset}")
# elif af_offset is not None:
#     st.write(f"Atrial Fibrillation offset detected at sample {af_offset}")
# else:
#     st.write("No Atrial Fibrillation detected in this segment")
