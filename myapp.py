import streamlit as st
import pandas as pd
import numpy as np
import wfdb
import neurokit2 as nk
import plotly.graph_objects as go

from read_record import Record, RecordReader
from scanning_window import scan_without_interval

st.title("Simple ECG Visualizer App")
st.write("""
**Note:**
- This app is designed to work with the MIT-BIH Arrhythmia Database.
- Please select a record from the dropdown menu to visualize the ECG signal.
""")
record_selection = st.selectbox("Select a record", ["100","101","102","103","104","105","106","107","108","109","111","112","113","114","115","116","117","118","119","121","122","123","124","200","201","202","203","204","205","206","207","208","209","210", "211", "212", "213", "214", "215", "216", "217", "218", "219"])
record_name = int(record_selection)
st.write(f"This is a simple app to visualize ECG signals.\n Here's the record we're using is {record_name} from MIT-BIH Arrhythmia Database.")

# About MIT-BIH Arrhythmias Database
with st.expander("About MIT-BIH Arrhythmias Database"):
    st.write("""
             
    **Background**
    
    Since 1975, the laboratories at Boston's Beth Israel Hospital (now the Beth Israel Deaconess Medical Center) and at MIT have supported  own research into arrhythmia analysis and related subjects. One of the first major products of that effort was the MIT-BIH Arrhythmia Database, which completed and began distributing in 1980. The database was the first generally available set of standard test material for evaluation of arrhythmia detectors, and has been used for that purpose as well as for basic research into cardiac dynamics at more than 500 sites worldwide.

    **Data Description**
    
    The MIT-BIH Arrhythmia Database contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, obtained from 47 subjects studied by the BIH Arrhythmia Laboratory between 1975 and 1979. Twenty-three recordings were chosen at random from a set of 4000 24-hour ambulatory ECG recordings collected from a mixed population of inpatients (about 60%) and outpatients (about 40%) at Boston's Beth Israel Hospital; the remaining 25 recordings were selected from the same set to include less common but clinically significant arrhythmias that would not be well-represented in a small random sample.

    The recordings were digitized at 360 samples per second per channel with 11-bit resolution over a 10 mV range. Two or more cardiologists independently annotated each record; disagreements were resolved to obtain the computer-readable reference annotations for each beat (approximately 110,000 annotations in all) included with the database.

   About half (25 of 48 complete records, and reference annotation files for all 48 records) of this database has been freely available here since PhysioNet's inception in September 1999. The 23 remaining signal files, which had been available only on the MIT-BIH Arrhythmia Database CD-ROM, were posted here in February 2005.
   
   **Source:**
   
   https://archive.physionet.org/physiobank/database/mitdb/
   
    """)

current_record = RecordReader.read(f"{record_name}",0,0,None)
current_signal = current_record['signal']
current_annotations=current_record['symbol']
current_annotated_pt=current_record['sample']

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

segment_length = st.selectbox("Choose the segment length (in seconds)", [2,3,5,10])

# Segment the data for the current minute
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

# Clean the ECG segment
clean_ecg = nk.ecg_clean(current_signal[segment_start:segment_end], sampling_rate=sampling_rate)

# Create the segment plot with two signals
fig_segment = go.Figure()

# Plot the original ECG data
fig_segment.add_trace(go.Scatter(
    y=current_signal[segment_start:segment_end], 
    mode='lines', 
    name='Original ECG',
    line=dict(color='blue')
))

# Plot the cleaned ECG data
fig_segment.add_trace(go.Scatter(
    y=clean_ecg, 
    mode='lines', 
    name='Cleaned ECG',
    line=dict(color='red')
))

# Update layout for the segment plot
fig_segment.update_layout(
    title=f'{segment_length}-Second Segment Visualization (Segment {st.session_state.segment_index + 1}/{num_segments})',
    xaxis_title='Sample',
    yaxis_title='Amplitude',
    height=400,
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

# Add a note about the signals
st.write("""
**Signal Comparison:**
- Blue line: Original ECG signal
- Red line: Cleaned ECG signal (filtered using NeuroKit2)
""")

# Display the segment plot
st.plotly_chart(fig_segment, use_container_width=True)
# Display segment information
st.write(f"Showing segment {st.session_state.segment_index + 1} out of {num_segments}")
