# -*- coding: utf-8 -*-
"""
B-PASS_processing_script.py

================================================================================
B-PASS Data Processing for the Niigata Experiment
================================================================================

Description:
This script provides a comprehensive workflow for processing and analyzing B-PASS
data from the Niigata experiment. It handles data loading from various
sensor types (Atom and SmartSolo), preprocessing, cropping into experimental
time windows, cross-correlation analysis, and other advanced analyses.

Author: Ahmad Bahaa/The University of Tokyo
Date: 2024-09-05
Version: 1.0

License:
This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc-nd/4.0/ or see the LICENSE file.
"""

# =============================================================================
# SECTION 1: SETUP - LIBRARIES AND HELPER FUNCTIONS
# =============================================================================

# --- Core Libraries ---
import os
import mpu
import h5py
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- ObsPy - Seismic Data Processing ---
import obspy
from obspy import UTCDateTime, read, Stream, Trace
from obspy.signal import cross_correlation

# --- SciPy - Scientific Computing ---
from scipy.signal import butter, lfilter, hilbert, find_peaks
from scipy.fftpack import fft, ifft

# --- Utilities ---
from tqdm import tqdm
from IPython.display import clear_output

# This clears the output of the cell after execution, keeping the notebook clean.
clear_output()

# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #

def normalize_data(data):
    """
    Normalizes a NumPy array to the range [-1, 1].

    Args:
        data (np.ndarray): The input data array.

    Returns:
        np.ndarray: The normalized data.
    """
    data = data.astype(float)
    min_val, max_val = np.min(data), np.max(data)
    if max_val - min_val == 0:
        return data  # Avoid division by zero
    # Normalize to [0, 1]
    normalized = (data - min_val) / (max_val - min_val)
    # Scale to [-1, 1]
    return normalized * 2 - 1

def cross_coherence(receiver_data, source_data):
    """
    Computes the cross-coherence between two signals in the frequency domain.

    Args:
        receiver_data (np.ndarray): The receiver waveform.
        source_data (np.ndarray): The source waveform.

    Returns:
        np.ndarray: The real part of the cross-coherence result.
    """
    fft_receiver = fft(np.copy(receiver_data))
    fft_source = fft(np.copy(source_data))

    # Avoid division by zero for silent traces
    if np.all(np.abs(fft_source) == 0):
        return np.zeros_like(receiver_data)

    # Phase spectrum of receiver and source
    phase_receiver = fft_receiver / np.where(np.abs(fft_receiver) == 0, 1e-9, np.abs(fft_receiver))
    phase_source = fft_source / np.where(np.abs(fft_source) == 0, 1e-9, np.abs(fft_source))

    # Inverse FFT of the phase division gives the cross-coherence
    coherence = ifft(phase_receiver / phase_source)
    return np.real(coherence)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Applies a Butterworth bandpass filter to the data.

    Args:
        data (np.ndarray): The input signal.
        lowcut (float): Low frequency cutoff.
        highcut (float): High frequency cutoff.
        fs (int): Sampling rate.
        order (int): Filter order.

    Returns:
        np.ndarray: The filtered data.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def read_jgi_ascii_source(file_path, time_shift_ms=0):
    """
    Reads a JGI ASCII source file and returns an ObsPy Trace.

    Args:
        file_path (str): The full path to the ASCII file.
        time_shift_ms (int): A time shift in milliseconds to apply for alignment.

    Returns:
        obspy.Trace: The data as an ObsPy Trace object.
    """
    # Load data from the text file
    seismic_data = np.loadtxt(file_path)

    # Apply time shift if necessary (shift is in samples for np.roll)
    if time_shift_ms != 0:
        seismic_data = np.roll(seismic_data, -time_shift_ms)

    # Create a Trace object
    trace = Trace(data=seismic_data)

    # Extract start time from the filename (format: '..._YYYYMMDD-HHMMSS')
    filename = os.path.basename(file_path)
    time_str = filename[-15:].replace("-", "T")
    trace.stats.starttime = UTCDateTime(time_str) - (9 * 3600) # Convert from JST to UTC
    trace.stats.sampling_rate = 1000 # 1 kHz sampling rate

    return trace

# =============================================================================
# SECTION 2: DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_prepare_data():
    """
    Loads, merges, and prepares all raw seismic data (Atom and SmartSolo).
    This function is comprehensive and may require significant memory.

    Returns:
        dict: A dictionary of streams for 'Z', 'E', and 'N' components.
    """

    # --- Load 3-Component Atom Data (Stations A0114 and A0115) ---
    print("Loading 3-C Atom data...")
    S350114, S350115 = Stream(), Stream()

    # Helper to process a 3C Atom station
    def process_3c_atom(station_id, stream_obj):
        data_path = f'Atom Data/3-C/{station_id}/sg2 files/'
        for f in sorted(os.listdir(data_path)):
            st_temp = read(os.path.join(data_path, f))
            filetime = UTCDateTime(f.replace('_', '')[:-4])
            for i, ch in enumerate(['GHN', 'GHE', 'GHZ']):
                st_temp[i].stats.starttime = filetime
                st_temp[i].stats.channel = ch
                st_temp[i].stats.station = f'A{station_id[1:]}' # e.g., A0114
                st_temp[i].stats.location = 'R-CR-6'
            stream_obj += st_temp
        stream_obj.merge(method=1, fill_value=0)

    process_3c_atom('350114', S350114)
    process_3c_atom('350115', S350115)

    # --- Initialize Component Streams ---
    component_streams = {'Z': Stream(), 'E': Stream(), 'N': Stream()}

    # --- Load 1-Component Atom Data (Z-component only) ---
    print("Loading 1-C Atom data...")
    for atom_num in sorted(os.listdir('Atom Data/1-C/')):
        atom_folder = f'Atom Data/1-C/{atom_num}/sg2 files/'
        temp_stream = Stream()
        for f in sorted(os.listdir(atom_folder)):
            tr = read(os.path.join(atom_folder, f))[0]
            tr.stats.starttime = UTCDateTime(f.replace('_', '')[:-4])
            temp_stream += tr
        temp_stream.merge(method=1, fill_value=0)
        temp_stream[0].stats.station = f'A{atom_num}'
        component_streams['Z'] += temp_stream

    # --- Load SmartSolo Data for all three components ---
    print("Loading SmartSolo data...")
    for comp_char in ['Z', 'E', 'N']:
        mseed_dir = f'Smart Solo/MSEED/{comp_char}/'
        for f in sorted(os.listdir(mseed_dir)):
            component_streams[comp_char] += read(os.path.join(mseed_dir, f), format='MSEED')

    # --- Assign Locations and Finalize Streams ---
    print("Assigning metadata and finalizing streams...")
    station_names_Z = ['R-CR-8', 'R-CR-8*', 'R-CR-7', 'R-CR-9*', 'R-CR-9', 'R-CR-6*', 'R-CR-7*', 'R-CR-9*', 'R-CR-x10', 'R-CR-8', 'R-CR-7*', 'R-CR-7', 'R-CR-6*', 'R-CR-9', 'source', 'R-CR-8*', 'R-CR-x10', 'R-58-7', 'R-58-2', 'R-58-6', 'R-58R-4', 'R-58R-5', 'R-CR-4*', 'source', 'source', 'R-58R-3', 'R-CR-3', 'R-CR-1', 'R-58R-6', 'R-58-5', 'R-CR-4', 'R-58-4', 'R-58R-1', 'R-CR-5', 'R-58R-2', 'R-58-1', 'R-58-3', 'R-CR-3*', 'R-CR-2']
    station_names_EN = ['R-58-7', 'R-58-2', 'R-58-6', 'R-58R-4', 'R-58R-5', 'R-CR-4*', 'source', 'source', 'R-58R-3', 'R-CR-3', 'R-CR-1', 'R-58R-6', 'R-58-5', 'R-CR-4', 'R-58-4', 'R-58R-1', 'R-CR-5', 'R-58R-2', 'R-58-1', 'R-58-3', 'R-CR-3*', 'R-CR-2']

    for i, tr in enumerate(component_streams['Z']): tr.stats.location = station_names_Z[i]
    for i, tr in enumerate(component_streams['E']): tr.stats.location = station_names_EN[i]
    for i, tr in enumerate(component_streams['N']): tr.stats.location = station_names_EN[i]

    # Append the 3C Atom data
    component_streams['Z'] += S350114.select(channel="GHZ") + S350115.select(channel="GHZ")
    component_streams['E'] += S350114.select(channel="GHE") + S350115.select(channel="GHE")
    component_streams['N'] += S350114.select(channel="GHN") + S350115.select(channel="GHN")

    # Sort all streams
    for stream in component_streams.values():
        stream.sort(keys=['location'])

    print("\nData loading complete.")
    return component_streams

def crop_and_save_experiments(all_streams, experiments):
    """
    Crops full data streams into experimental time windows and saves them.

    Args:
        all_streams (dict): Dictionary of component streams.
        experiments (dict): Configuration for each experiment.
    """
    for name, config in experiments.items():
        print(f"Processing and saving experiment: {name}...")
        for comp_char, stream in all_streams.items():
            output_dir = f"clean_data_MSEED/{comp_char}_component/{name}/"
            os.makedirs(output_dir, exist_ok=True)

            stream_cut = stream.slice(config['start'], config['end']).copy()

            if 'remove_stations' in config:
                for station_id in config['remove_stations']:
                    for tr in stream_cut.select(station=station_id):
                        stream_cut.remove(tr)

            for i, tr in enumerate(stream_cut):
                tr.stats.network = str(i + 1)
                tr.stats.channel = f"GH{comp_char}"
                if tr.stats.station.startswith('S0'):
                    tr.stats.station = 'S' + tr.stats.station[2:]
                filename = f"{tr.stats.location}.{tr.stats.station}.MSEED"
                tr.write(os.path.join(output_dir, filename), format="MSEED")
    print("\nAll experiments cropped and saved.")

# =============================================================================
# SECTION 3: CROSS-CORRELATION ANALYSIS
# =============================================================================

def run_cross_correlation_analysis(experiment_name, component, station_distances, plot_params, source_station_id):
    """
    Runs cross-correlation, stacks the results, and generates a wiggle plot.

    Args:
        experiment_name (str): The name of the experiment folder.
        component (str): The component to process ('Z', 'E', or 'N').
        station_distances (list): List of station offsets in meters.
        plot_params (dict): Dictionary of plotting parameters.
        source_station_id (str): The station ID to use as the source.
    """
    print(f"--- Running analysis for {experiment_name}, Component: {component} ---")

    # --- Load Data ---
    data_dir = f"clean_data_MSEED/{component}_component/{experiment_name}/"
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}. Please run preprocessing first.")
        return

    data_stream = Stream()
    for f in sorted(os.listdir(data_dir)):
        data_stream += read(os.path.join(data_dir, f))

    # --- Cross-Correlation and Stacking ---
    all_correlations = Stream()
    window_length_sec = 29.999
    window_slide_sec = plot_params.get('slide', 120)

    for window in tqdm(data_stream.slide(window_length_sec, window_slide_sec), desc="Correlating"):
        source_trace_list = window.select(station=source_station_id)
        if not source_trace_list: continue # Skip if source is not in window
        source_trace = source_trace_list[0]

        for trace in window:
            # Shift=len//2 centers the correlation peak
            corr_result = cross_correlation.correlate(trace, source_trace, shift=len(trace.data) // 2)
            corr_trace = Trace(data=corr_result, header=trace.stats)
            all_correlations += corr_trace

    stacked_correlations = all_correlations.stack(group_by='id')

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 15))
    sampling_rate = stacked_correlations[0].stats.sampling_rate
    time_axis = np.arange(-window_length_sec / 2, window_length_sec / 2, 1 / sampling_rate)

    for i, trace in enumerate(stacked_correlations):
        normalized_trace = normalize_data(trace.data)
        ax.plot(time_axis, (normalized_trace * plot_params['amp']) + station_distances[i], 'k', lw=0.8)

    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Offset (m)', fontsize=14)
    title = f"{experiment_name.replace('_', ' ').title()} ({component}-Component)\nStacked Cross-Correlations"
    ax.set_title(title, fontsize=16)
    ax.set_xlim(plot_params['t_min'], plot_params['t_max'])
    ax.invert_yaxis()
    ax.grid(True, linestyle='--', alpha=0.5)

    fig_dir = "Figures/"
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(f"{fig_dir}{experiment_name}_{component}_corr_stack.png", dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# SECTION 4: MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """Main function to run the desired processing and analysis steps."""

    # --- Step 1: Preprocessing ---
    # This step is memory-intensive and should be run once.
    # It loads raw data and saves cropped, clean MSEED files for analysis.
    # ---------------------------------------------------------------------
    run_preprocessing = False # <<< SET TO TRUE TO RUN PREPROCESSING
    if run_preprocessing:
        # Define experimental time windows and configurations
        experiments = {
            "Borehole_PASS_0m": {"start": UTCDateTime(2023, 5, 17, 5, 0, 0), "end": UTCDateTime(2023, 5, 17, 23, 46, 0)},
            "Borehole_PASS_25m": {"start": UTCDateTime(2023, 5, 18, 0, 34, 0), "end": UTCDateTime(2023, 5, 18, 5, 0, 0)},
            "Borehole_PASS_50m_night": {"start": UTCDateTime(2023, 5, 18, 5, 36, 0), "end": UTCDateTime(2023, 5, 18, 14, 19, 0)},
            "Borehole_PASS_50m_morning": {"start": UTCDateTime(2023, 5, 18, 21, 15, 0), "end": UTCDateTime(2023, 5, 19, 1, 0, 0)}
        }
        all_component_streams = load_and_prepare_data()
        crop_and_save_experiments(all_component_streams, experiments)

    # --- Step 2: Cross-Correlation Analysis ---
    # This step requires the preprocessed data from Step 1.
    # ---------------------------------------------------------------------
    run_analysis = True # <<< SET TO TRUE TO RUN ANALYSIS
    if run_analysis:
        # Define station distances and plotting configurations
        distances_loc1 = [6,  16, 25, 34, 43, 53, 62, 11, 26, 52, 83, 119, 147, 178, 237, 315, 357, 400, 444, 487, 597, 649, 700, 755, 815, 874, 916, 967, 0]

        plot_config = {"amp": 3, "t_min": -1.0, "t_max": 3.0, "slide": 120}

        # Example analysis runs:
        run_cross_correlation_analysis(
            "Borehole_PASS_0m", 'Z', distances_loc1,
            plot_config, source_station_id="S893"
        )
        # Add more analysis calls here for other experiments or components.

if __name__ == "__main__":
    main()
