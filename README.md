# B-PASS: Passive Seismic Data Processing for the Niigata Experiment

## 1. Project Overview
This repository contains a Python script designed for the processing and analysis of passive seismic data collected during the Niigata experiment. The workflow handles data from various sensor types (Atom and SmartSolo) and performs a series of standard and advanced seismological analyses.

The primary goals of the processing pipeline are:

- **Data Ingestion and Preprocessing**: Load raw seismic data, merge disparate time segments, and organize data by component (Z, E, N).
- **Experiment Cropping**: Isolate and save clean, continuous data streams corresponding to specific experimental time windows.
- **Cross-Correlation Analysis**: Utilize cross-correlation to identify coherent seismic phases between source signals and receiver stations.
- **Advanced Analysis**: Conduct specialized analyses, including evaluating the stability of signal stacking over time and assessing the repeatability of the seismic source.

This script is intended to provide a clear, commented, and reproducible workflow for researchers working with similar datasets.

## 2. Repository Structure
```
.
├── Atom Data/                  # (Input) Raw 1-C and 3-C Atom sensor data (not in repo)
├── Smart Solo/                 # (Input) Raw SmartSolo sensor data (not in repo)
├── clean_data_MSEED/           # (Output) Processed and cropped data in MSEED format (ignored by git)
├── Figures/                    # (Output) Generated plots and figures (ignored by git)
├── B-PASS_processing_script.py # Main Python script for all processing and analysis
├── README.md                   # This file
├── LICENSE                     # Project license file
└── .gitignore                  # Specifies files/directories to be ignored by Git
```

Note: The raw data directories (`Atom Data/`, `Smart Solo/`) are expected to be present but are not tracked by Git to avoid committing large datasets.

## 3. Requirements
This script is written in Python 3 and relies on several key scientific computing libraries. You can install them using pip:

```bash
pip install numpy pandas matplotlib seaborn obspy tqdm
```

## 4. Usage
The main script, `B-PASS_processing_script.py`, is organized into sections. To run a specific part of the workflow, you will need to uncomment the main execution blocks within the script.

### Data Preprocessing:
1. Place your raw data in the `Atom Data/` and `Smart Solo/` directories according to the structure expected by the script.
2. In `B-PASS_processing_script.py`, uncomment the lines under the "Main Execution Block for Preprocessing".
3. Run the script. This will generate the `clean_data_MSEED/` directory with the processed data.

### Cross-Correlation Analysis:
1. Ensure the preprocessing step has been completed.
2. In `B-PASS_processing_script.py`, uncomment the desired analysis calls under the "Main Execution Block for Analysis".
3. Run the script. This will generate figures in the `Figures/` directory.

## 5. License
This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You are free to:

- **Share** — copy and redistribute the material in any medium or format.

Under the following terms:

- **Attribution** — You must give appropriate credit.
- **NonCommercial** — You may not use the material for commercial purposes.
- **NoDerivatives** — If you remix, transform, or build upon the material, you may not distribute the modified material.

For more details, see the LICENSE file.
