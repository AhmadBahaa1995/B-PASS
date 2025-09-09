
# B-PASS: Seismic Data Processing and Simulation

## 1. Project Overview
This repository contains Python scripts for processing seismic data from the Niigata experiment and for simulating seismic wave propagation to model the effects of CO₂ injection. The project combines real-world data analysis with forward modeling to understand subsurface changes (Doi:10.5281/zenodo.17075959).

The primary goals of the project are:

- **Data Ingestion and Preprocessing**: Load raw seismic data (from Atom and SmartSolo sensors), merge time segments, and organize data by component.
- **Cross-Correlation Analysis**: Utilize cross-correlation to identify coherent seismic phases and analyze signal stability and source repeatability.
- **Seismic Forward Modeling**: Use the Marmousi-II elastic model to simulate seismic surveys.
- **CO₂ Injection Simulation**: Model the geophysical response of CO₂ injection by altering the elastic properties of the subsurface model and simulating the time-lapse seismic response.

This repository provides reproducible workflows for both empirical data analysis and numerical simulation.

## 2. Repository Structure
```
.
├── Atom Data/                              # (Input) Raw 1-C and 3-C Atom sensor data 
├── Smart Solo/                             # (Input) Raw SmartSolo sensor data 
├── siesmic model/                          # (Input) Contains the Marmousi-II elastic model (.npy file)
│   └── Marmousi2_elastic/
│       └── model.npy
├── clean_data_MSEED/                       # (Output) Processed and cropped data in MSEED format 
├── Figures/                                # (Output) Generated plots from data analysis
├── U-net project/                          # (Output) Generated plots from simulation script 
├── B-PASS_processing_script.py             # Main script for data preprocessing and analysis
├── B_pass_simulation_modeling.py           # Main script for seismic modeling and CO₂ simulation
├── LICENSE                                 # License file for the project
└── README.md                               # This file
```

## 3. Dependencies
The scripts require several scientific computing and seismology libraries.

**For `B-PASS_processing_script.py`:**
```bash
pip install numpy pandas matplotlib seaborn obspy tqdm
```

**For `B_pass_simulation_modeling.py`:**  
This script requires a GPU-enabled environment with PyTorch and Deepwave installed.

```bash
# Install PyTorch (ensure correct version for your CUDA setup)
# See: https://pytorch.org/get-started/locally/
pip install torch numpy matplotlib scipy

# Install Deepwave
pip install deepwave
```

## 4. Usage
The repository contains two main scripts for different workflows.

### 4.1.  Data Processing (`B-PASS_processing_script.py`)
1. Place your raw data in the `Atom Data/` and `Smart Solo/` directories.  
2. In `B-PASS_processing_script.py`, uncomment the main execution blocks for either preprocessing or analysis.  
3. Run the script:
```bash
python B-PASS_processing_script.py
```
Processed data will be saved in `clean_data_MSEED/` and figures in `Figures/`.

### 4.2. Seismic Simulation (`B_PASS_simulation_modeling.py`)
1. Ensure the Marmousi-II model is located at `siesmic model/Marmousi2_elastic/model.npy`.  
2. Update the `filename` variable in the script if your path is different.  
3. The script will:
   - Load the elastic model.
   - Inject a simulated CO₂ plume by altering Vp, Vs, and density.
   - Run both elastic and scalar (P-wave) seismic surveys using Deepwave.
   - Generate and save comparison plots in the `U-net project/` directory.

Run the script:
```bash
python B_PASS_simulation_modeling.py
```
Data set:
The raw and stacked B-PASS source functions, along with formatted seismometer datasets, are archived here under [DOI: 10.5281/zenodo.17075241].

## 5. License
This work is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](LICENSE).

You are free to:

- **Share** — copy and redistribute the material in any medium or format.

Under the following terms:

- **Attribution** — You must give appropriate credit.
- **NonCommercial** — You may not use the material for commercial purposes.
- **NoDerivatives** — If you remix, transform, or build upon the material, you may not distribute the modified material.
