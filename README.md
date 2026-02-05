# raman-open-forge-imai

A user-friendly PyQt5-based application for controlling line spectrometers, acquiring spectral data, performing Raman shift conversions with calibration, preprocessing spectra, managing a reference database, and searching for spectral matches. Optimized for Raman spectroscopy workflows. The app is specifically optimized for Hamamatsu CCD sensors and has been tested with the IRM785 spectrometer from Imai Optics. It supports DIY Raman systems based on Hamamatsu CCDs (see separate GitHub repository for details).
It also allows laser control for Imai Optics probe YM_RPL_785_500.

## Overview
Raman Open Forge Imai provides an integrated interface for spectrometer operation and data analysis. It handles device communication via serial port, real-time spectrum acquisition, advanced preprocessing using the `ramanspy` library, software/hardware calibration, and a database management for storing and querying reference spectra.
The application is designed to streamline Raman spectroscopy tasks, from data collection to identification.
Key capabilities include device parameter control, background subtraction, Raman shift calibration to correct peak positions, spectral preprocessing pipelines, peak detection, database management, and similarity-based searches with multiple metrics.

## [Ramanbase.org](https://ramanbase.org/) integration
We  provide [.ipynb](https://github.com/ACDBio/raman-open-forge-imai/blob/main/RBaseproc.ipynb) showing how to make custom reference databases based on ramanbase.org spectra for working with the app. An example of a small-scale database is presented here () and a larger processed database covering ~85k spectra can be downloaded [here](https://huggingface.co/datasets/khittit15/ramanbase-for-imai-app/blob/main/rbase_specdictcur.zip).

Key capabilities include:

 - Device control (integration time, averaging, gain, offset, laser voltage (if probe is present), smoothing). 
 - Spectrum acquisition (single or continuous mode) with background subtraction. 
 - Conversion to Raman shifts with optional quadratic software calibration to correct peak positions.
 - Advanced preprocessing: cropping, denoising (SavGol), baseline correction (ASLS), normalization.
 - Peak detection and export.
 - Spectral database management (add, delete, load/save as pickle files).
 - Database search using similarity metrics (SAD, SID, MAE, MSE, IUR for peaks).
 - Interactive plotting with zoom, reference overlay, and export options.



## Features
### Device Connection and Control

 - Scan and connect to serial ports (e.g., USB spectrometers). 
 - Set integration time (1–60,000 ms), hardware averaging (1–255). 
 - Advanced settings: gain (0–255), offset (-255–255), laser voltage (0–5000 mV), trigger out (HIGH/LOW), smoothing level (1–10). 
 - Read current device parameters on connection. 
 - Save parameters to device flash. 

### Spectrum Acquisition

 - Single acquisition or continuous mode with pause.
 - Background spectrum acquisition and automatic subtraction.
 - Real-time FPS display and acquisition count.
 - Raw spectrum plotting with optional auto-zoom.

### Raman Shift Conversion and Calibration

 - Toggle conversion to Raman shifts using excitation wavelength (adjustable, default 785 nm).
 - Software calibration: polynomial fit on observed vs expected shifts (from table input).
 - Hardware calibration: read/write coefficients for wavelength, Raman shift, and intensity groups.
 - Preview calibrated axis and save to device flash.
 - Load/save calibration files as CSV.

### Spectrum Processing

Pipeline based on ramanspy:
 - Crop region (min/max shift).
 - Savitzky-Golay denoising (window length, poly order).
 - ASLS baseline correction.
 - Normalization (MinMax or Vector).

 - Peak finding with prominence and width thresholds.
 - Processed spectrum plotting with peak labels.
 - Export processed data and peaks as CSV.

### Database Management

 - Load/save databases as pickle files (.pkl).
 - Add current spectrum to database with custom name.
 - Delete spectra from database.
 - Create new empty database.
 - Auto-load default database on startup.
 - Reload current database.

### Database Search

 - Enable/disable search in processing panel.
 - Metrics: SAD, SID, MAE, MSE, IUR (peak-based intersection over union with tolerance).
 - Preprocess database spectra option.
 - Minimum axis overlap threshold.
 - Top-N results display and download as CSV.
 - Reference plotting from search results or entire database.

### Plotting and Visualization

 - Interactive plots using pyqtgraph.
 - Overlay reference spectra (green line).
 - Auto-zoom with margin; default views for wavelength (796–1119 nm) and Raman (0–2000 cm⁻¹).
 - Peak lines and labels (dashed red).
 - Log widget for application events.

### UI Layout

 - Main window: control buttons, plot, log.
 - Docks: Process Spectra (left), Calibration (right), Manage Database (left), Advanced Settings (right).
 - Status bar with FPS and acquisition count.




## Installation

1. Clone the repository:


![calib_0](https://github.com/user-attachments/assets/3e29ccb8-ef0c-4062-9ce2-c1112adef72e)
