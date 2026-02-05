# raman-open-forge-imai

![App Screenshot](path/to/screenshot.png)  <!-- Replace with actual screenshot if available -->

A user-friendly PyQt5-based application for controlling line spectrometers, acquiring spectral data, performing Raman shift conversions with calibration, preprocessing spectra, managing a reference database, and searching for spectral matches. Optimized for Raman spectroscopy workflows. The app is specifically optimized for Hamamatsu CCD sensors and has been tested with the IRM785 spectrometer from Imai Optics. It supports DIY Raman systems based on Hamamatsu CCDs (see separate GitHub repository for details).
It also allows laser control for Imai Optics probe YM_RPL_785_500.

## Overview
Raman Open Forge Imai provides an integrated interface for spectrometer operation and data analysis. It handles device communication via serial port, real-time spectrum acquisition, advanced preprocessing using the `ramanspy` library, software/hardware calibration, and a database management for storing and querying reference spectra.
The application is designed to streamline Raman spectroscopy tasks, from data collection to identification.
Key capabilities include device parameter control, background subtraction, Raman shift calibration to correct peak positions, spectral preprocessing pipelines, peak detection, database management, and similarity-based searches with multiple metrics.

## [Ramanbase.org](https://ramanbase.org/) integration
We  provide .ipynb showing how to make custom reference databases based on ramanbase.org spectra for working with the app. An example of a small-scale database is presented here () and a larger processed database covering ~85k spectra can be downloaded here: [ramanbase.org - based db for working in the app]([https://github.com](https://huggingface.co/datasets/khittit15/ramanbase-for-imai-app/blob/main/rbase_specdictcur.zip))

Key capabilities include:

Device control (integration time, averaging, gain, offset, laser voltage, smoothing).
Spectrum acquisition (single or continuous mode) with background subtraction.
Conversion to Raman shifts with optional quadratic calibration to correct peak positions.
Advanced preprocessing: cropping, denoising (SavGol), baseline correction (ASLS), normalization.
Peak detection and export.
Spectral database management (add, delete, load/save as pickle files).
Database search using similarity metrics (SAD, SID, MAE, MSE, IUR for peaks).
Interactive plotting with zoom, reference overlay, and export options.

## Installation

1. Clone the repository:


![calib_0](https://github.com/user-attachments/assets/3e29ccb8-ef0c-4062-9ce2-c1112adef72e)
