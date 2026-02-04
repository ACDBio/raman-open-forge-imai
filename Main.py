import sys
import serial
import numpy as np
from PyQt5 import QtWidgets, QtCore
import serial.tools.list_ports
import pyqtgraph as pg
import time
from scipy.signal import find_peaks
import ramanspy as rp
import pickle
from scipy.interpolate import interp1d
import pandas as pd
import warnings
import copy
import os
warnings.filterwarnings("ignore", category=RuntimeWarning)
class SpectrometerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.serial_port = serial.Serial()
        self.serial_port.timeout = 10
        self.continuous_mode = False
        self.background_spectrum = None
        self.average_count = 1
        self.wavelength_min = 796
        self.wavelength_max = 1119
        self.wavelengths = np.linspace(self.wavelength_min, self.wavelength_max, 2048)
        self.frame_count = 0
        self.start_time = time.time()
        self.collection_count = 0
        self.max_cache_size = 5
        self.smoothing_factor = 0.1
        self.current_spectrum_1 = None
        self.original_spectrum = None
        self.processed_spectrum = None
        self.peaks = None
        self.rspec=None
        self.cur_spectrum_is_db=False
        self.current_db_path = None
        self.specdict = None
        self.searchres = None
        self.calib_coeffs = {
                    1: [0.0, 0.0, 0.0, 0.0], # group 1 – usually wavelength
                    2: [0.0, 0.0, 0.0, 0.0], # group 2 – Raman shift
                    3: [0.0, 0.0, 0.0, 0.0], # group 3 – analog/light source?
                }
        self.calib_group_names = {
            1: "Wavelength (nm)",
            2: "Raman shift (cm⁻¹)",
            3: "Analog / Intensity corr."
        }
        self.calib_coeffs_soft = None # [a, b, c] for ax^2 + bx + c
        self.is_calibrated = False
        self.current_calib_path = None
        self.spectral_axis = None
        self.use_raman = False
        self.smoothing_level = 6  # default
        self.init_ui()
        default_path = os.getcwd()+"/rbase_specdictcur.pkl"
        if os.path.exists(default_path):
            try:
                with open(default_path, "rb") as f:
                    self.specdict = pickle.load(f)
                self.current_db_path = default_path
                self.log(f"Auto-loaded current database: {default_path} (n={len(self.specdict)} spectra)")
            except Exception as e:
                self.log(f"Failed to auto-load default database '{default_path}': {e}")
                self.specdict = {}
        else:
            self.log("Default database file not found. Please load one manually.")
            self.specdict = {}
        self.load_default_calibration()
        self.searchres = None
        self.peak_lines = []
        self.peak_labels = []
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMaximumSize(50, 10)
        self.progress_bar.setTextVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.progress_bar.hide()
        self.create_process_panel()
        self.create_manage_db_panel()
        self.create_advanced_panel()
    def load_default_calibration(self):
        default_path = "calibration_cur.csv"
        if os.path.exists(default_path):
            try:
                df = pd.read_csv(default_path, header=None)
                self.calib_coeffs_soft = df.values.flatten()[:3].tolist()
                self.current_calib_path = default_path
                self.is_calibrated = True
                self.use_raman = True
                self.log(f"Loaded default calibration: {self.calib_coeffs_soft}")
            except Exception as e:
                self.log(f"Failed to load default calibration: {e}")
                self.calib_coeffs_soft = None
                self.is_calibrated = False
                self.use_raman = False
        else:
            self.log("Calibration file not found")
            self.calib_coeffs_soft = None
            self.is_calibrated = False
            self.use_raman = False
    def init_ui(self):
        self.setWindowTitle('Line Spectra Viewer v2.0')
        screen = QtWidgets.QDesktopWidget().screenGeometry()
        self.resize(int(screen.width() * 0.8), int(screen.height() * 0.85))
        self.move(int(screen.width() * 0.1), int(screen.height() * 0.075))
        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(1)
        control_layout = self.create_control_layout()
        layout.addLayout(control_layout)
        self.plot_widget = pg.PlotWidget(background='w')
        layout.addWidget(self.plot_widget)
        self.plot_curve_1 = self.plot_widget.plot(pen=pg.mkPen('b', width=2), name="Spectrum")
        self.plot_curve_ref = self.plot_widget.plot(pen=pg.mkPen('g', width=2), name="Reference")
        self.plot_curve_ref.setVisible(False)
        self.plot_widget.setLabel('left', 'Light Intensity')
        self.plot_widget.setLabel('bottom', 'Wavelength (nm) or Raman shift (cm<sup>-1</sup>)')
        self.status_bar = self.statusBar()
        self.fps_label = QtWidgets.QLabel("FPS: 0")
        self.status_bar.addPermanentWidget(self.fps_label)
        self.collection_label = QtWidgets.QLabel("Acquisitions: 0")
        self.status_bar.addPermanentWidget(self.collection_label)
        self.log_widget = QtWidgets.QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumHeight(120)
        self.log_widget.setPlaceholderText("Application log...")
        layout.addWidget(self.log_widget)
        central_scroll = QtWidgets.QScrollArea()
        central_scroll.setWidget(central_widget)
        central_scroll.setWidgetResizable(True)
        central_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        central_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setCentralWidget(central_scroll)
        self.process_dock = QtWidgets.QDockWidget("Process Spectra", self)
        self.process_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.process_dock)
        self.process_dock.setVisible(False)
        self.calib_dock = QtWidgets.QDockWidget("Calibration", self)
        self.calib_dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea | QtCore.Qt.LeftDockWidgetArea)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.calib_dock)
        self.calib_dock.setVisible(False)
        self.calib_dock.setMinimumWidth(380)
        calib_widget = QtWidgets.QWidget()
        calib_layout = QtWidgets.QVBoxLayout(calib_widget)
        self.calib_tabs = QtWidgets.QTabWidget()
        for grp in [1,2,3]:
            tab = QtWidgets.QWidget()
            lay = QtWidgets.QFormLayout(tab)
            self.calib_coeffs[grp] = [QtWidgets.QDoubleSpinBox() for _ in range(4)]
            for i, coef in enumerate(['f₀ (×10⁶)', 'f₁ (×10³)', 'f₂', 'f₃']):
                spin = self.calib_coeffs[grp][i]
                spin.setRange(-10000, 10000)
                spin.setDecimals(10)
                spin.setSingleStep(0.000001 if i < 2 else 0.001)
                lay.addRow(f"{coef}:", spin)
            self.calib_tabs.addTab(tab, self.calib_group_names[grp])
        # Software Calibration tab
        tab_soft = QtWidgets.QWidget()
        lay_soft = QtWidgets.QVBoxLayout(tab_soft)
        self.lbl_current_calib = QtWidgets.QLabel("No calibration loaded")
        self.lbl_current_calib.setStyleSheet("color: gray; font-style: italic;")
        lay_soft.addWidget(self.lbl_current_calib)
        btn_load_calib = QtWidgets.QPushButton("Load Calibration...")
        btn_load_calib.clicked.connect(self.load_calibration_file)
        lay_soft.addWidget(btn_load_calib)
        btn_save_calib = QtWidgets.QPushButton("Save Calibration as...")
        btn_save_calib.clicked.connect(self.save_calibration_as)
        lay_soft.addWidget(btn_save_calib)
        btn_save_default_calib = QtWidgets.QPushButton("Save as Default")
        btn_save_default_calib.clicked.connect(self.save_as_default_calib)
        lay_soft.addWidget(btn_save_default_calib)
        lay_soft.addStretch()
        self.calib_tabs.addTab(tab_soft, "Software Calib")
        # Perform Calibration tab
        tab_perform = QtWidgets.QWidget()
        lay_perform = QtWidgets.QVBoxLayout(tab_perform)
        self.calib_table = QtWidgets.QTableWidget(10, 2)
        self.calib_table.setHorizontalHeaderLabels(["Observed Raman Shift (cm⁻¹)", "Expected Raman Shift (cm⁻¹)"])
        lay_perform.addWidget(self.calib_table)
        btn_fit = QtWidgets.QPushButton("Fit Polynomial")
        btn_fit.clicked.connect(self.fit_calibration)
        lay_perform.addWidget(btn_fit)
        lay_perform.addStretch()
        self.calib_tabs.addTab(tab_perform, "Perform Calib")
        calib_layout.addWidget(self.calib_tabs)
        btn_read = QtWidgets.QPushButton("Read All Coefficients from Device")
        btn_read.clicked.connect(self.read_all_calibration)
        calib_layout.addWidget(btn_read)
        btn_write = QtWidgets.QPushButton("Write Selected Group to Device")
        btn_write.clicked.connect(self.write_selected_calibration)
        calib_layout.addWidget(btn_write)
        btn_save_flash = QtWidgets.QPushButton("Save Parameters to Flash (0x22)")
        btn_save_flash.clicked.connect(self.save_parameters_to_flash)
        calib_layout.addWidget(btn_save_flash)
        btn_preview = QtWidgets.QPushButton("Preview Axis (current group)")
        btn_preview.clicked.connect(self.preview_calibration_axis)
        calib_layout.addWidget(btn_preview)
        calib_layout.addStretch()
        calib_scroll = QtWidgets.QScrollArea()
        calib_scroll.setWidget(calib_widget)
        calib_scroll.setWidgetResizable(True)
        calib_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        calib_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.calib_dock.setWidget(calib_scroll)
        btn_calib = QtWidgets.QPushButton("Calibration...")
        btn_calib.clicked.connect(lambda: self.calib_dock.setVisible(not self.calib_dock.isVisible()))
        control_layout.addWidget(btn_calib)
    def load_calibration_file(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Calibration", "", "CSV Files (*.csv)")
        if file_name:
            try:
                df = pd.read_csv(file_name, header=None)
                self.calib_coeffs_soft = df.values.flatten()[:3].tolist()
                self.current_calib_path = file_name
                self.is_calibrated = True
                self.use_raman = True
                self.checkbox_to_raman.setChecked(True)
                self.lbl_current_calib.setText(f"Calib: {os.path.basename(file_name)}")
                self.lbl_current_calib.setStyleSheet("color: green;")
                self.log(f"Loaded calibration: {file_name} coeffs {self.calib_coeffs_soft}")
                if self.current_spectrum_1 is not None and not self.cur_spectrum_is_db:
                    self.spectral_axis = self.get_current_axis()
                    self.update_plot(self.spectral_axis, self.current_spectrum_1)
            except Exception as e:
                self.log(f"Failed to load calibration: {e}")
    def save_calibration_as(self):
        if self.calib_coeffs_soft is None:
            self.log("No calibration to save")
            return
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Calibration", "", "CSV Files (*.csv)")
        if file_name:
            try:
                pd.DataFrame(self.calib_coeffs_soft).to_csv(file_name, index=False, header=False)
                self.log(f"Saved calibration to {file_name}")
            except Exception as e:
                self.log(f"Failed to save calibration: {e}")
    def save_as_default_calib(self):
        if self.calib_coeffs_soft is None:
            self.log("No calibration to save")
            return
        default_path = "calibration_cur.csv"
        if os.path.exists(default_path):
            reply = QtWidgets.QMessageBox.question(self, "Overwrite?", "Overwrite default calibration?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if reply != QtWidgets.QMessageBox.Yes:
                return
        try:
            pd.DataFrame(self.calib_coeffs_soft).to_csv(default_path, index=False, header=False)
            self.current_calib_path = default_path
            self.lbl_current_calib.setText(f"Calib: {os.path.basename(default_path)} (default)")
            self.lbl_current_calib.setStyleSheet("color: green;")
            self.log(f"Saved as default calibration: {default_path}")
        except Exception as e:
            self.log(f"Failed to save default calibration: {e}")
    def fit_calibration(self):
        observed_shifts = []
        expected_shifts = []
        for row in range(self.calib_table.rowCount()):
            obs_item = self.calib_table.item(row, 0)
            exp_item = self.calib_table.item(row, 1)
            if obs_item and exp_item:
                try:
                    obs = float(obs_item.text())
                    exp = float(exp_item.text())
                    observed_shifts.append(obs)
                    expected_shifts.append(exp)
                except:
                    pass
        if len(observed_shifts) < 3:
            QtWidgets.QMessageBox.warning(self, "Error", "Need at least 3 points for quadratic fit")
            return
        coeffs = np.polyfit(observed_shifts, expected_shifts, 2)
        self.calib_coeffs_soft = coeffs.tolist()
        self.is_calibrated = True
        self.use_raman = True
        self.checkbox_to_raman.setChecked(True)
        self.log(f"Fitted calibration coeffs: {self.calib_coeffs_soft}")
        self.lbl_current_calib.setText("Calib: Fitted (unsaved)")
        self.lbl_current_calib.setStyleSheet("color: blue;")
        if self.current_spectrum_1 is not None and not self.cur_spectrum_is_db:
            self.spectral_axis = self.get_current_axis()
            self.update_plot(self.spectral_axis, self.current_spectrum_1)
    def create_control_layout(self):
        control_layout = QtWidgets.QHBoxLayout()
        self.combo_ports = QtWidgets.QComboBox()
        self.update_serial_ports()
        control_layout.addWidget(self.combo_ports)
        btn_refresh = QtWidgets.QPushButton('Refresh Ports')
        btn_refresh.clicked.connect(self.update_serial_ports)
        control_layout.addWidget(btn_refresh)
        self.combo_baudrate = QtWidgets.QComboBox()
        self.combo_baudrate.addItems(["19200", "38400", "115200"])
        self.combo_baudrate.setCurrentText("19200")
        control_layout.addWidget(self.combo_baudrate)
        self.btn_connect = QtWidgets.QPushButton('Open Port')
        self.btn_connect.clicked.connect(self.connect_serial)
        control_layout.addWidget(self.btn_connect)
        self.btn_close = QtWidgets.QPushButton('Close Port')
        self.btn_close.clicked.connect(self.close_serial)
        control_layout.addWidget(self.btn_close)
        self.btn_single = QtWidgets.QPushButton('Single Acquire')
        self.btn_single.clicked.connect(self.single_acquisition)
        control_layout.addWidget(self.btn_single)
        self.btn_continuous = QtWidgets.QPushButton('Continuous Acquire')
        self.btn_continuous.clicked.connect(self.continuous_acquisition)
        control_layout.addWidget(self.btn_continuous)
        self.btn_pause = QtWidgets.QPushButton('Pause')
        self.btn_pause.clicked.connect(self.pause_acquisition)
        control_layout.addWidget(self.btn_pause)
        control_layout.addWidget(QtWidgets.QLabel('Integration Time (ms):'))
        self.spin_integration_time = QtWidgets.QSpinBox()
        self.spin_integration_time.setRange(1, 60000)
        self.spin_integration_time.setValue(100)
        control_layout.addWidget(self.spin_integration_time)
        btn_set_time = QtWidgets.QPushButton('Set')
        btn_set_time.clicked.connect(self.set_integration_time)
        control_layout.addWidget(btn_set_time)
        control_layout.addWidget(QtWidgets.QLabel('Average (1-255):'))
        self.spin_average_count = QtWidgets.QSpinBox()
        self.spin_average_count.setRange(1, 255)
        self.spin_average_count.setValue(1)
        self.spin_average_count.valueChanged.connect(self.set_average_count)
        control_layout.addWidget(self.spin_average_count)
        btn_save = QtWidgets.QPushButton('Save Data')
        btn_save.clicked.connect(self.save_data)
        control_layout.addWidget(btn_save)
        btn_process = QtWidgets.QPushButton('Process and Search Spectra')
        btn_process.clicked.connect(self.toggle_process_panel)
        control_layout.addWidget(btn_process)
        btn_background = QtWidgets.QPushButton('Acquire Background')
        btn_background.clicked.connect(self.acquire_background)
        control_layout.addWidget(btn_background)
        btn_clear_bg = QtWidgets.QPushButton('Clear Background')
        btn_clear_bg.clicked.connect(self.clear_background)
        control_layout.addWidget(btn_clear_bg)
        btn_clear_log = QtWidgets.QPushButton('Clear Log')
        btn_clear_log.clicked.connect(self.clear_log)
        control_layout.addWidget(btn_clear_log)
        self.checkbox_zoom = QtWidgets.QCheckBox('Zoom / Smooth')
        self.checkbox_zoom.stateChanged.connect(self.on_zoom_checkbox_changed)
        control_layout.addWidget(self.checkbox_zoom)
        control_layout.addWidget(QtWidgets.QLabel('DB Search:'))
        self.db_search_edit = QtWidgets.QLineEdit()
        control_layout.addWidget(self.db_search_edit)
        btn_plot_db = QtWidgets.QPushButton('Plot DB Spectrum')
        btn_plot_db.clicked.connect(self.plot_db_spectrum)
        control_layout.addWidget(btn_plot_db)
        btn_set_db = QtWidgets.QPushButton('Set DB Spectrum for processing')
        btn_set_db.clicked.connect(self.set_db_spectrum_forproc)
        control_layout.addWidget(btn_set_db)
        btn_manage_db = QtWidgets.QPushButton('Manage DB...')
        btn_manage_db.clicked.connect(self.toggle_manage_db_panel)
        control_layout.addWidget(btn_manage_db)
        btn_advanced = QtWidgets.QPushButton('Advanced...')
        btn_advanced.clicked.connect(lambda: self.advanced_dock.setVisible(not self.advanced_dock.isVisible()))
        control_layout.addWidget(btn_advanced)
        control_layout.addStretch()
        return control_layout
    def create_manage_db_panel(self):
        self.manage_db_dock = QtWidgets.QDockWidget("Manage Database", self)
        self.manage_db_dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.manage_db_dock)
        self.manage_db_dock.setVisible(False)
        db_widget = QtWidgets.QWidget()
        db_layout = QtWidgets.QVBoxLayout(db_widget)
        self.lbl_current_db = QtWidgets.QLabel("No database loaded")
        self.lbl_current_db.setStyleSheet("color: gray; font-style: italic;")
        db_layout.addWidget(self.lbl_current_db)
        btn_load_db = QtWidgets.QPushButton('Load Database...')
        btn_load_db.clicked.connect(self.load_database_file)
        db_layout.addWidget(btn_load_db)
        btn_reload_db = QtWidgets.QPushButton('Reload Current DB')
        btn_reload_db.clicked.connect(self.reload_current_database)
        db_layout.addWidget(btn_reload_db)
        btn_add_to_db = QtWidgets.QPushButton('Add spectrum to the base')
        btn_add_to_db.clicked.connect(self.add_current_to_db)
        db_layout.addWidget(btn_add_to_db)
        btn_remove_from_db = QtWidgets.QPushButton('Delete from base')
        btn_remove_from_db.clicked.connect(self.remove_from_db)
        db_layout.addWidget(btn_remove_from_db)
        btn_save_db_as = QtWidgets.QPushButton('Save base as...')
        btn_save_db_as.clicked.connect(self.save_db_as)
        db_layout.addWidget(btn_save_db_as)
        btn_new_empty_db = QtWidgets.QPushButton('New empty base')
        btn_new_empty_db.clicked.connect(self.create_new_empty_db)
        db_layout.addWidget(btn_new_empty_db)
        btn_save_as_default = QtWidgets.QPushButton('Save as Default DB')
        btn_save_as_default.clicked.connect(self.save_as_default_database)
        db_layout.addWidget(btn_save_as_default)
        db_layout.addStretch()
        db_scroll = QtWidgets.QScrollArea()
        db_scroll.setWidget(db_widget)
        db_scroll.setWidgetResizable(True)
        db_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        db_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.manage_db_dock.setWidget(db_scroll)
    def create_advanced_panel(self):
        self.advanced_dock = QtWidgets.QDockWidget("Advanced Settings", self)
        self.advanced_dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.advanced_dock)
        self.advanced_dock.setVisible(False)
        adv_widget = QtWidgets.QWidget()
        adv_layout = QtWidgets.QVBoxLayout(adv_widget)
        adv_layout.addWidget(QtWidgets.QLabel('Laser Voltage (mV):'))
        self.spin_laser_voltage = QtWidgets.QSpinBox()
        self.spin_laser_voltage.setRange(0, 5000)
        self.spin_laser_voltage.setValue(0)
        self.spin_laser_voltage.setSingleStep(50)
        adv_layout.addWidget(self.spin_laser_voltage)
        btn_set_voltage = QtWidgets.QPushButton('Set Voltage')
        btn_set_voltage.clicked.connect(self.set_laser_voltage)
        adv_layout.addWidget(btn_set_voltage)
        self.cb_trigger_out = QtWidgets.QCheckBox('Trigger Out HIGH (5V)')
        self.cb_trigger_out.stateChanged.connect(self.set_trigger_out)
        adv_layout.addWidget(self.cb_trigger_out)
        adv_layout.addWidget(QtWidgets.QLabel('Gain (0-255):'))
        self.spin_gain = QtWidgets.QSpinBox()
        self.spin_gain.setRange(0, 255)
        self.spin_gain.setValue(128)
        adv_layout.addWidget(self.spin_gain)
        btn_set_gain = QtWidgets.QPushButton('Set Gain')
        btn_set_gain.clicked.connect(self.set_gain)
        adv_layout.addWidget(btn_set_gain)
        adv_layout.addWidget(QtWidgets.QLabel('Offset:'))
        self.spin_offset = QtWidgets.QSpinBox()
        self.spin_offset.setRange(-255, 255)
        self.spin_offset.setValue(0)
        adv_layout.addWidget(self.spin_offset)
        btn_set_offset = QtWidgets.QPushButton('Set Offset')
        btn_set_offset.clicked.connect(self.set_offset)
        adv_layout.addWidget(btn_set_offset)
        adv_layout.addWidget(QtWidgets.QLabel('Start Wavelength:'))
        self.spin_start_wl = QtWidgets.QSpinBox()
        self.spin_start_wl.setRange(200, 2000)
        self.spin_start_wl.setValue(self.wavelength_min)
        adv_layout.addWidget(self.spin_start_wl)
        adv_layout.addWidget(QtWidgets.QLabel('End Wavelength:'))
        self.spin_end_wl = QtWidgets.QSpinBox()
        self.spin_end_wl.setRange(200, 2000)
        self.spin_end_wl.setValue(self.wavelength_max)
        adv_layout.addWidget(self.spin_end_wl)
        btn_set_range = QtWidgets.QPushButton('Set Range')
        btn_set_range.clicked.connect(self.set_wavelength_range)
        adv_layout.addWidget(btn_set_range)
        adv_layout.addStretch()

        adv_layout.addWidget(QtWidgets.QLabel('Smoothing Level (1-10):'))
        self.spin_smooth = QtWidgets.QSpinBox()
        self.spin_smooth.setRange(1, 10)
        self.spin_smooth.setValue(self.smoothing_level)
        self.spin_smooth.valueChanged.connect(self.set_smoothing_level)
        adv_layout.addWidget(self.spin_smooth)

        btn_set_smooth = QtWidgets.QPushButton('Apply Smoothing')
        btn_set_smooth.clicked.connect(lambda: self.set_smoothing_level(self.spin_smooth.value()))
        adv_layout.addWidget(btn_set_smooth)


        adv_scroll = QtWidgets.QScrollArea()
        adv_scroll.setWidget(adv_widget)
        adv_scroll.setWidgetResizable(True)
        adv_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        adv_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.advanced_dock.setWidget(adv_scroll)

    def set_smoothing_level(self, level):
            if not self.serial_port.is_open:
                self.log("Cannot set smoothing: port not open")
                return
            level = max(1, min(10, int(level)))
            self.send_command_with_data(0x25, level, 0x00)
            self.smoothing_level = level
            self.log(f"Smoothing level set to {level}")


    def set_wavelength_range(self):
        self.wavelength_min = self.spin_start_wl.value()
        self.wavelength_max = self.spin_end_wl.value()
        self.wavelengths = np.linspace(self.wavelength_min, self.wavelength_max, 2048)
        self.log(f"Updated wavelength range: {self.wavelength_min} - {self.wavelength_max}")
        if self.current_spectrum_1 is not None:
            self.spectral_axis = self.get_current_axis()
            self.update_plot(self.spectral_axis, self.current_spectrum_1)
    def get_current_axis(self):
        if self.use_raman:
            initial_shifts = rp.utils.wavelength_to_wavenumber(self.wavelengths, self.exc_wlen_spin.value())
            if self.is_calibrated and self.calib_coeffs_soft:
                a, b, c = self.calib_coeffs_soft
                calibrated_shifts = a * initial_shifts**2 + b * initial_shifts + c
                return calibrated_shifts
            else:
                return initial_shifts
        else:
            return self.wavelengths
        

    def on_to_raman_changed(self, state):
        self.use_raman = state == QtCore.Qt.Checked
        if self.current_spectrum_1 is not None and not self.cur_spectrum_is_db:
            self.spectral_axis = self.get_current_axis()
            self.update_plot(self.spectral_axis, self.current_spectrum_1, zoom=True)
            self.plot_widget.setLabel('bottom', 'Raman shift (cm<sup>-1</sup>)' if self.use_raman else 'Wavelength (nm)')

    def toggle_manage_db_panel(self):
        self.manage_db_dock.setVisible(not self.manage_db_dock.isVisible())
        if self.manage_db_dock.isVisible():
            if self.current_db_path:
                self.lbl_current_db.setText(f"DB: {os.path.basename(self.current_db_path)}")
                self.lbl_current_db.setStyleSheet("color: green;")
            else:
                self.lbl_current_db.setText("No database loaded")
                self.lbl_current_db.setStyleSheet("color: gray; font-style: italic;")
    def create_process_panel(self):
        process_widget = QtWidgets.QWidget()
        process_layout = QtWidgets.QVBoxLayout(process_widget)
        wlen_group = QtWidgets.QGroupBox("Excitation wavelength")
        wlen_layout = QtWidgets.QHBoxLayout()
        self.checkbox_to_raman = QtWidgets.QCheckBox("Convert to Raman Shifts")
        self.checkbox_to_raman.setChecked(self.use_raman)
        self.checkbox_to_raman.stateChanged.connect(self.on_to_raman_changed)
        wlen_layout.addWidget(self.checkbox_to_raman)
        self.exc_wlen_spin = QtWidgets.QSpinBox(self)
        self.exc_wlen_spin.setRange(0, 1000)
        self.exc_wlen_spin.setValue(785)
        self.exc_wlen_spin.setSingleStep(1)
        wlen_layout.addWidget(self.exc_wlen_spin)
        wlen_group.setLayout(wlen_layout)
        process_layout.addWidget(wlen_group)
      
        crop_group = QtWidgets.QGroupBox("Crop Region")
        crop_layout = QtWidgets.QHBoxLayout()
        self.checkbox_crop = QtWidgets.QCheckBox("Enable")
        crop_layout.addWidget(self.checkbox_crop)
        crop_layout.addWidget(QtWidgets.QLabel("Min Shift:"))
        self.spin_crop_min = QtWidgets.QSpinBox()
        self.spin_crop_min.setRange(0, 4000)
        self.spin_crop_min.setValue(0)
        crop_layout.addWidget(self.spin_crop_min)
        crop_layout.addWidget(QtWidgets.QLabel("Max Shift:"))
        self.spin_crop_max = QtWidgets.QSpinBox()
        self.spin_crop_max.setRange(0, 4000)
        self.spin_crop_max.setValue(4000)
        crop_layout.addWidget(self.spin_crop_max)
        crop_group.setLayout(crop_layout)
        process_layout.addWidget(crop_group)
        savgol_group = QtWidgets.QGroupBox("Savitzky-Golay Filter")
        savgol_layout = QtWidgets.QHBoxLayout()
        self.checkbox_savgol = QtWidgets.QCheckBox("Enable")
        savgol_layout.addWidget(self.checkbox_savgol)
        savgol_layout.addWidget(QtWidgets.QLabel("Window Length:"))
        self.spin_savgol_window = QtWidgets.QSpinBox()
        self.spin_savgol_window.setRange(3, 101)
        self.spin_savgol_window.setSingleStep(2)
        self.spin_savgol_window.setValue(7)
        savgol_layout.addWidget(self.spin_savgol_window)
        savgol_layout.addWidget(QtWidgets.QLabel("Poly Order:"))
        self.spin_savgol_poly = QtWidgets.QSpinBox()
        self.spin_savgol_poly.setRange(1, 5)
        self.spin_savgol_poly.setValue(3)
        savgol_layout.addWidget(self.spin_savgol_poly)
        savgol_group.setLayout(savgol_layout)
        process_layout.addWidget(savgol_group)
        baseline_group = QtWidgets.QGroupBox("ASLS Baseline Correction")
        baseline_layout = QtWidgets.QHBoxLayout()
        self.checkbox_asls = QtWidgets.QCheckBox("Enable")
        baseline_layout.addWidget(self.checkbox_asls)
        baseline_group.setLayout(baseline_layout)
        process_layout.addWidget(baseline_group)
        norm_group = QtWidgets.QGroupBox("Normalization")
        norm_layout = QtWidgets.QHBoxLayout()
        self.checkbox_norm = QtWidgets.QCheckBox("Enable")
        norm_layout.addWidget(self.checkbox_norm)
        self.combo_norm_type = QtWidgets.QComboBox()
        self.combo_norm_type.addItems(["MinMax", "Vector"])
        norm_layout.addWidget(self.combo_norm_type)
        norm_group.setLayout(norm_layout)
        process_layout.addWidget(norm_group)
        peaks_group = QtWidgets.QGroupBox("Find Peaks")
        peaks_layout = QtWidgets.QHBoxLayout()
        self.checkbox_peaks = QtWidgets.QCheckBox("Enable")
        peaks_layout.addWidget(self.checkbox_peaks)
        peaks_layout.addWidget(QtWidgets.QLabel("Peak Prominence:"))
        self.spin_peaks_prominence = QtWidgets.QDoubleSpinBox()
        self.spin_peaks_prominence.setRange(0.0, 10000.0)
        self.spin_peaks_prominence.setValue(0.1)
        peaks_layout.addWidget(self.spin_peaks_prominence)
        peaks_layout.addWidget(QtWidgets.QLabel("Peak Width:"))
        self.spin_peaks_width = QtWidgets.QDoubleSpinBox()
        self.spin_peaks_width.setRange(0, 1000)
        self.spin_peaks_width.setValue(2)
        peaks_layout.addWidget(self.spin_peaks_width)
        self.btn_download_peaks = QtWidgets.QPushButton("Download Peaks")
        self.btn_download_peaks.clicked.connect(self.download_peaks)
        self.btn_download_peaks.setEnabled(False)
        peaks_layout.addWidget(self.btn_download_peaks)
        peaks_group.setLayout(peaks_layout)
        process_layout.addWidget(peaks_group)
        self.btn_download_procspectrum = QtWidgets.QPushButton("Download Processing Results")
        self.btn_download_procspectrum.clicked.connect(self.download_processing_results)
        self.btn_download_procspectrum.setEnabled(False)
        process_layout.addWidget(self.btn_download_procspectrum)
        search_group = QtWidgets.QGroupBox("Search Spectra")
        search_layout = QtWidgets.QVBoxLayout()
        self.checkbox_search = QtWidgets.QCheckBox("Enable")
        search_layout.addWidget(self.checkbox_search)
        methods_group = QtWidgets.QGroupBox("Methods")
        methods_layout = QtWidgets.QVBoxLayout()
        self.checkbox_sad = QtWidgets.QCheckBox("SAD")
        self.checkbox_sid = QtWidgets.QCheckBox("SID")
        self.checkbox_mae = QtWidgets.QCheckBox("MAE")
        self.checkbox_mse = QtWidgets.QCheckBox("MSE")
        self.checkbox_iur = QtWidgets.QCheckBox("IUR (Peaks)")
        self.checkbox_spearmanr = QtWidgets.QCheckBox("Spearman R")
        methods_layout.addWidget(self.checkbox_sad)
        methods_layout.addWidget(self.checkbox_sid)
        methods_layout.addWidget(self.checkbox_mae)
        methods_layout.addWidget(self.checkbox_mse)
        methods_layout.addWidget(self.checkbox_iur)
        methods_layout.addWidget(self.checkbox_spearmanr)
        self.method_button_group = QtWidgets.QButtonGroup(methods_group)
        self.method_button_group.setExclusive(True)
        self.method_button_group.addButton(self.checkbox_sad)
        self.method_button_group.addButton(self.checkbox_sid)
        self.method_button_group.addButton(self.checkbox_mae)
        self.method_button_group.addButton(self.checkbox_mse)
        self.method_button_group.addButton(self.checkbox_iur)
        self.method_button_group.addButton(self.checkbox_spearmanr)
        self.checkbox_sad.setChecked(True)
        methods_group.setLayout(methods_layout)
        search_layout.addWidget(methods_group)
        iur_settings_group = QtWidgets.QGroupBox("IUR settings")
        iur_settings_layout = QtWidgets.QHBoxLayout()
        iur_settings_layout.addWidget(QtWidgets.QLabel("DB Peak Prominence:"))
        self.spin_iur_prominence = QtWidgets.QDoubleSpinBox()
        self.spin_iur_prominence.setRange(0.0, 10000.0)
        self.spin_iur_prominence.setValue(0.1)
        iur_settings_layout.addWidget(self.spin_iur_prominence)
        iur_settings_layout.addWidget(QtWidgets.QLabel("DB Peak Width:"))
        self.spin_iur_width = QtWidgets.QDoubleSpinBox()
        self.spin_iur_width.setRange(0, 1000)
        self.spin_iur_width.setValue(2)
        iur_settings_layout.addWidget(self.spin_iur_width)
        iur_settings_layout.addWidget(QtWidgets.QLabel("Tolerance:"))
        self.spin_iur_tol = QtWidgets.QDoubleSpinBox()
        self.spin_iur_tol.setRange(0, 1000)
        self.spin_iur_tol.setValue(30)
        iur_settings_layout.addWidget(self.spin_iur_tol)
        iur_settings_group.setLayout(iur_settings_layout)
        search_layout.addWidget(iur_settings_group)
        search_layout.addWidget(QtWidgets.QLabel("Min spectra axes overlap:"))
        self.spin_min_olap=QtWidgets.QDoubleSpinBox()
        self.spin_min_olap.setRange(1, 3500)
        self.spin_min_olap.setValue(3000)
        search_layout.addWidget(self.spin_min_olap)
        self.checkbox_process_db = QtWidgets.QCheckBox("Preprocess DB spectra")
        self.checkbox_process_db.setChecked(True)
        search_layout.addWidget(self.checkbox_process_db)
        topn_layout = QtWidgets.QHBoxLayout()
        topn_layout.addWidget(QtWidgets.QLabel("Top N:"))
        self.spin_topn = QtWidgets.QSpinBox()
        self.spin_topn.setRange(1, 1000)
        self.spin_topn.setValue(5)
        topn_layout.addWidget(self.spin_topn)
        search_layout.addLayout(topn_layout)
        self.combo_reference = QtWidgets.QComboBox() #n
        self.combo_reference.setEditable(True)                      
        self.combo_reference.setInsertPolicy(QtWidgets.QComboBox.NoInsert)  
        self.combo_reference.setDuplicatesEnabled(False)

     
        completer = self.combo_reference.completer()
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)     
        completer.setFilterMode(QtCore.Qt.MatchContains)            
        completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self.combo_reference.setCompleter(completer)



        self.combo_reference.addItem("None")
        self.combo_reference.currentIndexChanged.connect(self.plot_reference)
        search_layout.addWidget(QtWidgets.QLabel("Plot Reference:"))
        search_layout.addWidget(self.combo_reference)
        search_group.setLayout(search_layout)
        process_layout.addWidget(search_group)
        download_layout = QtWidgets.QHBoxLayout()
        self.btn_download = QtWidgets.QPushButton("Download Search Results")
        self.btn_download.clicked.connect(self.download_search_results)
        self.btn_download.setEnabled(False)
        download_layout.addWidget(self.btn_download)
        process_layout.addLayout(download_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_apply.clicked.connect(self.apply_processing)
        btn_layout.addWidget(self.btn_apply)
        self.btn_revert = QtWidgets.QPushButton("Revert")
        self.btn_revert.clicked.connect(self.revert_processing)
        self.btn_revert.setEnabled(False)
        btn_layout.addWidget(self.btn_revert)
        process_layout.addLayout(btn_layout)
        process_layout.addStretch()
        process_scroll = QtWidgets.QScrollArea()
        process_scroll.setWidget(process_widget)
        process_scroll.setWidgetResizable(True)
        process_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        process_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.process_dock.setWidget(process_scroll)
    def set_db_spectrum_forproc(self):
        self.plot_db_spectrum()
        self.current_spectrum_1=self.rspec.spectral_data
        self.original_spectrum=self.current_spectrum_1
        self.spectral_axis=self.rspec.spectral_axis
        self.processed_spectrum = None
        self.peaks = None
        self.plot_curve_ref.setVisible(False)
        self.update_plot(self.rspec.spectral_axis, self.current_spectrum_1, zoom=True)
        self.plot_widget.setTitle(f"DB Spectrum: {self.rspec_name}")
        self.plot_widget.setLabel('bottom', 'Raman shift (cm<sup>-1</sup>)')
        self.cur_spectrum_is_db=True
        return
  
    def toggle_process_panel(self):
        if self.current_spectrum_1 is None:
            QtWidgets.QMessageBox.information(self, "No Data", "Acquire a spectrum first.")
            return
        self.process_dock.setVisible(not self.process_dock.isVisible())
        if self.process_dock.isVisible():
            self.original_spectrum = self.current_spectrum_1.copy()
            self.update_reference_combo_all()
    def update_serial_ports(self):
        self.combo_ports.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.combo_ports.addItem(port.device)
    def connect_serial(self):
        if self.serial_port.is_open:
            return
        port = self.combo_ports.currentText()
        baudrate = int(self.combo_baudrate.currentText())
        try:
            self.serial_port.port = port
            self.serial_port.baudrate = baudrate
            self.serial_port.open()
            self.status_bar.showMessage(f"Connected to {port}", 3000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Connection Error", str(e))
        self.set_trigger_mode(0)
        self.set_hardware_average(self.average_count)
        initial_unit = self.get_integration_unit()
        if initial_unit is not None:
            self.log(f"Initial device integration unit: {'ms' if initial_unit == 0x00 else 'µs'}")
        initial_time = self.get_integration_time()
        if initial_time is not None:
            self.log(f"Initial device integration time: {initial_time}")
            self.spin_integration_time.setValue(initial_time)
        else:
            self.log("Initial integration time unavailable")
        self.read_gain()
        self.read_offset()
        self.read_smoothing_level()

    # def update_reference_combo_all(self):
    #     """Заполняет combo_reference всеми спектрами из базы"""
    #     self.combo_reference.clear()
    #     self.combo_reference.addItem("None")
        
    #     if self.specdict:
    #         # Сортируем по имени для удобства
    #         sorted_names = sorted(self.specdict.keys())
    #         for name in sorted_names:
    #             self.combo_reference.addItem(name)
        
    #     self.log(f"Reference combo updated with {len(sorted_names)} spectra from database")

    def update_reference_combo_all(self):
        self.combo_reference.clear()
        self.combo_reference.addItem("None")

        if not self.specdict:
            self.log("База пуста")
            return

        # Собираем пары (отображаемое имя, ключ)
        items = []
        for key, data in self.specdict.items():
            display_name = data.get('name', f"Спектр {key}")  # если 'name' нет — fallback
            items.append((display_name, key))

        # Сортируем по отображаемому имени
        items.sort(key=lambda x: x[0])

        for display_name, _ in items:
            self.combo_reference.addItem(display_name)

        self.log(f"Добавлено {len(items)} спектров в список референсов")



    def close_serial(self):
        if self.serial_port.is_open:
            self.pause_acquisition()
            self.serial_port.close()
            self.status_bar.showMessage("Port closed", 2000)
    def single_acquisition(self):
        self.rspec=None
        self.cur_spectrum_is_db=False
        self.plot_curve_ref.clear()
        self.plot_curve_ref.setVisible(False)
        self.plot_widget.setYRange(0, 65535)
        self.plot_widget.setXRange(self.wavelength_min, self.wavelength_max)
        self.revert_processing()
        self.btn_download_peaks.setEnabled(False)
        self.btn_revert.setEnabled(False)
        self.btn_download.setEnabled(False)
        self.btn_download_procspectrum.setEnabled(False)
        if self.peaks is not None or self.processed_spectrum is not None:
            self.processed_spectrum = None
            self.peaks = None
            for item in self.plot_widget.items():
                if isinstance(item, pg.InfiniteLine):
                    self.plot_widget.removeItem(item)
            self.plot_widget.setYRange(0, 65535)
            self.plot_widget.setXRange(self.wavelength_min, self.wavelength_max)
            self.processed_spectrum = None
            self.peaks = None
            self.plot_widget.setTitle("Raw Spectrum")
        if not self.serial_port.is_open:
            QtWidgets.QMessageBox.warning(self, "Error", "Serial port not open!")
            return
      
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.progress_bar.show()
        QtWidgets.QApplication.processEvents()
        self.send_command(0x01)
        data_1 = self.read_spectral_data().astype(np.float64)
        time.sleep(0.15)
        QtWidgets.QApplication.processEvents()
        self.progress_bar.hide()
        if len(data_1) != 2048:
            data_1 = np.zeros(2048)
        if self.background_spectrum is not None:
            data_1 = data_1 - self.background_spectrum
        self.current_spectrum_1 = data_1.copy()
        self.original_spectrum = data_1.copy()
        self.spectral_axis = self.get_current_axis()
        self.update_plot(self.spectral_axis, data_1)
        self.update_fps()
        self.collection_count += 1
        self.collection_label.setText(f"Acquisitions: {self.collection_count}")
        self.on_zoom_checkbox_changed()
    def continuous_acquisition(self):
        self.rspec=None
        self.cur_spectrum_is_db=False
        self.plot_curve_ref.clear()
        self.plot_curve_ref.setVisible(False)
        self.plot_widget.setYRange(0, 65535)
        self.plot_widget.setXRange(self.wavelength_min, self.wavelength_max)
        self.revert_processing()
        self.btn_download_peaks.setEnabled(False)
        self.btn_revert.setEnabled(False)
        self.btn_download.setEnabled(False)
        self.btn_download_procspectrum.setEnabled(False)
        if self.peaks is not None or self.processed_spectrum is not None:
            for item in self.plot_widget.items():
                if isinstance(item, pg.InfiniteLine):
                    self.plot_widget.removeItem(item)
            self.plot_widget.setYRange(0, 65535)
            self.plot_widget.setXRange(self.wavelength_min, self.wavelength_max)
            self.processed_spectrum = None
            self.peaks = None
            self.plot_widget.setTitle("Raw Spectrum")
        if not self.serial_port.is_open:
            return
        self.continuous_mode = True
        self.send_command(0x02)
        self.collection_count = 0
        self.frame_counter = 0
        while self.continuous_mode:
            self.single_acquisition_logic()
            QtWidgets.QApplication.processEvents()
    def single_acquisition_logic(self):
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.progress_bar.show()
        QtWidgets.QApplication.processEvents()
        data_1 = self.read_spectral_data()
        if len(data_1) != 2048:
            data_1 = np.zeros(2048)
        if self.background_spectrum is not None:
            data_1 = np.maximum(data_1 - self.background_spectrum, 0)
        self.current_spectrum_1 = data_1.copy()
        self.original_spectrum = data_1.copy()
        self.spectral_axis = self.get_current_axis()
        self.update_plot(self.spectral_axis, data_1)
        self.update_fps()
        self.collection_count += 1
        self.collection_label.setText(f"Acquisitions: {self.collection_count}")
        self.frame_counter += 1
        self.on_zoom_checkbox_changed()
        self.progress_bar.hide()
    def pause_acquisition(self):
        self.continuous_mode = False
        self.send_command(0x06)
    def send_command(self, cmd_byte):
        cmd = bytearray([0x81, cmd_byte, 0x00, 0x00])
        crc = sum(cmd) & 0xFF
        cmd.append(crc)
        self.serial_port.write(cmd)
    def read_reply(self, expected_cmd, timeout=1):
        start = time.time()
        buf = bytearray()
        while time.time() - start < timeout:
            if self.serial_port.in_waiting:
                buf += self.serial_port.read(self.serial_port.in_waiting)
                if len(buf) >= 5:
                    pkt = buf[:5]
                    if pkt[0] == 0x81 and pkt[1] == expected_cmd:
                        crc = (pkt[0] + pkt[1] + pkt[2] + pkt[3]) & 0xFF
                        if crc == pkt[4]:
                            return pkt
                    buf = buf[1:]
            time.sleep(0.001)
        return None
    def get_integration_time(self):
        time.sleep(0.3)
        self.serial_port.reset_input_buffer()
        cmd = bytearray([0x81, 0x0A, 0x00, 0x00])
        cmd.append(sum(cmd) & 0xFF)
        self.serial_port.write(cmd)
        reply = self.read_reply(0x02)
        if reply is None:
            self.log("GET integration time: no response")
            return None
        time_val = (reply[2] << 8) | reply[3]
        self.log(f"Device integration time value = {time_val}")
        return time_val
    def get_integration_unit(self):
        time.sleep(0.3)
        self.serial_port.reset_input_buffer()
        cmd = bytearray([0x81, 0x12, 0x00, 0x00])
        cmd.append(sum(cmd) & 0xFF)
        self.serial_port.write(cmd)
        reply = self.read_reply(0x12)
        if reply is None:
            self.log("GET integration unit: no response")
            return None
        unit = reply[2]
        unit_str = "ms" if unit == 0x00 else "µs"
        self.log(f"Device integration unit = {unit_str}")
        return unit
    def set_trigger_mode(self, mode):
        self.send_command_with_data(0x07, mode, 0x00)
        self.log(f"Trigger mode set to {mode} (0: soft, 1: external continuous, 2: external monopulse)")
    def set_integration_time(self):
        if not self.serial_port.is_open:
            return
        self.send_command(0x06)
        self.log("Acquisition STOP sent")
        time.sleep(0.3)
        self.serial_port.reset_input_buffer()
        current_unit = self.get_integration_unit()
        if current_unit is not None:
            self.log(f"Current integration unit before set: {'ms' if current_unit == 0x00 else 'µs'}")
        self.send_command_with_data(0x11, 0x00, 0x00)
        time.sleep(0.05)
        new_unit = self.get_integration_unit()
        if new_unit is not None:
            self.log(f"CONFIRMED integration unit after set: {'ms' if new_unit == 0x00 else 'µs'}")
        else:
            self.log("Unit readback unavailable")
        time_ms = self.spin_integration_time.value()
        hi = (time_ms >> 8) & 0xFF
        lo = time_ms & 0xFF
        self.send_command_with_data(0x03, hi, lo)
        self.log(f"Integration time SET request = {time_ms} ms")
        time.sleep(0.05)
        val = self.get_integration_time()
        if val is not None:
            self.log(f"CONFIRMED by device: {val} ms")
        else:
            self.log("Integration time readback unavailable (firmware busy)")
     
        time_ms = self.spin_integration_time.value()
        self.serial_port.timeout = (time_ms / 1000.0) + 10.0
        self.log(f"Serial timeout updated to {self.serial_port.timeout} seconds")
        self.save_parameters()
    def save_parameters(self):
        self.send_command_with_data(0x22, 0x00, 0x00)
        self.log("Parameters saved to device")
    def send_command_with_data(self, cmd, d1, d2):
        cmd = bytearray([0x81, cmd, d1, d2])
        crc = sum(cmd) & 0xFF
        cmd.append(crc)
        self.serial_port.write(cmd)
        self.log(f"Sent command {cmd}")
    def set_average_count(self, value):
        self.average_count = value
        self.set_hardware_average(value)
    def read_spectral_data(self):
        if not self.serial_port.is_open:
            return np.zeros(2048)
        head = self.serial_port.read(5)
        if len(head) != 5 or head[0] != 0x81 or head[1] != 0x01 or head[4] != 0x00:
            self.log(f"Invalid head: {head}")
            return np.zeros(2048)
        length = (head[2] << 8) | head[3]
        self.log(f"Data length from head: {length}")
        data = self.serial_port.read(length + 2)
        if len(data) != length + 2:
            self.log(f"Incomplete data: expected {length + 2}, got {len(data)}")
            return np.zeros(2048)
        pixel_data = data[:-2]
        crc_received = (data[-2] << 8) | data[-1]
        data_1 = pixel_data[:min(4096, length)]
        spectral_data_1 = np.frombuffer(data_1, dtype='>u2') if len(data_1) == 4096 else np.zeros(2048)
        return spectral_data_1
    def update_plot(self, x, y, zoom=False):
        # Interpolate to uniform axis if Raman and non-uniform to avoid visual broadening
#        if self.use_raman:
#            diff = np.diff(x)
#            if len(np.unique(diff.round(decimals=4))) > 1:  # non-uniform
#                uniform_x = np.linspace(np.min(x), np.max(x), len(x))
#                interp_func = interp1d(x, y, kind='linear', fill_value='extrapolate')
#                y = interp_func(uniform_x)
#                x = uniform_x
#                self.log("Interpolated spectrum to uniform Raman shift axis for plotting")
        self.plot_curve_1.setData(x, y)
        if self.checkbox_zoom.isChecked() or zoom==True:
            min_y = np.min(y)
            max_y = np.max(y) * 1.05
            self.plot_widget.setYRange(min_y, max_y)
            self.plot_widget.setXRange(np.min(x), np.max(x))
        else:
            self.plot_widget.setYRange(0, 65535)
        if np.max(x) > 1500:
            self.plot_widget.setLabel('bottom', 'Raman shift (cm<sup>-1</sup>)')
        else:
            self.plot_widget.setLabel('bottom', 'Wavelength (nm)')
        for line in self.peak_lines:
            self.plot_widget.removeItem(line)
        for label in self.peak_labels:
            self.plot_widget.removeItem(label)
        self.peak_lines = []
        self.peak_labels = []
        if self.peaks is not None:
            for peak in self.peaks:
                pos = x[peak]
                line = pg.InfiniteLine(pos=pos, angle=90, pen=pg.mkPen('r', style=QtCore.Qt.DashLine))
                self.plot_widget.addItem(line)
                self.peak_lines.append(line)
                label_text = str(int(pos))
                text = pg.TextItem(label_text, anchor=(0.5, 1.1), color='r')
                text.setPos(pos, y[peak])
                self.plot_widget.addItem(text)
                self.peak_labels.append(text)
    def apply_processing(self):
        if self.original_spectrum is None:
            return
        spectrum = np.array(self.original_spectrum.copy())
        if self.cur_spectrum_is_db:
            specax = self.spectral_axis
        else:
            specax = self.get_current_axis()
        robj = rp.SpectralContainer(np.array([spectrum]), specax)
        self.log(f'Robj specax: {robj.spectral_axis}')
        proclist = []
        if self.checkbox_crop.isChecked():
            cropreg = (self.spin_crop_min.value(), self.spin_crop_max.value())
            proclist.append(rp.preprocessing.misc.Cropper(region=cropreg))
        if self.checkbox_savgol.isChecked():
            window = self.spin_savgol_window.value()
            poly = self.spin_savgol_poly.value()
            proclist.append(rp.preprocessing.denoise.SavGol(window_length=window, polyorder=poly))
        if self.checkbox_asls.isChecked():
            proclist.append(rp.preprocessing.baseline.ASLS())
        if self.checkbox_norm.isChecked():
            norm_type = self.combo_norm_type.currentText()
            if norm_type == "MinMax":
                proclist.append(rp.preprocessing.normalise.MinMax())
            elif norm_type == "Vector":
                proclist.append(rp.preprocessing.normalise.Vector())
        self.preprocessing_pipeline = rp.preprocessing.Pipeline(proclist)
        self.preprocessed_robj = self.preprocessing_pipeline.apply(robj)
        self.log(f'Preproc robj specax: {self.preprocessed_robj.spectral_axis}')
        self.peaks = None
        self.btn_download_peaks.setEnabled(False)
        if self.checkbox_peaks.isChecked():
            prominence = self.spin_peaks_prominence.value()
            width = int(self.spin_peaks_width.value())
            peaks, props = find_peaks(
                self.preprocessed_robj.spectral_data[0],
                prominence=prominence,
                width=width
            )
            self.peaks = peaks
            self.peakshifts = self.preprocessed_robj.spectral_axis[peaks]
            self.log(f"Peaks found: {self.peaks}")
            self.btn_download_peaks.setEnabled(True)
        self.processed_spectrum = self.preprocessed_robj.spectral_data[0]
        self.current_spectrum_1 = self.processed_spectrum
        self.spectral_axis = self.preprocessed_robj.spectral_axis
        self.log(f"Preprocessed axis {self.spectral_axis}")
        self.log(f"Preprocessed spectrum {self.processed_spectrum}")
        self.log(f"Spectral data {self.preprocessed_robj.spectral_data}")
        self.log(f"{len(self.spectral_axis) == len(self.processed_spectrum)}")
        if max(self.processed_spectrum) <= 1:
            self.update_plot(self.spectral_axis, self.processed_spectrum * 100, zoom=True)
        else:
            self.update_plot(self.spectral_axis, self.processed_spectrum, zoom=True)
        self.plot_widget.setTitle("Processed Spectrum")
        if self.checkbox_search.isChecked():
            if self.specdict is None:
                self.log("No database loaded - skipping search")
                return
            progress = QtWidgets.QProgressDialog("Searching database...", "Cancel", 0, len(self.specdict), self)
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()
            if self.checkbox_sad.isChecked():
                method = 'sad'
            elif self.checkbox_sid.isChecked():
                method = 'sid'
            elif self.checkbox_mae.isChecked():
                method = 'mae'
            elif self.checkbox_mse.isChecked():
                method = 'mse'
            elif self.checkbox_iur.isChecked():
                method = 'iur'
            topn = self.spin_topn.value()
            self.log('Started search')
            self.searchres = self.run_dbsearch_rbase(robj=self.preprocessed_robj, rbase_specdict=self.specdict, nleads=topn, metric=method, progress=progress)
            self.log('Search results ready')
            if self.searchres.empty:
                self.log('Search canceled or no results')
            else:
                self.log(self.searchres.head().drop(['aligned_intensity_comp', 'spectral_axis_comp'], axis=1))
            self.combo_reference.clear()
            self.combo_reference.addItem("None")
            for idx, row in self.searchres.iterrows():
                self.combo_reference.addItem(row['component'])
            self.btn_download.setEnabled(True)
        self.btn_download_procspectrum.setEnabled(True)
        self.btn_revert.setEnabled(True)
    # def plot_reference(self, index):
    #     if index == 0: # "None"
    #         self.plot_curve_ref.clear()
    #         self.plot_curve_ref.setVisible(False)
    #         return
    #     if self.searchres is None:
    #         return
    #     selected_name = self.combo_reference.currentText()
    #     matching_rows = self.searchres[self.searchres['component'] == selected_name]
    #     if matching_rows.empty:
    #         self.log(f"No data for {selected_name}")
    #         return
    #     row = matching_rows.iloc[0]
    #     rbid = row['id']
    #     rspec_data = row['aligned_intensity_comp']
    #     rspec_axis = row['spectral_axis_comp']
    #     if np.max(rspec_data) <= 1 and np.max(self.current_spectrum_1) <= 1:
    #         rspec_data *= 100
    #     self.plot_curve_ref.setData(rspec_axis, rspec_data)
    #     self.plot_curve_ref.setVisible(True)
    # def plot_reference(self, index):
    #     if index == 0:  # "None"
    #         self.plot_curve_ref.clear()
    #         self.plot_curve_ref.setVisible(False)
    #         self.plot_widget.setTitle("")  # убираем лишний заголовок
    #         return

    #     selected_name = self.combo_reference.currentText()
        
    #     # Проверяем, есть ли такой спектр в базе
    #     if selected_name in self.specdict:
    #         rspec = self.specdict[selected_name]['spectrum']
    #         axis = rspec.spectral_axis
    #         data = rspec.spectral_data.copy()
            
    #         if np.max(data) <= 1 and np.max(self.current_spectrum_1) <= 1:
    #             data *= 100
            
    #         self.plot_curve_ref.setData(axis, data)
    #         self.plot_curve_ref.setVisible(True)
            
    #         # Улучшаем отображение: подстраиваем масштаб под оба спектра
    #         self.plot_widget.setXRange(min(np.min(axis), np.min(self.spectral_axis)),
    #                                 max(np.max(axis), np.max(self.spectral_axis)))
            
    #         max_y = max(np.max(self.current_spectrum_1), np.max(data)) * 1.05
    #         self.plot_widget.setYRange(0, max_y)
            
    #         self.plot_widget.setTitle(f"Reference: {selected_name}")
    #         self.log(f"Plotted reference spectrum: {selected_name}")
    #     else:
    #         self.log(f"Спектр '{selected_name}' не найден в базе")

    def plot_reference(self, index):
        if index == 0:
            self.plot_curve_ref.clear()
            self.plot_curve_ref.setVisible(False)
            return

        selected_display = self.combo_reference.currentText()

        # Ищем по полю 'name'
        for key, data in self.specdict.items():
            if data.get('name', f"Спектр {key}") == selected_display:
                rspec = data['spectrum']
                axis = rspec.spectral_axis
                intensity = rspec.spectral_data.copy()

                if np.max(intensity) <= 1 and np.max(self.current_spectrum_1) <= 1:
                    intensity *= 100

                self.plot_curve_ref.setData(axis, intensity)
                self.plot_curve_ref.setVisible(True)
                return

        self.log(f"Не найден спектр '{selected_display}'")

    def revert_processing(self):
        if self.original_spectrum is not None:
            self.current_spectrum_1 = self.original_spectrum.copy()
            self.processed_spectrum = None
            self.peaks = None
            self.btn_download_peaks.setEnabled(False)
            if not self.cur_spectrum_is_db:
                self.spectral_axis = self.get_current_axis()
                self.update_plot(self.spectral_axis, self.current_spectrum_1)
                self.plot_widget.setTitle("")
                for line in self.peak_lines:
                    self.plot_widget.removeItem(line)
                for label in self.peak_labels:
                    self.plot_widget.removeItem(label)
                self.peak_lines = []
                self.peak_labels = []
                self.plot_widget.setYRange(0, 65535)
                self.plot_widget.setXRange(self.wavelength_min, self.wavelength_max)
            else:
                self.update_plot(self.rspec.spectral_axis, self.current_spectrum_1)
                self.plot_widget.setTitle("DB Spectrum")
                self.plot_widget.setLabel('bottom', 'Raman shift (cm<sup>-1</sup>)')
                self.cur_spectrum_is_db = True
            self.plot_curve_ref.clear()
            self.plot_curve_ref.setVisible(False)
            self.combo_reference.setCurrentIndex(0)
            self.btn_download_peaks.setEnabled(False)
            self.btn_revert.setEnabled(False)
            self.btn_download.setEnabled(False)
            self.btn_download_procspectrum.setEnabled(False)
    def download_peaks(self):
        if self.peaks is None or self.processed_spectrum is None:
            self.log("No peak data present")
            return
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Peaks', '', 'CSV Files (*.csv)')
        if file_name:
            data = pd.DataFrame.from_dict({'Raman Shifts': self.peakshifts,
                                           'Intensities': self.preprocessed_robj.spectral_data[0][np.where(np.isin(self.preprocessed_robj.spectral_axis, self.peakshifts))]})
            data.to_csv(file_name, index=False)
            return
    def download_processing_results(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Results', '', 'CSV Files (*.csv)')
        data = pd.DataFrame.from_dict({'Raman Shifts': self.preprocessed_robj.spectral_axis,
                                       'Intensities': self.preprocessed_robj.spectral_data[0]})
        data.to_csv(file_name, index=False)
        return
    def download_search_results(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Results', '', 'CSV Files (*.csv)')
        self.searchres_fmt = copy.deepcopy(self.searchres)
        self.searchres_fmt['aligned_intensity_comp'] = '; '.join(self.searchres_fmt['aligned_intensity_comp'].astype(str))
        self.searchres_fmt['spectral_axis_comp'] = '; '.join(self.searchres_fmt['spectral_axis_comp'].astype(str))
        self.searchres.to_csv(file_name, index=False)
        return
    def set_hardware_average(self, count):
        if not self.serial_port.is_open:
            return
        count = max(1, min(255, int(count)))
        self.send_command_with_data(0x0c, count, 0x00) # 0x0c = set average times
        self.log(f"Hardware average set to {count}")
    def save_data(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Data', '', 'CSV Files (*.csv)')
        if file_name:
            x, y1 = self.plot_curve_1.getData()
            if x is not None and y1 is not None:
                data = np.column_stack((x, y1))
                header = 'x,Intensity'
                np.savetxt(file_name, data, delimiter=',', header=header, comments='')
    def update_fps(self):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1:
            fps = self.frame_count / elapsed
            self.fps_label.setText(f"FPS: {int(fps)}")
            self.frame_count = 0
            self.start_time = time.time()
    def on_zoom_checkbox_changed(self):
        if self.current_spectrum_1 is not None:
            x = self.spectral_axis
            self.update_plot(x, self.current_spectrum_1)
    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_widget.appendPlainText(f"[{timestamp}] {message}")
    def acquire_background(self):
        if not self.serial_port.is_open:
            QtWidgets.QMessageBox.warning(self, "Error", "Port is not opened!")
            return
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        QtWidgets.QApplication.processEvents()
        try:
            self.send_command(0x01)
            data = self.read_spectral_data().astype(np.float64)
          
            if len(data) == 2048:
                self.background_spectrum = data.copy()
                self.log("Background is acquired")
            else:
                self.log("Background acquisition failed")
        finally:
            self.progress_bar.hide()
    def clear_background(self):
        self.background_spectrum = None
        self.log("Background cleared")
    def clear_log(self):
        self.log_widget.clear()
    def run_dbsearch_rbase(self, robj, rbase_specdict, nleads=10, metric='sad', progress=None):
        results = []
        ind = 0
        sres = []
        i = 0 # counter for processEvents
        total_items = len(rbase_specdict)
        for rbid, d in rbase_specdict.items():
            if progress is not None:
                progress.setValue(i)
                if progress.wasCanceled():
                    self.log("Search canceled by user")
                    return pd.DataFrame() # Return empty DataFrame on cancel
            try:
                rspec = d['spectrum']
                name = d['name']
                url = d['url']
                ident = d['identifier']
                if self.checkbox_process_db.isChecked():
                    rspec = self.preprocessing_pipeline.apply(rspec)
                olap = min(robj.spectral_axis.max(), rspec.spectral_axis.max()) - max(robj.spectral_axis.min(), rspec.spectral_axis.min())
                if olap > self.spin_min_olap.value():
                    common_axis = np.linspace(
                        max(robj.spectral_axis.min(), rspec.spectral_axis.min()),
                        min(robj.spectral_axis.max(), rspec.spectral_axis.max()),
                        2000
                    )
                    if len(common_axis) == 0:
                        i += 1
                        continue
                    query_data = robj.spectral_data[0] if robj.spectral_data.ndim > 1 else robj.spectral_data
                    ref_data = rspec.spectral_data[0] if rspec.spectral_data.ndim > 1 else rspec.spectral_data
                    interp1 = interp1d(robj.spectral_axis, query_data, kind='linear', fill_value='extrapolate')
                    interp2 = interp1d(rspec.spectral_axis, ref_data, kind='linear', fill_value='extrapolate')
                    aligned_intensity1 = interp1(common_axis)
                    aligned_intensity2 = interp2(common_axis)
                    if np.any(np.isnan(aligned_intensity1)) or np.any(np.isnan(aligned_intensity2)):
                        i += 1
                        continue
                    if np.linalg.norm(aligned_intensity1) == 0 or np.linalg.norm(aligned_intensity2) == 0:
                        i += 1
                        continue
                    if metric == 'mae':
                        s = rp.metrics.MAE(aligned_intensity1, aligned_intensity2)
                    elif metric == 'mse':
                        s = rp.metrics.MSE(aligned_intensity1, aligned_intensity2)
                    elif metric == 'sad':
                        s = rp.metrics.SAD(aligned_intensity1, aligned_intensity2)
                    elif metric == 'sid':
                        s = rp.metrics.SID(aligned_intensity1, aligned_intensity2)
                    elif metric == 'iur':
                        q_max = np.max(aligned_intensity1)
                        r_max = np.max(aligned_intensity2)
                        q = aligned_intensity1 / q_max if q_max > 0 else aligned_intensity1
                        r = aligned_intensity2 / r_max if r_max > 0 else aligned_intensity2
                        delta = common_axis[1] - common_axis[0] if len(common_axis) > 1 else 1.0
                        width_samples_q = self.spin_peaks_width.value() / delta if delta > 0 else 1.0
                        width_samples_db = self.spin_iur_width.value() / delta if delta > 0 else 1.0
                        peaks_q_idx = find_peaks(q, prominence=self.spin_peaks_prominence.value(), width=width_samples_q)[0]
                        peaks_r_idx = find_peaks(r, prominence=self.spin_iur_prominence.value(), width=width_samples_db)[0]
                        peaks_q_pos = common_axis[peaks_q_idx]
                        peaks_r_pos = common_axis[peaks_r_idx]
                        len_q = len(peaks_q_pos)
                        len_r = len(peaks_r_pos)
                        if len_q == 0 and len_r == 0:
                            s = 0.0 # Perfect match if no peaks in both
                        else:
                            i_p, j_p, matched = 0, 0, 0
                            while i_p < len_q and j_p < len_r:
                                if abs(peaks_q_pos[i_p] - peaks_r_pos[j_p]) <= self.spin_iur_tol.value():
                                    matched += 1
                                    i_p += 1
                                    j_p += 1
                                elif peaks_q_pos[i_p] < peaks_r_pos[j_p]:
                                    i_p += 1
                                else:
                                    j_p += 1
                            total = len_q + len_r
                            iou = matched / (total - matched) if (total - matched) != 0 else 0
                            if matched < 1.0:
                                iou = 0.0
                                s = 1.0
                            else:
                                s = 1.0 - iou # Convert to distance (lower better)
                    sres.append({'component': name, 'url': url, 'id': rbid, 'identifier': ident, 'distance_score': s, 'source': 'rbase', 'aligned_intensity_comp': aligned_intensity2,
                                 'spectral_axis_comp': common_axis})
                i += 1
                if i % 10 == 0:
                    QtWidgets.QApplication.processEvents()
            except Exception as e:
                print(e)
                results.append(None)
                i += 1
        if progress is not None:
            progress.setValue(total_items)
        if not sres:
            self.log("No valid search results found.")
            return pd.DataFrame()
        sdf = pd.DataFrame(sres).sort_values('distance_score', ascending=True)[:nleads].reset_index(drop=True)
        sdf['metric'] = metric
        return sdf
    def plot_db_spectrum(self):
        if self.specdict is None:
            self.log("No database loaded")
            return
        query = self.db_search_edit.text().strip().lower()
        if not query:
            self.log("Enter search term")
            return
        matches = []
        for rbid, d in self.specdict.items():
            try:
                if query in d['name'].lower():
                    matches.append((d['name'], d['spectrum']))
            except Exception as e:
                pass
        if not matches:
            self.log("No matches found")
            return
        if len(matches) > 1:
            self.log(f"{len(matches)} matches found:")
            self.log(f'{"; ".join([i[0] for i in matches])}')
            self.log(f"Plotting first: {matches[0][0]}")
        name, rspec = matches[0]
        axis = rspec.spectral_axis
        data = rspec.spectral_data.copy()
        self.rspec = rspec
        self.rspec_name = name
        if np.max(data) <= 1:
            data *= 100
        self.plot_curve_ref.setData(axis, data)
        self.plot_curve_ref.setVisible(True)
        self.plot_widget.setLabel('bottom', 'Raman shift (cm<sup>-1</sup>)')
        min_y = np.min(data)
        max_y = np.max(data) * 1.05
        self.plot_widget.setYRange(min_y, max_y)
        self.plot_widget.setXRange(np.min(axis), np.max(axis))
        self.log(f"DB spectrum: {name}")
    def set_laser_voltage(self):
        if not self.serial_port.is_open:
            self.log("Port is not opened")
            return
        mV = self.spin_laser_voltage.value()
        hi = (mV >> 8) & 0xFF
        lo = mV & 0xFF
        self.send_command_with_data(0x0D, hi, lo)
        self.log(f"Analog output (laser voltage) → {mV} mV")
    def set_trigger_out(self, state):
        if not self.serial_port.is_open:
            self.log("Port is not opened")
            return
        value = 1 if state == QtCore.Qt.Checked else 0
        self.send_command_with_data(0x10, value, 0x00)
        self.log(f"Trigger Out → {'HIGH (5V)' if value else 'LOW (0V)'}")
    def set_gain(self):
        if not self.serial_port.is_open:
            return
        gain = self.spin_gain.value()
        self.send_command_with_data(0x04, gain, 0x00)
        self.log(f"Gain is set → {gain}")
    def set_offset(self):
        if not self.serial_port.is_open:
            return
        offset = self.spin_offset.value()
        data = abs(offset) & 0xFF
        sign = 0x01 if offset >= 0 else 0x00
        self.send_command_with_data(0x05, data, sign)
        self.log(f"Offset is set → {offset} (data={data}, sign={sign})")
    def read_gain(self):
        self.send_command_with_data(0x23, 0x00, 0x00) # запрос
        reply = self.read_reply(0x23, timeout=1.0)
        if reply and len(reply) >= 3:
            gain = reply[2]
            self.spin_gain.setValue(gain)
            self.log(f"Current gain = {gain}")
    def read_offset(self):
        self.send_command_with_data(0x24, 0x00, 0x00)
        reply = self.read_reply(0x24, timeout=1.0)
        if reply and len(reply) >= 4:
            data = reply[2]
            sign = reply[3]
            offset = data if sign == 0x01 else -data
            self.spin_offset.setValue(offset)
            self.log(f"Current offset = {offset}")
    def add_current_to_db(self):
        if self.current_spectrum_1 is None:
            self.log("No spectrum to add")
            return
        if self.specdict is None:
            self.specdict = {}
        name, ok = QtWidgets.QInputDialog.getText(self, "Spectrum name",
                                                "Enter name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in self.specdict:
            reply = QtWidgets.QMessageBox.question(self, "Replace?",
                f"Spectrum '{name}' already exists. Replace?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.No:
                return
        axis = self.spectral_axis
        intensity = self.current_spectrum_1.copy()
        rspec = rp.Spectrum(intensity, axis)
        self.specdict[name] = {
            'name': name,
            'spectrum': rspec,
            'url': '',
            'identifier': name,
        }
        self.update_reference_combo_all()
        self.log(f"Spectrum '{name}' is added to the base (total n: {len(self.specdict)})")
    def remove_from_db(self):
        if self.specdict is None or not self.specdict:
            self.log("The base is empty")
            return
        names = list(self.specdict.keys())
        name, ok = QtWidgets.QInputDialog.getItem(self, "Deleting a spectrum",
            "Choose the spectrum to delete:", names, 0, False)
        if ok and name:
            del self.specdict[name]
            self.log(f"Spectrum '{name}' is removed. Remaining: {len(self.specdict)}")

        self.update_reference_combo_all()
    def save_db_as(self):
        if self.specdict is None:
            self.log("Base is not loaded")
            return
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self,
            "Save spectra db", "", "Pickle (*.pkl)")
        if file_name:
            try:
                with open(file_name, "wb") as f:
                    pickle.dump(self.specdict, f)
                self.log(f"Base is saved: {file_name}")
            except Exception as e:
                self.log(f"Error saving base: {e}")
    def create_new_empty_db(self):
        reply = QtWidgets.QMessageBox.question(self, "Create empty base?",
            "Current base will be replaced with an empty base. Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.specdict = {}
            self.log("New spectra base is created")

        self.update_reference_combo_all()
    def load_database_file(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Spectral Database",
            "",
            "Pickle Files (*.pkl);;All Files (*)"
        )
        if not file_name:
            return
        try:
            with open(file_name, "rb") as f:
                loaded_dict = pickle.load(f)
          
            if not isinstance(loaded_dict, dict):
                raise ValueError("Loaded file does not contain a dictionary")
            self.specdict = loaded_dict
            self.current_db_path = file_name
            self.searchres = None # reset previous search
            self.combo_reference.clear()
            self.combo_reference.addItem("None")
            self.log(f"Database loaded successfully: {file_name} ({len(self.specdict)} entries)")
            self.lbl_current_db.setText(f"DB: {os.path.basename(file_name)}")
            self.lbl_current_db.setStyleSheet("color: green;")
            self.update_reference_combo_all()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", f"Failed to load database:\n{str(e)}")
            self.log(f"Database load failed: {e}")
    def reload_current_database(self):
        if not self.current_db_path or not os.path.exists(self.current_db_path):
            QtWidgets.QMessageBox.information(self, "No Current DB",
                "No current database file set or file missing.\nPlease load one first.")
            return
        try:
            with open(self.current_db_path, "rb") as f:
                self.specdict = pickle.load(f)
          
            self.searchres = None
            self.combo_reference.clear()
            self.combo_reference.addItem("None")
            self.log(f"Reloaded current database: {self.current_db_path} ({len(self.specdict)} entries)")
            self.lbl_current_db.setText(f"DB: {os.path.basename(self.current_db_path)}")
            self.lbl_current_db.setStyleSheet("color: green;")
            self.update_reference_combo_all()
        except Exception as e:
            self.log(f"Failed to reload database: {e}")
            QtWidgets.QMessageBox.warning(self, "Reload Error", str(e))
    def save_as_default_database(self):
        if self.specdict is None or len(self.specdict) == 0:
            self.log("No database loaded or database is empty — nothing to save as default")
            return
        default_path = "rbase_specdictcur.pkl"
        if os.path.exists(default_path):
            reply = QtWidgets.QMessageBox.question(
                self,
                "Overwrite Default Database?",
                f"The file '{default_path}' already exists.\nOverwrite it with the current database?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            if reply != QtWidgets.QMessageBox.Yes:
                self.log("Save as default canceled by user")
                return
        try:
            with open(default_path, "wb") as f:
                pickle.dump(self.specdict, f)
          
            self.current_db_path = default_path
          
            self.log(f"Successfully saved current database as default: {default_path} ({len(self.specdict)} entries)")
            self.lbl_current_db.setText(f"DB: {os.path.basename(default_path)} (default)")
            self.lbl_current_db.setStyleSheet("color: green;")
        except Exception as e:
            self.log(f"Failed to save as default database: {e}")
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Could not save default database:\n{str(e)}")
    def read_calibration_group(self, group: int):
        if not self.serial_port.is_open:
            self.log("Port not open")
            return None
        self.send_command_with_data(0x29, group, 0x00)
        time.sleep(0.1)
        raw = self.serial_port.read(5 + 1 + 64 + 1) # conservative
        if len(raw) < 70 or raw[0] != 0x81 or raw[1] != 0x29 or raw[2] != group or raw[3] != 0x40:
            self.log(f"Invalid reply for group {group}")
            return None
        data64 = raw[5:5+64]
        strings = []
        pos = 0
        for _ in range(4):
            end = data64.find(b'\x00', pos)
            if end == -1: end = 16
            s = data64[pos:end].decode('ascii', errors='ignore').strip()
            try:
                val = float(s)
            except:
                val = 0.0
            strings.append(val)
            pos = end + 1
            if pos >= 64: break
        if len(strings) == 4:
            self.calib_coeffs[group] = strings
            for i, v in enumerate(strings):
                self.calib_coeffs[group][i].setValue(v)
            self.log(f"Group {group} ({self.calib_group_names[group]}): f0={strings[0]:.8e}, f1={strings[1]:.8e}, f2={strings[2]:.6f}, f3={strings[3]:.6f}")
            return strings
        else:
            self.log(f"Failed to parse coefficients for group {group}")
            return None
    def read_all_calibration(self):
        for g in [1,2,3]:
            self.read_calibration_group(g)
        self.log("Calibration coefficients read complete")
    def write_calibration_group(self, group: int):
        if not self.serial_port.is_open:
            return
        coeffs = [self.calib_coeffs[group][i].value() for i in range(4)]
        data = bytearray()
        for c in coeffs:
            s = f"{c:.10e}".encode('ascii')
            if len(s) > 15: s = s[:15]
            data.extend(s.ljust(16, b'\x00'))
        if len(data) != 64:
            self.log("Internal error: calibration data not 64 bytes")
            return
        head = bytearray([0x81, 0x28, group, 0x00])
        head_crc = sum(head) & 0xFF
        head.append(head_crc)
        full_packet = head + data
        total_crc = sum(full_packet) & 0xFF
        full_packet.append(total_crc)
        self.serial_port.write(full_packet)
        self.log(f"Wrote calibration group {group} ({self.calib_group_names[group]}) → device")
        time.sleep(0.3) # give device time
    def write_selected_calibration(self):
        idx = self.calib_tabs.currentIndex()
        if idx < 3:
            group = idx + 1 # tabs 0,1,2 → groups 1,2,3
            self.write_calibration_group(group)
        self.log("Write complete. Remember to SAVE to flash if needed.")
    def save_parameters_to_flash(self):
        self.save_parameters() # your existing method (0x22)
        self.log("Calibration changes saved to non-volatile memory")
    def preview_calibration_axis(self):
        idx = self.calib_tabs.currentIndex()
        if idx < 3:
            group = idx + 1
            f0, f1, f2, f3 = [self.calib_coeffs[group][i].value() for i in range(4)]
            pixels = np.arange(2048)
            if abs(f0) < 1e-12 and abs(f1) < 1e-9:
                axis = f2 * pixels + f3
            else:
                axis = f0 * pixels**3 + f1 * pixels**2 + f2 * pixels + f3
            minv, maxv = np.min(axis), np.max(axis)
            self.log(f"Preview group {group}: {minv:.2f} → {maxv:.2f} (mean step ≈ {(maxv-minv)/2047:.4f})")
            reply = QtWidgets.QMessageBox.question(self, "Preview",
                f"Group {group}: {minv:.2f} – {maxv:.2f}\nPlot preview axis?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.Yes:
                self.plot_curve_ref.setData(pixels, axis)
                self.plot_curve_ref.setVisible(True)
                self.plot_widget.setLabel('bottom', 'Pixel')
                self.plot_widget.setLabel('left', self.calib_group_names[group])

    def read_smoothing_level(self):
            if not self.serial_port.is_open:
                return
            self.send_command(0x25)  # get command (according to protocol)
            reply = self.read_reply(0x25, timeout=1.0)
            if reply and len(reply) >= 3:
                level = reply[2]
                if 1 <= level <= 10:
                    self.smoothing_level = level
                    self.spin_smooth.setValue(level)
                    self.log(f"Read smoothing level from device: {level}")
                else:
                    self.log(f"Invalid smoothing level read: {level}")
            else:
                self.log("Could not read smoothing level from device")
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = SpectrometerApp()
    window.show()
    sys.exit(app.exec_())
