import os
import sys
import time
import pandas as pd
import numpy as np

from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel
from PySide6.QtCore import QFile, QThreadPool
from app_components.main_window_qt import Ui_MainWindow
from app_components.worker import Worker
from app_components.app_logic import Logic
from app_components.draggable_scatter import DraggableScatter
from suppnet.NN_utility import MIN_SMOOTHING_FACTOR

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from matplotlib.gridspec import GridSpec


class MainWindow(QMainWindow):
    def __init__(self, path=None, show_segmentation=False,resampling_step=0.05,which_weights="active"):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.filename_label = QLabel("")
        self.ui.statusbar.addPermanentWidget(self.filename_label)
        self.logic = Logic(resampling_step=resampling_step, which_weights=which_weights)
        self.spline = self.logic.spline
        self.show_segmentation = show_segmentation
        self.is_normalizing = False
        self.is_smoothing = False

        self.threadpool = QThreadPool()
        if path is None:
            self.for_threading()

        self.configure_slider()
        self.ui.slider_value.setText("1.00")
        self.ui.update_normalization.clicked.connect(
            self.on_update_normalization)

        self.configure_spinning_box()
        self.ui.sampling_spin_box.valueChanged.connect(
            self.on_update_sampling)

        self.addmpl()

        self.ui.actionOpen_spectrum.triggered.connect(self.load_spectrum)
        self.ui.actionSave_normed_spectrum.triggered.connect(
            self.save_normed_spectrum)
        self.ui.actionSave_results.triggered.connect(self.save_all_results)
        self.ui.actionOpen_processed_spectrum.triggered.connect(
            self.on_load_processed_spectrum)
        self.ui.actionClose.triggered.connect(QApplication.quit)

        self.ui.action_normalize.triggered.connect(self.normalize)

        if path is not None:
            self.logic.read_processed_spectrum(path)
            self.update_plots_and_data(resize=True)

    def addmpl(self):
        self.dpi = 100
        self.fig = Figure(dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas,
                                         self.ui.centralwidget
                                         )
        self.create_plots()
        self.ui.mplvl.addWidget(self.canvas)
        self.ui.mplvl.addWidget(self.toolbar)

        self.canvas.draw()

    def configure_slider(self):
        self.slider_minimum = MIN_SMOOTHING_FACTOR
        self.slider_maximum = 2.0
        self.slider_step = 0.01
        self.slider_number_of_steps = int(round(
            (self.slider_maximum-self.slider_minimum)/self.slider_step))
        self.slider_ticks = max(1, self.slider_number_of_steps//10)
        self.ui.horizontalSlider.setMinimum(0)
        self.ui.horizontalSlider.setMaximum(self.slider_number_of_steps)
        self.ui.horizontalSlider.setTickInterval(self.slider_ticks)
        self.ui.horizontalSlider.setSingleStep(1)

        self.ui.horizontalSlider.setValue(self.slider_position_from_value(1.0))

        self.ui.horizontalSlider.valueChanged.connect(self.update_label)

    def slider_value_from_position(self, position):
        slider_value = self.slider_minimum + position*self.slider_step
        return min(self.slider_maximum, max(self.slider_minimum, slider_value))

    def slider_position_from_value(self, value):
        value = min(self.slider_maximum, max(self.slider_minimum, float(value)))
        return int(round((value-self.slider_minimum)/self.slider_step))

    def update_label(self, position):
        slider_value = self.slider_value_from_position(position)
        self.ui.slider_value.setText(f"{slider_value:.2f}")

    def on_update_normalization(self):
        if self.is_normalizing or self.is_smoothing:
            return
        if self.logic.continuum is None or self.logic.continuum_error is None:
            return
        slider_value = self.slider_value_from_position(
            self.ui.horizontalSlider.value())
        self.is_smoothing = True
        self.ui.statusbar.showMessage("Updating smoothing in background...")
        self._set_smoothing_ui(True)
        worker = Worker(self._calculate_smoothing_update, slider_value)
        worker.signals.result.connect(self._on_smoothing_result)
        worker.signals.error.connect(self._on_smoothing_error)
        worker.signals.finished.connect(self._on_smoothing_done)
        self.threadpool.start(worker)

    def _calculate_smoothing_update(self, slider_value):
        knots_x, knots_y = self.logic.calculate_spline_knots(slider_value)
        return slider_value, knots_x, knots_y

    def _set_smoothing_ui(self, smoothing):
        self.ui.update_normalization.setEnabled(not smoothing)
        self.ui.horizontalSlider.setEnabled(not smoothing)

    def _on_smoothing_result(self, result):
        slider_value, knots_x, knots_y = result
        self.logic.apply_spline_knots(slider_value, knots_x, knots_y)
        self.update_plots_and_data(resize=False)
        self.ui.statusbar.showMessage("Smoothing updated.")

    def _on_smoothing_error(self, error_info):
        exctype, value, tb = error_info
        self.ui.statusbar.showMessage(f"Smoothing failed: {value}")
        print(tb)

    def _on_smoothing_done(self):
        self.is_smoothing = False
        self._set_smoothing_ui(False)

    def configure_spinning_box(self):
        self.ui.sampling_spin_box.setMinimum(0.003)
        self.ui.sampling_spin_box.setMaximum(0.5)
        self.ui.sampling_spin_box.setSingleStep(0.001)
        self.ui.sampling_spin_box.setDecimals(3)

    def on_update_sampling(self):
        sampling_value = float(self.ui.sampling_spin_box.value())
        print(f"Sampling changed : {sampling_value:.4f}")
        self.logic.set_sampling(sampling_value)

    def for_threading(self):
        worker = Worker(*self.logic.get_model())
        worker.signals.result.connect(self.load_model)
        worker.signals.finished.connect(self.thread_complete)

        # Execute
        self.threadpool.start(worker)

    def thread_complete(self):
        self.ui.action_normalize.setEnabled(True)
        self.ui.sampling_spin_box.setValue(self.logic.get_sampling())
        self.ui.statusbar.showMessage("Model loaded")

    def load_model(self, model):
        self.logic.set_model(model)

    # READ/WRITE FILES
    def load_spectrum(self):
        filename, filetype = QFileDialog.getOpenFileName(self, 'OpenFile')
        if filename:
            print(f"Reading spectrum: {filename}")
            self.logic.read_spectrum(filename)
            self.update_plots_and_data(resize=True)
            self.filename_label.setText(self.logic.opened_file_name)

    def on_load_processed_spectrum(self):
        filename, filetype = QFileDialog.getOpenFileName(self, 'OpenFile')
        if filename:
            print(f"Reading normed spectrum: {filename}")
            self.logic.read_processed_spectrum(filename)
            self.update_plots_and_data(resize=True)

    def save_all_results(self):
        default_fn = os.path.splitext(self.logic.opened_file_name)[0]+'.all'
        filename, filetype = QFileDialog.getSaveFileName(
            self, 'SaveFile', default_fn)
        if filename:
            print(f"Saving result to file: {filename}")
            self.logic.save_all_results(filename)
            print(f"{filename} saved!")

    def save_normed_spectrum(self,):
        default_fn = os.path.splitext(self.logic.opened_file_name)[0]+'.norm'
        filename, filetype = QFileDialog.getSaveFileName(
            self, 'SaveFile', default_fn)
        if filename:
            print(f"Saving result to file: {filename}")
            self.logic.save_normed_spectrum(filename)
            print(f"{filename} saved!")

    # PLOTTING

    def update_plots_and_data(self, resize=False):
        self.logic.update_all()
        spectrum = self.logic.get_plotting_data()
        wave = spectrum["wave"].values
        flux = spectrum["flux"].values
        self.line11.set_data(wave, flux)

        if self.logic.continuum is not None:
            self.line12.set_data(wave, self.logic.continuum)
        else:
            self.line12.set_data([], [])

        self.ds.update_plot()

        if self.logic.normed_flux is not None:
            self.line21.set_data(wave, self.logic.normed_flux)
            self.line22.set_data([wave[0], wave[-1]], [1, 1])

        else:
            self.line21.set_data([], [])
            self.line22.set_data([], [])

        if self.show_segmentation:
            if self.logic.segmentation is not None:
                self.line31.set_data(wave, self.logic.segmentation)
            else:
                self.line31.set_data([], [])

        if resize:
            self.ax1.set_autoscale_on(True)
            self.ax1.relim()
            self.ax1.autoscale_view(True, True, True)
            self.toolbar.update()
        self.canvas.draw()

    def normalize(self):
        if self.is_normalizing:
            return
        self.is_normalizing = True
        self.ui.statusbar.showMessage("Normalising in background...")
        self._set_normalizing_ui(True)
        worker = Worker(self.logic.compute_continuum)
        worker.signals.error.connect(self._on_normalize_error)
        worker.signals.finished.connect(self._on_normalize_done)
        self.threadpool.start(worker)

    def _set_normalizing_ui(self, normalizing):
        self.ui.action_normalize.setEnabled(not normalizing)
        self.ui.update_normalization.setEnabled(not normalizing)
        self.ui.horizontalSlider.setEnabled(not normalizing)
        self.ui.actionOpen_spectrum.setEnabled(not normalizing)
        self.ui.actionOpen_processed_spectrum.setEnabled(not normalizing)

    def _on_normalize_error(self, error_info):
        exctype, value, tb = error_info
        self.ui.statusbar.showMessage(f"Normalisation failed: {value}")
        print(tb)
        self.is_normalizing = False
        self._set_normalizing_ui(False)

    def _on_normalize_done(self):
        self.update_plots_and_data(resize=False)
        self.ui.statusbar.showMessage("Normalisation done.")
        self.is_normalizing = False
        self._set_normalizing_ui(False)

    def create_plots(self):
        self.fig.subplots_adjust(
            wspace=0.0, hspace=0.0, top=0.95, bottom=0.10, left=0.10, right=0.95)

        if self.show_segmentation:
            gs = GridSpec(6, 1)
        else:
            gs = GridSpec(5, 1)

        self.ax1 = self.fig.add_subplot(gs[:3])
        self.ax1.grid(True)
        self.line11, = self.ax1.plot([], [], 'k-', zorder=20)
        self.line12, = self.ax1.plot([], [], 'b-', zorder=30)
        self.ds = DraggableScatter(self.ax1, [], [], self)
        self.ax1.set_ylabel("Flux")

        self.ax2 = self.fig.add_subplot(gs[3:5], sharex=self.ax1)
        self.ax2.grid(True)
        self.line21, self.line22, = self.ax2.plot([], [], 'k', [], [], 'b--')

        self.ax1.set_ylim(auto=True)
        self.ax2.set_ylim([0.1, 2.0])
        self.ax2.set_ylabel("Normed flux")

        if self.show_segmentation:
            self.ax3 = self.fig.add_subplot(gs[5:], sharex=self.ax1)
            self.ax3.grid(True)
            self.line31, = self.ax3.plot([], [], 'k', zorder=20)
            self.ax3.set_ylim([-0.1, 1.1])
            self.ax3.set_xlabel("Segmentation")
            self.ax3.set_xlabel(r"Wavelength [$\AA$]")
        else:
            self.ax2.set_xlabel(r"Wavelength [$\AA$]")


def run_window_app(path=None, show_segmentation=False, resampling_step=0.05,which_weights="active"):
    app = QApplication(sys.argv)

    window = MainWindow(path=path, show_segmentation=show_segmentation,
                        resampling_step=resampling_step, which_weights=which_weights)
    window.show()

    sys.exit(app.exec())


def argument_parser():
    import argparse
    from argparse import RawTextHelpFormatter

    description = "\n".join([
        "Code for stellar spectrum normalisation based on neural network SUPPNet",
        " ",
        "Usage scenarios:",
        "1. Spectrum-by-spectrum normalisation using interactive app:", " ",
        "    python suppnet.py [--segmentation]",
        " ",
        "2. Normalisation of group of spectra without any supervision:", " ",
        "    python suppnet.py --quiet [--skip number_of_rows_to_skip] path_to_spectrum_1.txt [path_to_spectrum_2.txt ...]",
        " ",
        "3. Manual inspection and correction of previously normalised spectrum, SUPPNet will not be loaded (often used in pair with 2.):", " ",
        "    python suppnet.py [--segmentation] --path path_to_processing_results.all",
        " ",
    ])

    parser = argparse.ArgumentParser(
        description=description, formatter_class=RawTextHelpFormatter)

    parser.add_argument('--quiet',
                        dest='without_window_app',
                        action='store_false',
                        help='Do not open window app for manual normalisation.',
                        required=False
                        )
    
    # add argument that set the value of the resampling_step (float, default=0.05)
    parser.add_argument('--sampling',
                        dest='resampling_step',
                        action='store',
                        help='Set the sampling of the spectrum.',
                        required=False,
                        default=0.05,
                        type=float
                        )
    
    # add argment that defines the weights used (default=active, type string)
    parser.add_argument('--weights',
                        dest='which_weights',
                        action='store',
                        help='Set the weights used. Avalible are: active, synth, emission',
                        required=False,
                        default='active',
                        type=str
                        )
    parser.add_argument('--smoothing',
                        dest='smoothing',
                        action='store',
                        help='Set continuum smoothing factor used in quiet mode.',
                        required=False,
                        default=1.0,
                        type=float
                        )
    available_weights = ['active', 'synth', 'emission']

    if '--quiet' in sys.argv:
        parser.add_argument('file_names',
                            type=str,
                            nargs='+',
                            help='Spectra to be normed.')
        parser.add_argument('--skip',
                            nargs=1,
                            dest='skipRows',
                            required=False,
                            default=[0],
                            help="No of rows to skip.",
                            type=int)
    else:
        parser.add_argument('--segmentation',
                            dest='show_segmentation',
                            action='store_true',
                            required=False,
                            help='Display segmentation on the plot.')
        parser.add_argument('--path',
                            type=str,
                            default=[None],
                            nargs=1,
                            help='Normed spectrum to be automatically loaded for manual correction. Neural network will not be loaded!')

    args = parser.parse_args()

    # check if the weights are available
    if args.which_weights not in available_weights:
        print('The weights you selected are not available. Please select one of the following: active, synth, emission')
        sys.exit()

    return args


if __name__ == "__main__":

    args = argument_parser()

    if args.without_window_app:
        run_window_app(path=args.path[0],
                       show_segmentation=args.show_segmentation,
                       resampling_step=args.resampling_step,
                       which_weights=args.which_weights)
    else:
        from suppnet.NN_utility import process_all_spectra
        process_all_spectra(args.file_names,
                            skip_rows=args.skipRows[0],
                            resampling_step=args.resampling_step,
                            smoothing=args.smoothing,
                            which_weights=args.which_weights)
