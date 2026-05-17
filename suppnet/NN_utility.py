import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline
from numpy import ma
import numpy as np
import os


MIN_SMOOTHING_FACTOR = 0.05
MIN_CONTINUUM_STD = 1e-4
MAX_SPLINE_WEIGHT = 1e4
MAX_SMOOTHING_KNOTS = 1200


def safe_median_scale(flux, fallback=1.0):
    """Return a finite, non-zero flux scale for pre-normalizing spectra."""
    flux = np.asarray(flux, dtype=float)
    finite_flux = flux[np.isfinite(flux)]
    if finite_flux.size == 0:
        if fallback is None:
            raise ValueError("Cannot normalize spectrum: flux contains no finite values.")
        return float(fallback)

    scale = np.nanmedian(finite_flux)
    if not np.isfinite(scale) or np.isclose(scale, 0.0):
        if fallback is None:
            raise ValueError(
                "Cannot normalize spectrum: median flux is zero or non-finite."
            )
        return float(fallback)
    return float(scale)


def _finite_smoothing_inputs(wave, continuum, continuum_std):
    wave = np.asarray(wave, dtype=float)
    continuum = np.asarray(continuum, dtype=float)
    continuum_std = np.asarray(continuum_std, dtype=float)

    if not (wave.shape == continuum.shape == continuum_std.shape):
        raise ValueError("wave, continuum, and continuum_std must have matching shapes.")

    mask = (
        np.isfinite(wave)
        & np.isfinite(continuum)
        & np.isfinite(continuum_std)
        & ~np.isclose(continuum, 0.0)
        & ~np.isclose(continuum_std, 0.0)
    )
    wave = wave[mask]
    continuum = continuum[mask]
    continuum_std = continuum_std[mask]

    if wave.size == 0:
        return wave, continuum, continuum_std

    order = np.argsort(wave)
    wave = wave[order]
    continuum = continuum[order]
    continuum_std = continuum_std[order]

    unique_wave, unique_idx = np.unique(wave, return_index=True)
    return unique_wave, continuum[unique_idx], continuum_std[unique_idx]


def _limit_knots(knots_x, knots_y, max_knots=MAX_SMOOTHING_KNOTS):
    knots_x = np.asarray(knots_x, dtype=float)
    knots_y = np.asarray(knots_y, dtype=float)
    if max_knots is None or knots_x.size <= max_knots:
        return knots_x, knots_y

    idx = np.unique(np.linspace(0, knots_x.size - 1, max_knots, dtype=int))
    return knots_x[idx], knots_y[idx]


def fit_smoothing_spline_knots(
    wave,
    continuum,
    continuum_std,
    smoothing_factor=1.0,
    max_knots=MAX_SMOOTHING_KNOTS,
):
    """Fit a robust continuum smoothing spline and return bounded knots.

    The model-facing normalization remains unchanged; this helper only guards the
    post-model continuum smoothing used by both the GUI and batch code.
    """
    smoothing_factor = max(MIN_SMOOTHING_FACTOR, float(smoothing_factor))
    wave, continuum, continuum_std = _finite_smoothing_inputs(
        wave, continuum, continuum_std
    )

    if wave.size < 4:
        if wave.size >= 2:
            return wave, continuum
        return np.array([]), np.array([])

    continuum_std = np.maximum(continuum_std * smoothing_factor, MIN_CONTINUUM_STD)
    weights = np.minimum(1.0 / continuum_std, MAX_SPLINE_WEIGHT)

    try:
        spl = UnivariateSpline(
            wave,
            continuum,
            w=weights,
            bbox=[None, None],
            k=3,
            s=None,
            ext=0,
            check_finite=False,
        )
        knots = spl.get_knots()
        knots_y = spl(knots)
        return _limit_knots(knots, knots_y, max_knots=max_knots)
    except Exception as exc:
        print(
            f"UnivariateSpline fitting failed ({exc}); "
            "falling back to evenly-spaced knots."
        )
        n_fallback = min(max_knots or MAX_SMOOTHING_KNOTS, wave.size)
        idx = np.unique(np.linspace(0, wave.size - 1, n_fallback, dtype=int))
        return wave[idx], continuum[idx]


def evaluate_smoothing_spline(wave_orig, knots_x, knots_y):
    """Evaluate smoothing knots on the original wavelength grid."""
    wave_orig = np.asarray(wave_orig, dtype=float)
    knots_x = np.asarray(knots_x, dtype=float)
    knots_y = np.asarray(knots_y, dtype=float)

    if knots_x.size == 0:
        return np.zeros_like(wave_orig, dtype=float)
    if knots_x.size == 1:
        return np.full_like(wave_orig, knots_y[0], dtype=float)

    k = min(3, knots_x.size - 1)
    return InterpolatedUnivariateSpline(knots_x, knots_y, k=k)(wave_orig)


class ProcessSpectrum:

    def __init__(self, model, normalizer, step_size=64, window_len=8192, resampling_step=0.05):
        self.normalizer = normalizer
        self.model = model

        self.window_len = window_len
        self.step_size = step_size
        self.only_norm = self.model.norm_only
        self.resampling_step = resampling_step

    def prepare_data(self, y):
        shifts = np.arange(0, self.window_len, self.step_size)

        y_shape = y.shape[0]
        pad_number = 2*self.window_len - y_shape % self.window_len
        padded_y = np.pad(y, (0, pad_number), mode='constant',
                          constant_values=np.nan)

        padded_all = np.stack([np.roll(padded_y, shift) for shift in shifts])
        for_processing = padded_all.reshape((-1, self.window_len))
        return for_processing, shifts

    def get_results(self, processed, shifts):
        reshaped = np.array([np.roll(part, shift=-shift).flatten()
                             for part, shift in zip(np.split(processed, shifts.shape[0]), shifts)])
        w = self.generate_weights(sigma=3, length=reshaped.shape[0])
        return self.weighted_avg_and_std(reshaped, w)
        # return np.nanmean(reshaped, axis=0), np.nanstd(reshaped, axis=0)

    def weighted_avg_and_std(self, values, weights):
        values = np.ma.masked_array(values, np.isnan(values))
        average = np.ma.average(values, weights=weights, axis=0)
        variance = np.ma.average((values-average)**2, weights=weights, axis=0)
        return average, np.sqrt(variance)

    def generate_weights(self, sigma, length):
        x = np.linspace(-3, 3, length)
        weights = np.exp(-(x/sigma)**2)
        return weights

    def resample(self, wave, flux):
        wave = np.asarray(wave, dtype=float)
        flux = np.asarray(flux, dtype=float)

        valid_wave = np.isfinite(wave)
        if np.count_nonzero(valid_wave) < 2:
            raise ValueError("Cannot resample spectrum: fewer than two finite wavelengths.")

        wave = wave[valid_wave]
        flux = flux[valid_wave]
        order = np.argsort(wave)
        wave = wave[order]
        flux = flux[order]

        unique_wave, unique_idx = np.unique(wave, return_index=True)
        wave = unique_wave
        flux = flux[unique_idx]
        if wave.size < 2:
            raise ValueError("Cannot resample spectrum: wavelength range is degenerate.")

        wave_range = wave[-1] - wave[0]
        if not np.isfinite(wave_range) or wave_range <= 0:
            raise ValueError("Cannot resample spectrum: wavelength range must be positive.")
        if not np.isfinite(self.resampling_step) or self.resampling_step <= 0:
            raise ValueError("Cannot resample spectrum: resampling_step must be positive.")

        no_samples = int(np.floor(wave_range / self.resampling_step)) + 1
        if no_samples < 2:
            raise ValueError("Cannot resample spectrum: resampling grid has fewer than two samples.")

        new_wave = np.linspace(wave[0], wave[-1], num=no_samples)
        flux = np.where(np.isfinite(flux), flux, 0.0)
        new_flux = np.interp(new_wave, wave, flux)
        return new_wave, new_flux

    def normalize(self, wave, flux):
        new_wave, new_flux = self.resample(wave, flux)
        flux_prepared, shifts = self.prepare_data(new_flux)

        normed_flux = self.normalizer.normalize(flux_prepared)
        normed_flux = normed_flux[..., None]
        result = self.model.predict(normed_flux)
        length = new_wave.shape[0]
        if self.only_norm:
            continuum, continuum_std = self.process_signal(
                result, shifts, length, norm=True)
            return (np.interp(wave, new_wave, np.asarray(continuum)),
                    np.interp(wave, new_wave, np.asarray(continuum_std)))
        else:
            continuum, continuum_std = self.process_signal(
                result["cont"], shifts, length, norm=True)
            segmentation, segmentation_std = self.process_signal(
                result["seg"], shifts, length, norm=False)
            return (np.interp(wave, new_wave, np.asarray(continuum)),
                    np.interp(wave, new_wave, np.asarray(continuum_std)),
                    np.interp(wave, new_wave, np.asarray(segmentation)),
                    np.interp(wave, new_wave, np.asarray(segmentation_std)))

    def process_signal(self, result, shifts, length, norm=True):
        processed = np.squeeze(result)
        if norm:
            processed = self.normalizer.denormalize(processed)

        processed, continuum_std = self.get_results(processed, shifts)
        processed = processed[:length]
        processed_std = continuum_std[:length]
        if ma.isMaskedArray(processed):
            processed = processed.filled(np.nan)
        if ma.isMaskedArray(processed_std):
            processed_std = processed_std.filled(np.nan)
        processed = np.asarray(processed, dtype=float)
        processed_std = np.asarray(processed_std, dtype=float)
        processed[processed < 0.] = 0.
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        processed_std = np.nan_to_num(processed_std, nan=0.0, posinf=0.0, neginf=0.0)
        return processed, processed_std


class MinMaxNormalizer:
    def __init__(self, eps=1e-8):
        self._max = None
        self._min = None
        self._denominator = None
        self.eps = eps

    def normalize(self, X, Y=None):
        X = np.asarray(X, dtype=float)
        masked_values = np.isfinite(X) & (X > 0)
        X_masked = ma.masked_array(X, mask=~masked_values)

        self._min = ma.min(X_masked, axis=1, keepdims=True).filled(0.0)
        self._max = ma.max(X_masked, axis=1, keepdims=True).filled(1.0)
        denominator = self._max - self._min
        invalid_scale = (~np.isfinite(denominator)) | (denominator <= self.eps)
        self._denominator = np.where(invalid_scale, 1.0, denominator)
        self._min = np.where(np.isfinite(self._min), self._min, 0.0)

        X_normed = np.where(masked_values, (X - self._min) / self._denominator, 0.0)
        X_normed = np.nan_to_num(X_normed, nan=0.0, posinf=0.0, neginf=0.0)
        if Y is not None:
            Y = np.asarray(Y, dtype=float)
            Y_normed = (Y - self._min) / self._denominator
            return X_normed, np.nan_to_num(Y_normed, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            return X_normed

    def denormalize(self, Y):
        if self._min is None or self._denominator is None:
            raise RuntimeError("Cannot denormalize before normalize has been called.")
        Y = np.asarray(Y, dtype=float)
        denormed = Y * self._denominator + self._min
        return np.nan_to_num(denormed, nan=0.0, posinf=0.0, neginf=0.0)


# Implement normalization script:


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Normalise some stellar spectra.')
    parser.add_argument('file_names', type=str, nargs='+',
                        help='Spectra to be normed')
    parser.add_argument('-s', '--skip', nargs=1, dest='skipRows',
                        required=False, default=[0], help="No of rows to skip.", type=int)

    args = parser.parse_args()

    process_all_spectra(args.file_names, skip_rows=args.skipRows[0])


def process_all_spectra(paths, skip_rows, resampling_step=0.05, smoothing=1.0, which_weights='active'):
    print("==================================\nNeural network loading\n==================================\n")
    nn = get_suppnet(resampling_step=resampling_step, step_size=256, norm_only=False, which_weights=which_weights)
    print("==================================\nNeural network loaded\n==================================\n")

    for sfn in paths:
        out_path = os.path.splitext(sfn)[0]+'.all'
        print(f"Processing {sfn} -> {out_path}")
        spectrum = pd.read_csv(sfn,
                               index_col=None,
                               header=None,
                               sep=r'\s+',
                               skiprows=skip_rows,
                               comment="#")
        spectrum[1] /= safe_median_scale(spectrum[1], fallback=None)
        process_spectrum(spectrum, out_path, nn, smoothing=smoothing)

def get_suppnet(resampling_step=0.05, step_size=256, norm_only=True, which_weights='active'):
    from suppnet.SUPPNet import get_suppnet_model

    """
    Returns ProcessSpectrum object that can be used for pseudo-continuum prediction:
    continuum, continuum_error = nn.normalize(wave, flux)
    when norm_only=False:
    continuum, continuum_error, segmentation, segmentation_error = nn.normalize(wave, flux)
    """
    model = get_suppnet_model(norm_only=norm_only, which_weights=which_weights)
    nn = ProcessSpectrum(model,
                         MinMaxNormalizer(),
                         step_size=step_size,
                         window_len=8192,
                         resampling_step=resampling_step
                         )
    return nn 


def process_spectrum(spectrum, filename, nn, smoothing=1.0):
    wave = spectrum[0].values
    flux = spectrum[1].values
    if nn.only_norm:
        cont, cont_err = nn.normalize(wave, flux)
    else:
        cont, cont_err, seg, seg_err = nn.normalize(wave, flux)
    cont_smo = get_smoothed_continuum(wave, cont, cont_err, smoothing_factor=smoothing)
    normed_flux = flux/cont_smo
    normed_flux_error = cont_err/cont_smo
    if nn.only_norm:
        save_results_norm(filename, wave, flux, normed_flux,
                          normed_flux_error, cont_smo, cont, cont_err)
    else:
        save_results_both(filename, wave, flux, normed_flux,
                          normed_flux_error, cont_smo, cont, cont_err, seg, seg_err)


def backend_normed_spectrum(wave, flux, nn):
    if nn.only_norm:
        cont, cont_err = nn.normalize(wave, flux)
        return cont, cont_err
    else:
        cont, cont_err, seg, seg_err = nn.normalize(wave, flux)
        return cont, cont_err, seg, seg_err


def get_smoothed_continuum(wave_orig, continuum, continuum_std, smoothing_factor=1.0):
    knots_x, knots_y = fit_smoothing_spline_knots(
        wave_orig, continuum, continuum_std, smoothing_factor=smoothing_factor
    )
    return evaluate_smoothing_spline(wave_orig, knots_x, knots_y)


def save_results_norm(filename, wave, flux, normed_flux, normed_flux_err, cont_smo, continuum, continuum_err):
    df = pd.DataFrame({"wave": wave,
                       "flux": flux,
                       "normed_flux": normed_flux,
                       "normed_error": normed_flux_err,
                       "smoothed_continuum": cont_smo,
                       "continuum": continuum,
                       "continuum_err": continuum_err,
                       })
    mask = (df['flux'] == 0)
    df.loc[mask, df.columns != 'wave'] = 0.
    df.to_csv(filename, sep=' ', index=False)


def save_results_both(filename, wave, flux, normed_flux, normed_flux_err, cont_smo, continuum, continuum_err, seg, seg_err):
    df = pd.DataFrame({"wave": wave,
                       "flux": flux,
                       "normed_flux": normed_flux,
                       "normed_error": normed_flux_err,
                       "smoothed_continuum": cont_smo,
                       "continuum": continuum,
                       "continuum_err": continuum_err,
                       "segmentation": seg,
                       "segmentation_err": seg_err
                       })
    mask = (df['flux'] == 0)
    df.loc[mask, df.columns != 'wave'] = 0.
    df.to_csv(filename, sep=' ', index=False)


if __name__ == "__main__":
    # eg. python NN_utility.py example_data/*dat
    main()
