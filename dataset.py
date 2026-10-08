"""
dataset.py -- column definitions and the data pipeline shared by every model.

load_and_prepare_data() turns the patient .mat files into model-ready windows:
    1. load every session and keep the INPUT_FEATURES columns of each gait cycle
    2. split each session chronologically by gait cycle into train / validation / test
    3. mean-normalize with train statistics: (x - mean) / (max - min)
    4. cut windows: train and validation windows start every `stride` timesteps;
       test windows are one window from the start of each test gait cycle
"""
from dataclasses import dataclass
import numpy as np
from utils import LoadData, split_train_val_test, NormalizedData
from modelutils import ModelUtils

## The 45 columns of every gait cycle, in file order
COLUMN_NAMES = ['DBSLAmp', 'DBSLFreq', 'DBSRAmp', 'DBSRFreq',
                'LCh1Delta', 'LCh1Theta', 'LCh1Alpha', 'LCh1Beta', 'LCh1Gamma',
                'LCh2Delta', 'LCh2Theta', 'LCh2Alpha', 'LCh2Beta', 'LCh2Gamma',
                'LCh3Delta', 'LCh3Theta', 'LCh3Alpha', 'LCh3Beta', 'LCh3Gamma',
                'LCh4Delta', 'LCh4Theta', 'LCh4Alpha', 'LCh4Beta', 'LCh4Gamma',
                'RCh1Delta', 'RCh1Theta', 'RCh1Alpha', 'RCh1Beta', 'RCh1Gamma',
                'RCh2Delta', 'RCh2Theta', 'RCh2Alpha', 'RCh2Beta', 'RCh2Gamma',
                'RCh3Delta', 'RCh3Theta', 'RCh3Alpha', 'RCh3Beta', 'RCh3Gamma',
                'RCh4Delta', 'RCh4Theta', 'RCh4Alpha', 'RCh4Beta', 'RCh4Gamma',
                'WPI']

## DBS settings: model inputs that stay constant over a gait cycle and are not predicted
CONSTANT_FEATURES = ['DBSLAmp', 'DBSLFreq', 'DBSRAmp', 'DBSRFreq']

## Band powers of Left Channel 1: model inputs and prediction targets (add e.g. 'RCh1Delta' to predict more bands)
PREDICTED_FEATURES = ['LCh1Delta', 'LCh1Theta', 'LCh1Alpha', 'LCh1Beta', 'LCh1Gamma']

## Model input columns: constants first, then predicted features (the models rely on this order)
INPUT_FEATURES = CONSTANT_FEATURES + PREDICTED_FEATURES


@dataclass
class PreparedData:
    """Model-ready windows, float32. X: (n, input_width, num_inputs), y: (n, label_width, num_outputs)."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    mean: np.ndarray        # per-column train mean of INPUT_FEATURES
    range: np.ndarray       # per-column train (max - min) of INPUT_FEATURES

    def denormalize_outputs(self, y):
        """Convert predictions or labels from normalized units back to the original band-power units."""
        n = len(CONSTANT_FEATURES)
        return y * self.range[n:] + self.mean[n:]

    def test_mae(self, prediction):
        """Score test predictions: MAE over everything (normalized units) and per band (original units)."""
        overall = float(np.mean(np.abs(prediction - self.y_test)))
        per_band = np.mean(np.abs(self.denormalize_outputs(prediction) - self.denormalize_outputs(self.y_test)), axis = (0, 1))
        return overall, dict(zip(PREDICTED_FEATURES, per_band))


def load_sessions(data_dir, features = INPUT_FEATURES):
    """Return {file name: list of gait cycles}, each cycle a (timesteps, len(features)) array."""
    column_index = [COLUMN_NAMES.index(name) for name in features]
    loader = LoadData(data_dir)
    sessions = {}
    for idx, file_name in enumerate(loader.get_files_name()):
        sessions[file_name] = [cycle[0][:, column_index] for cycle in loader.load_data_by_index(idx)]
    return sessions


def load_and_prepare_data(data_dir, input_width, label_width, shift, stride, val_ratio = 0.2, test_ratio = 0.2,
                          sessions = None):
    """Run the full pipeline described at the top of this file and return a PreparedData.
    Pass `sessions` (from load_sessions) to skip reading the files again."""
    if sessions is None:
        sessions = load_sessions(data_dir)
    train, val, test = split_train_val_test(sessions, list(sessions), (val_ratio, test_ratio))

    ## Normalize on the concatenation of all cycles, then split back into cycles
    normalize = NormalizedData(norm_type = "mean_normalization")
    train_flat, train_lengths = normalize.flatten_data(train, original = False)
    val_flat, val_lengths = normalize.flatten_data(val, original = False)
    test_flat, test_lengths = normalize.flatten_data(test, original = False)

    (mean, value_range), normed_train_flat = normalize.normalized_df(train_flat)
    normed_train = normalize.inversed_flatten_data(normed_train_flat, train_lengths)
    normed_val = normalize.inversed_flatten_data((val_flat - mean) / value_range, val_lengths)
    normed_test = normalize.inversed_flatten_data((test_flat - mean) / value_range, test_lengths)

    num_inputs, num_outputs = len(INPUT_FEATURES), len(PREDICTED_FEATURES)
    X_train, y_train = ModelUtils.prepare_time_series_data(normed_train, input_width, label_width, shift,
                                                           num_inputs, num_outputs, stride = stride)
    X_val, y_val = ModelUtils.prepare_time_series_data(normed_val, input_width, label_width, shift,
                                                       num_inputs, num_outputs, stride = stride)
    X_test, y_test = ModelUtils.prepare_test_series_data(normed_test, input_width, label_width, shift,
                                                         num_inputs, num_outputs)

    arrays = (np.asarray(a, dtype = np.float32) for a in (X_train, y_train, X_val, y_val, X_test, y_test))
    return PreparedData(*arrays, mean = mean, range = value_range)
