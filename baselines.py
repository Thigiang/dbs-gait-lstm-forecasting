"""
baselines.py -- simple, non-learned forecasts to compare the models against.

A model's test MAE only means something next to what a trivial method achieves on the same test windows:
    last value                  repeat the last observed value of each band for all LABEL_WIDTH steps
    average cycle (all)         the mean of all training gait cycles at each timestep, the same for every window
    average cycle (session)     the mean of the training gait cycles of the window's own session

Test windows start at the beginning of a gait cycle, so the average-cycle baselines capture the typical shape of
a gait cycle. A model that beats them uses the input window, not just that typical shape.

Uses the same data and settings as train.py; nothing is trained, so it runs in seconds.

Run:  python baselines.py
"""
import numpy as np
from dataset import CONSTANT_FEATURES, PREDICTED_FEATURES
from utils import split_train_val_test


def baseline_predictions(sessions, data, input_width, label_width, shift, val_ratio, test_ratio):
    """
    Return {baseline name: test predictions (n_test, label_width, n_outputs)}, normalized like the model's.

    sessions  output of dataset.load_sessions; data  the PreparedData built from the same sessions and settings
    """
    n_const = len(CONSTANT_FEATURES)
    window_size = input_width + shift

    def normalize(cycle):
        return (cycle - data.mean) / data.range

    def label_part(cycles):
        """Label timesteps of the first window of each cycle (cycles shorter than a window are skipped)."""
        return np.array([normalize(c)[window_size - label_width:window_size, n_const:]
                         for c in cycles if len(c) >= window_size])

    ## Split each session on its own (the same split as dataset.py) to keep track of which test window is whose
    train_labels, session_average, test_labels = {}, [], []
    for file_name, cycles in sessions.items():
        train_cycles, _, test_cycles = split_train_val_test({file_name: cycles}, [file_name], (val_ratio, test_ratio))
        train_labels[file_name] = label_part(train_cycles)
        test_labels.append(label_part(test_cycles))
        session_average.append(np.repeat(train_labels[file_name].mean(axis = 0, keepdims = True),
                                         len(test_labels[-1]), axis = 0))
    session_average = np.concatenate(session_average).astype(np.float32)
    assert np.allclose(np.concatenate(test_labels), data.y_test, atol = 1e-5), "test windows do not line up with the sessions"

    all_average = np.concatenate(list(train_labels.values())).mean(axis = 0)
    all_average = np.broadcast_to(all_average, data.y_test.shape).astype(np.float32)

    last_value = np.repeat(data.X_test[:, -1:, n_const:], label_width, axis = 1)

    return {"last value": last_value,
            "average cycle (all)": all_average,
            "average cycle (session)": session_average}


def print_comparison(scores):
    """Print a table of {name: (overall MAE, {band: MAE})}."""
    print(f"{'method':<26}{'test MAE (normalized)':>23}   " + "  ".join(f"{band:>10}" for band in PREDICTED_FEATURES))
    for name, (overall, per_band) in scores.items():
        print(f"{name:<26}{overall:>23.5f}   " + "  ".join(f"{per_band[band]:>10.4g}" for band in PREDICTED_FEATURES))
    print("(per-band columns are in original units)")


def main():
    import train  # settings; imported here because train.py imports this module
    from configs import path_to_data
    from dataset import load_sessions, load_and_prepare_data

    sessions = load_sessions(path_to_data)
    data = load_and_prepare_data(path_to_data, train.INPUT_WIDTH, train.LABEL_WIDTH, train.SHIFT, train.STRIDE,
                                 train.VAL_RATIO, train.TEST_RATIO, sessions = sessions)
    baselines = baseline_predictions(sessions, data, train.INPUT_WIDTH, train.LABEL_WIDTH, train.SHIFT,
                                     train.VAL_RATIO, train.TEST_RATIO)
    print_comparison({name: data.test_mae(prediction) for name, prediction in baselines.items()})


if __name__ == "__main__":
    main()
