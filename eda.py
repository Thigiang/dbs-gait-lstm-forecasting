"""
eda.py -- a look at the data the models are trained on, using the same pipeline and settings as train.py.

Prints:
    sessions and gait cycles per split, and gait-cycle lengths
    value ranges of every model input column (train split, original units)
    the DBS settings used in each session
    the shapes of the model-ready windows
Saves to the images folder:
    eda_cycle_lengths.png   histogram of gait-cycle lengths, with the minimum length a window needs
    eda_input_label.png     one test window: the input history and the labels of each predicted band

Run:  python eda.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from configs import path_to_data, path_to_save_models_images
from dataset import load_sessions, load_and_prepare_data, CONSTANT_FEATURES, PREDICTED_FEATURES, INPUT_FEATURES
from utils import split_train_val_test
import train


def describe_splits(train_cycles, val_cycles, test_cycles, window_size):
    """Print gait-cycle counts and lengths per split, and how many cycles are long enough for a window."""
    print(f"{'split':<12}{'cycles':>8}{'timesteps':>11}{'min len':>9}{'median':>8}{'max len':>9}{'usable':>8}")
    for name, cycles in (("train", train_cycles), ("validation", val_cycles), ("test", test_cycles)):
        lengths = np.array([len(cycle) for cycle in cycles])
        usable = int(np.sum(lengths >= window_size))
        print(f"{name:<12}{len(cycles):>8}{lengths.sum():>11}{lengths.min():>9}{int(np.median(lengths)):>8}"
              f"{lengths.max():>9}{usable:>8}")
    print(f"(usable = cycles with at least INPUT_WIDTH + SHIFT = {window_size} timesteps)")


def describe_columns(train_cycles):
    """Print min / mean / max of every input column over the train split."""
    values = np.concatenate(train_cycles)
    print(f"{'column':<12}{'min':>12}{'mean':>12}{'max':>12}")
    for j, name in enumerate(INPUT_FEATURES):
        print(f"{name:<12}{values[:, j].min():>12.4g}{values[:, j].mean():>12.4g}{values[:, j].max():>12.4g}")


def describe_dbs_settings(sessions):
    """Print the distinct DBS settings (rows of the constant columns) used in each session."""
    n = len(CONSTANT_FEATURES)
    for file_name, cycles in sessions.items():
        settings = np.unique(np.concatenate([cycle[:, :n] for cycle in cycles]), axis = 0)
        print(f"{file_name}: {len(cycles)} cycles, {len(settings)} DBS setting(s) {CONSTANT_FEATURES}")
        for row in settings:
            print(f"    {row.tolist()}")


def plot_cycle_lengths(cycles, window_size):
    lengths = [len(cycle) for cycle in cycles]
    plt.figure(figsize = (10, 4))
    plt.hist(lengths, bins = 40)
    plt.axvline(window_size, color = 'k', linestyle = '--', label = f'window size ({window_size})')
    plt.xlabel('Gait-cycle length (timesteps)')
    plt.ylabel('Number of cycles')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save_models_images, "eda_cycle_lengths.png"), dpi = 150)
    plt.close()


def plot_input_label(data, example = 0):
    """One subplot per predicted band: the normalized input history followed by the labels."""
    n_const = len(CONSTANT_FEATURES)
    input_steps = np.arange(data.X_test.shape[1])
    label_steps = np.arange(data.X_test.shape[1], data.X_test.shape[1] + data.y_test.shape[1])
    fig, axes = plt.subplots(len(PREDICTED_FEATURES), 1, figsize = (12, 2 * len(PREDICTED_FEATURES)), sharex = True)
    for j, (ax, band) in enumerate(zip(axes, PREDICTED_FEATURES)):
        ax.plot(input_steps, data.X_test[example, :, n_const + j], label = 'input')
        ax.plot(label_steps, data.y_test[example, :, j], 'k', label = 'label')
        ax.set_ylabel(band)
    axes[0].legend(loc = 'upper right')
    axes[-1].set_xlabel('Timestep')
    fig.suptitle(f'Test window {example} (normalized)')
    fig.tight_layout()
    fig.savefig(os.path.join(path_to_save_models_images, "eda_input_label.png"), dpi = 150)
    plt.close(fig)


def main():
    os.makedirs(path_to_save_models_images, exist_ok = True)
    window_size = train.INPUT_WIDTH + train.SHIFT

    sessions = load_sessions(path_to_data)
    train_cycles, val_cycles, test_cycles = split_train_val_test(sessions, list(sessions),
                                                                 (train.VAL_RATIO, train.TEST_RATIO))
    print(f"=== {len(sessions)} sessions, split {1 - train.VAL_RATIO - train.TEST_RATIO:.0%} / "
          f"{train.VAL_RATIO:.0%} / {train.TEST_RATIO:.0%} by gait cycle within each session\n")
    describe_splits(train_cycles, val_cycles, test_cycles, window_size)

    print("\n=== Input columns, train split (original units)\n")
    describe_columns(train_cycles)

    print("\n=== DBS settings per session\n")
    describe_dbs_settings(sessions)

    data = load_and_prepare_data(path_to_data, train.INPUT_WIDTH, train.LABEL_WIDTH, train.SHIFT, train.STRIDE,
                                 train.VAL_RATIO, train.TEST_RATIO, sessions = sessions)
    print(f"\n=== Model-ready windows (input {train.INPUT_WIDTH} -> predict {train.LABEL_WIDTH}, "
          f"train/val stride {train.STRIDE})\n")
    print(f"Train: {data.X_train.shape} -> {data.y_train.shape}")
    print(f"Validation: {data.X_val.shape} -> {data.y_val.shape}")
    print(f"Test: {data.X_test.shape} -> {data.y_test.shape}")

    plot_cycle_lengths(train_cycles + val_cycles + test_cycles, window_size)
    plot_input_label(data)
    print(f"\nSaved eda_cycle_lengths.png and eda_input_label.png to {path_to_save_models_images}")


if __name__ == "__main__":
    main()
