"""
train.py -- train and evaluate a model that forecasts Left Channel 1 band powers from DBS settings.

Each model reads INPUT_WIDTH timesteps of the DBS settings and the predicted band powers (by default the 5 Left
Channel 1 bands, Delta to Gamma; see PREDICTED_FEATURES in dataset.py) and predicts the next LABEL_WIDTH timesteps
of those band powers. Both models use the same data (dataset.py) and the same training settings, so their test
MAEs can be compared directly.

    --model feedback   FeedBack: warms up on the input window, then predicts one step at a time, feeding each
                       prediction back in as the next input (closed loop, INPUT_WIDTH + LABEL_WIDTH - 1 LSTM steps).
    --model direct     DirectLSTM: the same 2-layer LSTM reads the input window, then a Dense head predicts all
                       LABEL_WIDTH steps at once (INPUT_WIDTH LSTM steps; about 2.5x faster per epoch).

Training is safe to interrupt: progress is backed up after every epoch, and running the same command again
resumes where it stopped. The best weights so far and a per-epoch CSV log are written throughout training.

Outputs (named <model>_<settings>_*):
    results folder   best weights (.weights.h5), per-epoch log (.csv), full history (.pkl), test predictions (.pkl)
    images folder    loss curves and one prediction plot per band for a test example

Run:
    python train.py --model feedback
    python train.py --model direct --epochs 100 --patience 10
Paths come from configs.py (environment variables FEEDBACK_DATA_DIR, FEEDBACK_RESULTS_DIR, FEEDBACK_IMAGES_DIR).
TensorFlow uses an Apple-silicon GPU automatically when the tensorflow-metal plugin is installed.
"""
import os
import csv
import time
import pickle
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from configs import path_to_data, path_to_save_models_results, path_to_save_models_images
from dataset import load_and_prepare_data, CONSTANT_FEATURES, PREDICTED_FEATURES, INPUT_FEATURES
from modelutils import FeedBack, DirectLSTM, ModelUtils

## ---------------- settings ----------------
## Windows
INPUT_WIDTH = 150       # timesteps the model reads
LABEL_WIDTH = 400       # timesteps the model predicts
SHIFT = 400             # offset from the end of the input window to the end of the label window
STRIDE = 5              # train/validation windows start every STRIDE timesteps (1 would give near-duplicate windows)
VAL_RATIO, TEST_RATIO = 0.2, 0.2   # chronological split of each session's gait cycles

## Model (from a search on an earlier data setup; rerun finetune.py to tune for this one)
LSTM_UNITS = 414
DENSE_UNITS = 37
L2_REG = 2e-05

## Training
LEARNING_RATE = 7.9e-05
BATCH_SIZE = 128
SHUFFLE = True          # each batch mixes windows from different gait cycles
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description = "Train and evaluate a band-power forecasting model.")
    parser.add_argument("--model", choices = ["feedback", "direct"], required = True,
                        help = "feedback: closed-loop FeedBack LSTM; direct: DirectLSTM (all steps at once)")
    parser.add_argument("--epochs", type = int, default = 300, help = "maximum number of epochs (default: 300)")
    parser.add_argument("--patience", type = int, default = 20, help = "early-stopping patience in epochs (default: 20)")
    return parser.parse_args()


def build_model(model_type, sample_input, lstm_units = LSTM_UNITS, dense_units = DENSE_UNITS, l2_reg = L2_REG):
    """Create the model and build its weights on one input window (BackupAndRestore needs a built model)."""
    num_outputs = len(PREDICTED_FEATURES)
    if model_type == "feedback":
        model = FeedBack(lstm_units = lstm_units, dense_units = dense_units, out_steps = LABEL_WIDTH,
                         num_inputs = len(INPUT_FEATURES), num_features = num_outputs,
                         num_constants = len(CONSTANT_FEATURES), l2_reg = l2_reg)
    else:
        model = DirectLSTM(lstm_units = lstm_units, dense_units = dense_units, out_steps = LABEL_WIDTH,
                           num_features = num_outputs, l2_reg = l2_reg)
    model(sample_input)
    return model


def read_log(path):
    """Return a CSVLogger file as {column: list of floats}, or {} if it does not exist yet."""
    if not os.path.exists(path):
        return {}
    with open(path) as file:
        rows = list(csv.DictReader(file))
    return {key: [float(row[key]) for row in rows] for key in rows[0]} if rows else {}


def train(model, data, run_name, epochs, patience):
    """
    Fit with early stopping, backing up after every epoch so an interrupted run resumes.
    Leaves the model holding the best weights of the whole run and returns the full per-epoch history.
    """
    weights_path = os.path.join(path_to_save_models_results, f"{run_name}.weights.h5")
    log_path = os.path.join(path_to_save_models_results, f"{run_name}_log.csv")
    backup_dir = os.path.join(path_to_save_models_results, f"{run_name}_backup")  # removed when training ends

    ## When resuming, only save a checkpoint if it beats the best val_loss of the earlier epochs
    resuming = os.path.isdir(backup_dir)
    previous_val_loss = read_log(log_path).get('val_loss', []) if resuming else []
    if resuming:
        print(f"Resuming from {backup_dir} ({len(previous_val_loss)} epochs logged)")

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        weights_path, monitor = 'val_loss', mode = 'min', save_best_only = True, save_weights_only = True,
        initial_value_threshold = min(previous_val_loss) if previous_val_loss else None)
    callbacks = [tf.keras.callbacks.BackupAndRestore(backup_dir),
                 checkpoint,
                 tf.keras.callbacks.CSVLogger(log_path, append = resuming)]

    start_time = time.time()
    history = ModelUtils.compile_and_fit(model, (data.X_train, data.y_train), (data.X_val, data.y_val),
                                         (LEARNING_RATE, epochs), patience = patience, batch_size = BATCH_SIZE,
                                         shuffle = SHUFFLE, use_early_stopping = True, extra_callbacks = callbacks)
    training_time = time.time() - start_time
    num_epochs_run = len(history.history['loss'])
    print(f"Training time: {time.strftime('%Hh%Mm%Ss', time.gmtime(training_time))} "
          f"for {num_epochs_run} epochs ({training_time / num_epochs_run:.1f} s/epoch)")

    model.load_weights(weights_path)
    return read_log(log_path)


def evaluate(model, data):
    """Predict the test windows; return predictions and MAE overall (normalized) and per band (original units)."""
    prediction = model(data.X_test).numpy()
    test_mae = np.mean(np.abs(prediction - data.y_test))
    band_mae = np.mean(np.abs(data.denormalize_outputs(prediction) - data.denormalize_outputs(data.y_test)), axis = (0, 1))
    return prediction, test_mae, dict(zip(PREDICTED_FEATURES, band_mae))


def save_results(run_name, history, prediction, data):
    """Save the history and test predictions, the loss-curve plot and one prediction plot per band."""
    with open(os.path.join(path_to_save_models_results, f"{run_name}_history.pkl"), "wb") as file:
        pickle.dump(history, file)
    with open(os.path.join(path_to_save_models_results, f"{run_name}_prediction.pkl"), "wb") as file:
        pickle.dump(prediction, file)

    plt.figure(figsize = (12, 8))
    plt.plot(history['loss'], label = 'train loss')
    plt.plot(history['val_loss'], label = 'validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('MAE + L2 penalty (normalized)')
    plt.legend()
    plt.savefig(os.path.join(path_to_save_models_images, f"{run_name}_history.png"))
    plt.close()

    example = np.random.default_rng(SEED).integers(data.X_test.shape[0])
    ModelUtils.plot_each_column(data.X_test, data.y_test, prediction, PREDICTED_FEATURES, len(CONSTANT_FEATURES),
                                f"{run_name}_prediction", example_index = example)
    print(f"Saved {len(PREDICTED_FEATURES)} prediction plots for test example {example}")


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(SEED)
    os.makedirs(path_to_save_models_results, exist_ok = True)
    os.makedirs(path_to_save_models_images, exist_ok = True)
    print(f"Devices: {[device.name for device in tf.config.list_physical_devices()]}")

    data = load_and_prepare_data(path_to_data, INPUT_WIDTH, LABEL_WIDTH, SHIFT, STRIDE, VAL_RATIO, TEST_RATIO)
    print(f"Train: {data.X_train.shape} -> {data.y_train.shape}")
    print(f"Validation: {data.X_val.shape} -> {data.y_val.shape}")
    print(f"Test: {data.X_test.shape} -> {data.y_test.shape}")

    run_name = f"{args.model}_{INPUT_WIDTH}to{LABEL_WIDTH}_stride{STRIDE}_bs{BATCH_SIZE}_lr{LEARNING_RATE}"
    model = build_model(args.model, data.X_train[:1])
    history = train(model, data, run_name, args.epochs, args.patience)

    prediction, test_mae, band_mae = evaluate(model, data)
    print(f"Test MAE (normalized): {test_mae:.5f}")
    print(f"Best val MAE (normalized): {min(history['val_mean_absolute_error']):.5f}")  # val_loss also includes the L2 penalty
    print("Test MAE per band (original units):")
    for band, mae in band_mae.items():
        print(f"    {band:<10} {mae:.5g}")

    save_results(run_name, history, prediction, data)


if __name__ == "__main__":
    main()
