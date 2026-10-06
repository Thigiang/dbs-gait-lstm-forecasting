"""
finetune.py -- Bayesian hyperparameter search (scikit-optimize) for the models in train.py.

Uses exactly the data and fixed settings of train.py (windows, stride, split, model type) and searches:
    lstm_units      32 - 512
    dense_units     16 - 128
    learning_rate   1e-5 - 1e-3 (log scale)
    l2_reg          1e-6 - 1e-2 (log scale)
    batch_size      32, 64 or 128

Each trial trains a fresh model with early stopping and is scored by its best validation MAE (the MAE alone,
without the L2 penalty, so trials with different l2_reg are compared fairly). Trials are short (--trial-epochs)
to keep the search affordable; train the chosen setting fully with train.py afterwards.

Outputs, in the results folder:
    <model>_search_trials.csv          one row per finished trial, written as the search goes
    <model>_best_hyperparams.json      the best setting, written at the end
The best setting is also printed in the form of train.py's settings block, ready to paste.

Run:
    python finetune.py --model feedback
    python finetune.py --model direct --trials 50 --trial-epochs 30
"""
import os
import csv
import json
import argparse
import numpy as np
import tensorflow as tf
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from configs import path_to_data, path_to_save_models_results
from dataset import load_and_prepare_data
from modelutils import ModelUtils
import train

SEARCH_SPACE = [
    Integer(32, 512, name = 'lstm_units'),
    Integer(16, 128, name = 'dense_units'),
    Real(1e-5, 1e-3, prior = 'log-uniform', name = 'learning_rate'),
    Real(1e-6, 1e-2, prior = 'log-uniform', name = 'l2_reg'),
    Categorical([32, 64, 128], name = 'batch_size'),
]
FAILED_SCORE = 1e3   # score for a trial whose loss became NaN, so the search avoids that region


def parse_args():
    parser = argparse.ArgumentParser(description = "Bayesian hyperparameter search for train.py's models.")
    parser.add_argument("--model", choices = ["feedback", "direct"], required = True, help = "model to tune")
    parser.add_argument("--trials", type = int, default = 30, help = "number of settings to try (default: 30)")
    parser.add_argument("--trial-epochs", type = int, default = 50, help = "maximum epochs per trial (default: 50)")
    parser.add_argument("--patience", type = int, default = 5, help = "early-stopping patience per trial (default: 5)")
    return parser.parse_args()


def run_trial(model_type, data, params, epochs, patience):
    """Train one model with `params`; return its best validation MAE and the number of epochs it ran."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(train.SEED)
    model = train.build_model(model_type, data.X_train[:1], lstm_units = params['lstm_units'],
                              dense_units = params['dense_units'], l2_reg = params['l2_reg'])
    history = ModelUtils.compile_and_fit(model, (data.X_train, data.y_train), (data.X_val, data.y_val),
                                         (params['learning_rate'], epochs), patience = patience,
                                         batch_size = params['batch_size'], shuffle = train.SHUFFLE,
                                         use_early_stopping = True)
    val_mae = history.history['val_mean_absolute_error']
    score = FAILED_SCORE if np.isnan(val_mae).any() else float(min(val_mae))
    return score, len(val_mae)


def main():
    args = parse_args()
    os.makedirs(path_to_save_models_results, exist_ok = True)
    trials_path = os.path.join(path_to_save_models_results, f"{args.model}_search_trials.csv")
    best_path = os.path.join(path_to_save_models_results, f"{args.model}_best_hyperparams.json")
    param_names = [dim.name for dim in SEARCH_SPACE]

    data = load_and_prepare_data(path_to_data, train.INPUT_WIDTH, train.LABEL_WIDTH, train.SHIFT, train.STRIDE,
                                 train.VAL_RATIO, train.TEST_RATIO)
    print(f"Train: {data.X_train.shape}, Validation: {data.X_val.shape}")

    with open(trials_path, "w", newline = "") as file:
        csv.writer(file).writerow(["trial"] + param_names + ["best_val_mae", "epochs_run"])

    trial_count = [0]

    @use_named_args(SEARCH_SPACE)
    def objective(**params):
        params = {name: value.item() if hasattr(value, "item") else value for name, value in params.items()}  # numpy -> python
        trial_count[0] += 1
        print(f"\n=== Trial {trial_count[0]}/{args.trials}: {params}")
        score, epochs_run = run_trial(args.model, data, params, args.trial_epochs, args.patience)
        print(f"=== Trial {trial_count[0]}: best val MAE {score:.5f} after {epochs_run} epochs")
        with open(trials_path, "a", newline = "") as file:
            csv.writer(file).writerow([trial_count[0]] + [params[name] for name in param_names] + [score, epochs_run])
        return score

    result = gp_minimize(objective, SEARCH_SPACE, n_calls = args.trials,
                         n_initial_points = min(10, args.trials), random_state = train.SEED)

    best = {name: value.item() if hasattr(value, "item") else value for name, value in zip(param_names, result.x)}
    with open(best_path, "w") as file:
        json.dump({"model": args.model, "best_val_mae": float(result.fun), **best}, file, indent = 2)

    print(f"\nBest val MAE {result.fun:.5f}. Saved to {best_path}")
    print("Paste into the settings block of train.py:")
    print(f"LSTM_UNITS = {best['lstm_units']}")
    print(f"DENSE_UNITS = {best['dense_units']}")
    print(f"L2_REG = {best['l2_reg']:.3g}")
    print(f"LEARNING_RATE = {best['learning_rate']:.3g}")
    print(f"BATCH_SIZE = {best['batch_size']}")


if __name__ == "__main__":
    main()
