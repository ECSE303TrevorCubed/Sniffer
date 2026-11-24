"""Train the Neural Net (MLP) odor classifier and export it as ONNX."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .classify3_odors import save_onnx
from .data import load_dataset


def export_weight_csvs(features, scaler, mlp, data_dir):
    """Export scaler parameters and per-layer weight matrices as CSVs."""
    pd.DataFrame({
        "Feature": features,
        "Mean_u": scaler.mean_,
        "Std_sigma": scaler.scale_,
    }).to_csv(os.path.join(data_dir, "nn_scaler.csv"), index=False)

    names = ["nn_scaler.csv"]
    prev_labels = list(features)
    for i, (w, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        if i < len(mlp.coefs_) - 1:
            col_labels = [f"H{i+1}_N{j}" for j in range(w.shape[1])]
        else:
            col_labels = list(mlp.classes_)
        df_l = pd.DataFrame(w, index=prev_labels, columns=col_labels)
        df_l.loc["Bias"] = b
        fname = f"nn_layer{i+1}_{w.shape[0]}x{w.shape[1]}.csv"
        df_l.to_csv(os.path.join(data_dir, fname))
        names.append(fname)
        prev_labels = col_labels

    print(f"Saved tabular CSVs to {data_dir}/: {', '.join(names)}")


def main(args):
    df = load_dataset(args.data_dir)

    feature_cols = [col for col in df.columns if col.startswith("Gas_Step_")]
    X = df[feature_cols]
    y = df["Label"]

    # Same 80/20 stratified split for fair comparison with Random Forest
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Neural Network Pipeline (StandardScaler + 2-Layer MLP)
    nn_pipeline = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42,
        ),
    )

    nn_pipeline.fit(X_train, y_train)
    y_pred = nn_pipeline.predict(X_test)

    print(f"Neural Net Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    model_path = os.path.join(args.out_dir, "odor_model_nn.onnx")
    sidecar_path = save_onnx(nn_pipeline, X_train, feature_cols, model_path)
    print(f"Saved Neural Net ONNX model to {model_path}")
    print(f"Saved model metadata to {sidecar_path}")

    scaler = nn_pipeline.named_steps["standardscaler"]
    mlp = nn_pipeline.named_steps["mlpclassifier"]
    export_weight_csvs(feature_cols, scaler, mlp, args.data_dir)

    # Plot 1: Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Purples")
    plt.title("Neural Network (MLP) Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(args.out_dir, "confusion_matrix_nn.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved NN confusion matrix to: {cm_path}")

    # Plot 2: Training Loss Curve
    plt.figure(figsize=(8, 4))
    plt.plot(mlp.loss_curve_, linewidth=2, color="purple")
    plt.title("Neural Network Training Loss Curve")
    plt.xlabel("Epoch (Iteration)")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    loss_path = os.path.join(args.out_dir, "nn_loss_curve.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"Saved NN loss curve to: {loss_path}")
