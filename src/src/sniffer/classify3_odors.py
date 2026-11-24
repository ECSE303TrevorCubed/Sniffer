"""Train the Random Forest odor classifier and export it as ONNX."""

import json
import os
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from onnx import ModelProto
from skl2onnx import to_onnx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

from .data import load_dataset


def save_onnx(model, X_train, feature_cols, model_path):
    # Feed a float32 sample so the ONNX input is a single float tensor.
    # to_onnx() is unannotated upstream; it returns a ModelProto here.
    onnx_model = cast(
        ModelProto,
        to_onnx(
            model,
            X_train.to_numpy(dtype=np.float32),
            options={"zipmap": False},
        ),
    )
    with open(model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    sidecar_path = os.path.splitext(model_path)[0] + ".json"
    with open(sidecar_path, "w") as f:
        json.dump(
            {
                "features": list(feature_cols),
                "classes": [str(c) for c in model.classes_],
            },
            f,
            indent=2,
        )
    return sidecar_path


def main(args):
    df = load_dataset(args.data_dir)

    feature_cols = [col for col in df.columns if col.startswith("Gas_Step_")]
    X = df[feature_cols]
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    model_path = os.path.join(args.out_dir, "odor_model.onnx")
    sidecar_path = save_onnx(clf, X_train, feature_cols, model_path)
    print(f"Saved ONNX model to {model_path}")
    print(f"Saved model metadata to {sidecar_path}")

    # Plot 1: Average 10-Step Resistance Fingerprint per Class
    plt.figure(figsize=(10, 5))
    for label_name, group in df.groupby("Label"):
        mean_curve = group[feature_cols].mean()
        plt.plot(
            range(len(feature_cols)),
            mean_curve.to_numpy(dtype=float),
            marker="o",
            linewidth=2,
            label=label_name,
        )

    plt.title("BME688 10-Step Heater Resistance Fingerprint by Odor Class")
    plt.xlabel("Heater Step Index (0 to 9)")
    plt.ylabel("Gas Resistance (kOhm)")
    plt.xticks(
        range(len(feature_cols)),
        [c.replace("Gas_", "").replace("(kOhm)", "") for c in feature_cols],
        rotation=30,
    )
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    fingerprint_path = os.path.join(args.out_dir, "odor_fingerprints.png")
    plt.savefig(fingerprint_path, dpi=150)
    plt.close()
    print(f"Saved fingerprint chart to: {fingerprint_path}")

    # Plot 2: Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()

    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix chart to: {cm_path}")
