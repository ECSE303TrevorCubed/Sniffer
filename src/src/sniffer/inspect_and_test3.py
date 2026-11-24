"""Inspect the exported ONNX models and replay the dataset offline (no sensor)."""

import json
import os
from typing import cast

import numpy as np
import onnx
import onnxruntime as rt
from onnx import numpy_helper

from .data import load_dataset


def inspect_model(model_path, json_export_path=None):
    sidecar_path = os.path.splitext(model_path)[0] + ".json"
    with open(sidecar_path, "r") as f:
        meta = json.load(f)

    sess = rt.InferenceSession(model_path)
    model = onnx.load(model_path)

    print(f"=== {os.path.basename(model_path)} ===")
    print("Features:", meta["features"])
    print("Classes:", meta["classes"])
    for inp in sess.get_inputs():
        print(f"Input: {inp.name} {inp.shape} ({inp.type})")
    for out in sess.get_outputs():
        print(f"Output: {out.name} {out.shape} ({out.type})")

    tensors = []
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        print(f"Tensor {init.name}: shape {list(arr.shape)}")
        if arr.dtype.kind in "iuf":
            values = np.round(arr.astype(float), 6).tolist()
        else:
            values = [str(v) for v in arr.tolist()]
        tensors.append({
            "name": init.name,
            "shape": list(arr.shape),
            "values": values,
        })

    if json_export_path:
        with open(json_export_path, "w") as f:
            json.dump(
                {
                    "model": os.path.basename(model_path),
                    "features": meta["features"],
                    "classes": meta["classes"],
                    "tensors": tensors,
                },
                f,
                indent=2,
            )
        print(f"Exported readable weights to {json_export_path}")

    return sess, meta


def replay_accuracy(sess, X):
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: X.to_numpy(dtype=np.float32)})
    return cast(np.ndarray, outputs[0]).ravel()


def main(args):
    nn_model_path = os.path.join(args.out_dir, "odor_model_nn.onnx")
    rf_model_path = os.path.join(args.out_dir, "odor_model.onnx")
    json_export_path = os.path.join(args.out_dir, "nn_weights_export.json")

    nn_sess, nn_meta = inspect_model(nn_model_path, json_export_path)
    print()
    rf_sess, _ = inspect_model(rf_model_path)

    # Offline replay test on saved master CSVs (no sensor required)
    df = load_dataset(args.data_dir)
    X_all = df[nn_meta["features"]]
    y_true = df["Label"].to_numpy()

    nn_preds = replay_accuracy(nn_sess, X_all)
    rf_preds = replay_accuracy(rf_sess, X_all)

    print(f"\n=== OFFLINE REPLAY TEST ({len(df)} total rows) ===")
    print(f"Neural Network Accuracy: {(nn_preds == y_true).mean() * 100:.2f}%")
    print(f"Random Forest Accuracy:  {(rf_preds == y_true).mean() * 100:.2f}%")
