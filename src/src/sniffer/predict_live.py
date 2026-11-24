"""Run live odor predictions from the BME688 using an ONNX model."""

import json
import os
import time
from typing import cast

import numpy as np
import onnxruntime as rt


def main(args):
    import bme68xConstants as cst
    from bme68x import BME68X

    sidecar_path = os.path.splitext(args.model)[0] + ".json"
    with open(sidecar_path, "r") as f:
        meta = json.load(f)
    feature_cols = meta["features"]
    classes = meta["classes"]

    sess = rt.InferenceSession(args.model)
    input_name = sess.get_inputs()[0].name
    print(f"Loaded ONNX model from: {args.model} | Classes: {classes}")

    with open(args.config, "r") as f:
        cfg = json.load(f)

    profile = cfg["configBody"]["heaterProfiles"][0]
    vectors = profile["temperatureTimeVectors"]

    heater_temps = [int(step[0]) for step in vectors]
    heater_durs = [int(step[1]) for step in vectors]

    print(f"Loaded profile '{profile.get('id', 'default')}' with {len(heater_temps)} steps:")
    print(f"Temps (C): {heater_temps}")
    print(f"Durations (x140ms): {heater_durs}")

    sensor = BME68X(cst.BME68X_I2C_ADDR_HIGH, 0)
    sensor.set_heatr_conf(
        cst.BME68X_ENABLE,
        heater_temps,
        heater_durs,
        cst.BME68X_PARALLEL_MODE
    )

    num_steps = len(heater_temps)
    if num_steps != len(feature_cols):
        raise ValueError(
            f"Heater profile has {num_steps} steps but model expects {len(feature_cols)} features"
        )

    header = "Timestamp,Model,Actual_Label,Sweep,Prediction,Confidence(%)\n"

    file_exists = os.path.exists(args.log)
    print(f"\nLogging up to {args.sweeps} live sweeps for '{args.label}' to {args.log} (Ctrl+C to stop early)...\n")

    current_sweep = {}
    sweep_count = 0
    model_name = os.path.basename(args.model)

    with open(args.log, "a") as f:
        if not file_exists or os.path.getsize(args.log) == 0:
            f.write(header)

        try:
            while sweep_count < args.sweeps:
                data_list = sensor.get_data()

                if data_list:
                    if isinstance(data_list, dict):
                        data_list = [data_list]

                    for sample in data_list:
                        gas_res = sample.get("raw_gas", 0.0)
                        step_idx = sample.get("gas_index", -1)

                        if gas_res > 0 and 0 <= step_idx < num_steps:
                            current_sweep[step_idx] = gas_res

                            if len(current_sweep) == num_steps:
                                X_live = np.array(
                                    [[current_sweep[i] for i in range(num_steps)]],
                                    dtype=np.float32,
                                )
                                outputs = sess.run(None, {input_name: X_live})
                                labels = cast(np.ndarray, outputs[0])
                                probs = cast(np.ndarray, outputs[1])
                                pred = str(labels[0])
                                probs_row = probs[0]
                                conf = float(max(probs_row)) * 100.0
                                prob_str = " | ".join(f"{c}: {p * 100:.1f}%" for c, p in zip(classes, probs_row))

                                sweep_count += 1
                                print(f"Sweep #{sweep_count}/{args.sweeps} | Pred: {pred.upper()} ({conf:.1f}%) [{prob_str}]")
                                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{model_name},{args.label},{sweep_count},{pred},{conf:.1f}\n")
                                f.flush()
                                current_sweep.clear()

                                if sweep_count >= args.sweeps:
                                    print(f"\nReached {args.sweeps}-sweep limit (prevents C-driver memory leak). Finished cleanly!")
                                    break

                time.sleep(0.1)

        except KeyboardInterrupt:
            print(f"\nStopped early by user: {sweep_count} predictions logged to {args.log}.")
