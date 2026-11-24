"""Collect labeled BME688 sweeps into a dataset CSV (runs on the sensor host)."""

import json
import os
import time


def main(args):
    import bme68xConstants as cst
    from bme68x import BME68X

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

    log_filename = os.path.join(args.data_dir, f"dataset_{args.label}_master.csv")

    num_steps = len(heater_temps)

    step_cols = ",".join([f"Gas_Step_{i}_{heater_temps[i]}C(kOhm)" for i in range(num_steps)])
    header = f"Label,Timestamp(ms),Raw_Temp,Raw_Hum,Raw_Press,{step_cols}\n"

    file_exists = os.path.exists(log_filename)
    print(f"\nAppending {args.sweeps} sweeps of '{args.label}' to {log_filename}...\n")

    current_sweep = {}
    last_env = (0.0, 0.0, 0.0)
    sweep_count = 0

    with open(log_filename, "a") as f:
        if not file_exists:
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
                            last_env = (
                                sample.get("raw_temperature", 0.0),
                                sample.get("raw_humidity", 0.0),
                                sample.get("raw_pressure", 0.0)
                            )

                            if len(current_sweep) == num_steps:
                                ts = int(time.time() * 1000)
                                temp, hum, press = last_env
                                gas_values = ",".join([f"{current_sweep[i]:.3f}" for i in range(num_steps)])

                                f.write(f"{args.label},{ts},{temp:.2f},{hum:.2f},{press:.2f},{gas_values}\n")
                                f.flush()

                                sweep_count += 1
                                print(f"Sweep #{sweep_count}/{args.sweeps} [{args.label}] | Step0(320C): {current_sweep[0]:.2f}k -> Step1(100C): {current_sweep[1]:.2f}k -> Step9(320C): {current_sweep[num_steps-1]:.2f}k")
                                current_sweep.clear()

                time.sleep(0.1)

            print(f"\nBatch complete: {sweep_count} sweeps appended to {log_filename}.")

        except KeyboardInterrupt:
            print(f"\nStopped early: {sweep_count} sweeps saved to {log_filename}.")
