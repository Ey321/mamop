import json
import os
import time
import numpy as np
from picoscope import ps2000a
from matplotlib import pyplot as plt
import matplotlib.transforms as transforms
import winsound
import threading
import enum
from dataclasses import dataclass

FREQUENCY = 700e3
CHANNEL_A_AMPLITUDE = 1.5
CHANNEL_B_AMPLITUDE = 1e-3
CHANNEL_B_AMPLITUDE = 1
SAMPLING_INTERVAL = 4e-9  # 100ns
SAMPLING_DURATION = 0.1e-3  # 1ms

SOUND_FILE = "sounds/beep.wav"
SOUND_FILE = None
THRESHOLD = 0.06

is_sound_playing = False

recorded_data = []

AVERAGE_WINDOW = 30
BUFFER_LENGTH = 100


@dataclass
class Measurement:
    key: str
    mean: float = 0
    std: float = 1


measurements = {
    "nothing": Measurement("n"),
    # "phone": Measurement("p"),
    # "screw": Measurement("c"),
    "1": Measurement("1"),
    "2": Measurement("2"),
    # "3": Measurement("3"),
}


class State:
    RUNNING = 0
    RECORDING = 1


current_state = State.RUNNING
current_measurement = None


def start_sound():
    global is_sound_playing
    if not is_sound_playing:
        is_sound_playing = True
        play_sound_thread()


def stop_sound():
    global is_sound_playing
    if is_sound_playing:
        is_sound_playing = False
        winsound.PlaySound(None, winsound.SND_FILENAME)


def play_sound_thread():
    global is_sound_playing
    winsound.PlaySound(SOUND_FILE, winsound.SND_FILENAME | winsound.SND_LOOP | winsound.SND_ASYNC)


def measure_and_plot_single_block(ps, n_samples, sampling_interval, delay_to_settle=2):
    time.sleep(delay_to_settle)
    data_a, data_b = measure_block(ps, n_samples)
    data_time_axis = np.arange(n_samples) * sampling_interval

    plt.figure()
    plt.plot(data_time_axis, data_a, label="Channel A")
    plt.plot(data_time_axis, data_b, label="Channel B")
    plt.grid(True, which='major')
    plt.title("channel B RMS: %.3f mV" % (np.std(data_b) * 1e3))
    plt.ylabel("Voltage (V)")
    plt.xlabel("Time (ms)")
    plt.legend()
    plt.show()


def measure_block(ps, n_samples):
    ps.runBlock()
    ps.waitReady()
    ps.runBlock()
    ps.waitReady()
    data_a = ps.getDataV('A', n_samples, returnOverflow=False)
    data_b = ps.getDataV('B', n_samples, returnOverflow=False)
    return data_a, data_b


def set_wave_on(ps):
    ps.setSigGenBuiltInSimple(offsetVoltage=0.0, pkToPk=2.0, waveType="Sine",
                              frequency=FREQUENCY)


def set_wave_off(ps):
    ps.setSigGenBuiltInSimple(offsetVoltage=0.0, pkToPk=0.0, waveType="Sine",
                              frequency=FREQUENCY)


def init_measurements(ps):
    (actual_sampling_interval, n_samples, maxSamples) = \
        ps.setSamplingInterval(SAMPLING_INTERVAL, SAMPLING_DURATION)
    ps.setSimpleTrigger('B', 10e-3, 'Falling', timeout_ms=100, enabled=True)
    # ps.setSigGenBuiltInSimple(offsetVoltage=0.0, pkToPk=4.0, waveType="Sine",
    #                           frequency=FREQUENCY)
    set_wave_on(ps)
    channel_a_range = ps.setChannel('A', 'AC', CHANNEL_A_AMPLITUDE, 0.0,
                                    enabled=True, BWLimited=False)
    channel_b_range = ps.setChannel('B', 'AC', CHANNEL_B_AMPLITUDE, 0.0,
                                    enabled=True, BWLimited=False)
    return actual_sampling_interval, n_samples


def pdf(mean, std, value):
    return 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((value - mean) / std) ** 2)


def on_press(event):
    global current_state, current_measurement
    print("pressed")

    total = len(recorded_data)
    if total > 30:
        if current_state == State.RECORDING:
            mean = np.mean(recorded_data[total // 4:-total // 4])
            std = np.std(recorded_data[total // 4:-total // 4])
            recorded_data.clear()
            current_measurement.mean = mean
            current_measurement.std = std

    if event.key == "r":
        current_state = State.RUNNING
        print("running")
    else:
        for key, measurement in measurements.items():
            if event.key == measurement.key:
                current_state = State.RECORDING
                current_measurement = measurement
                print(f"Recording {key}")
                break


def examplePS2000():
    print("Attempting to open...")
    ps = ps2000a.PS2000a()

    actual_sampling_interval, n_samples = init_measurements(ps)
    i = 0
    data = []
    data_a_store = []
    times = []
    running_average = []
    start_time = time.time()
    fig, ax = plt.subplots()
    while True:
        data_a, data_b = measure_block(ps, n_samples)
        # # plot data_b
        # plt.plot(data_b)
        # plt.show()
        # continue
        data_b_std = np.std(data_b)
        data_b_std = np.sqrt(np.mean(np.square(data_b)))
        data_a_std = np.sqrt(np.mean(np.square(data_a)))
        # print(f"Channel B RMS: {np.std(data_b) * 1e3:.3f} mV")
        data.append(data_b_std)
        data_a_store.append(data_a_std)
        times.append(time.time() - start_time)
        ax.cla()

        measured_amplitude = data_b_std
        if len(data) > 30:
            measured_amplitude = np.mean(data[-30:])

        if current_state == State.RUNNING:
            detected = "---"
            best_likelihood = 0
            for obj, measurement in measurements.items():
                likelihood = pdf(measurement.mean, measurement.std, measured_amplitude)
                if likelihood > best_likelihood:
                    detected = obj
                    best_likelihood = likelihood
                # ax.axhline(measurement.mean, color="gray", linestyle="--")
                # trans = transforms.blended_transform_factory(
                #     ax.get_yticklabels()[0].get_transform(), ax.transData)
                # ax.text(0, measurement.mean, obj, color="gray", transform=trans,
                #         ha="right", va="center")
            ax.set_title(f"Detecting {detected}, best likelihood: {best_likelihood:.3f}")
        else:
            ax.set_title("Recording")
            recorded_data.append(data[-1])

        if len(data) > BUFFER_LENGTH:
            data = data[-BUFFER_LENGTH:]
            times = times[-BUFFER_LENGTH:]
            data_a_store = data_a_store[-BUFFER_LENGTH:]
        ax.scatter(times, data, label="B")
        # ax.scatter(times, data_a_store, label="A")
        # ax.scatter(times, [i / j for i, j in zip(data, data_a_store)], label="B/A")
        # ax.legend()
        for obj, measurement in measurements.items():
            ax.axhline(measurement.mean, color="gray", linestyle="--")
            trans = transforms.blended_transform_factory(
                ax.get_yticklabels()[0].get_transform(), ax.transData)
            ax.text(0, measurement.mean, obj, color="gray", transform=trans,
                    ha="right", va="center")
        if len(data) > AVERAGE_WINDOW:
            running_average.append(np.mean(data[-AVERAGE_WINDOW:]))
            # ax.axhline(running_average[-1], color="gray", linestyle="--")
            if len(running_average) < BUFFER_LENGTH:
                ax.plot(times[-len(running_average):], running_average, color="red")
            else:
                ax.plot(times[-BUFFER_LENGTH:], running_average[-BUFFER_LENGTH:], color="red")
        # add labels to axis
        ax.set_ylabel("RMS Voltage (V)")
        ax.set_xlabel("Time (s)")
        fig.canvas.mpl_connect('key_press_event', on_press)
        set_wave_off(ps)
        plt.pause(0.02)
        set_wave_on(ps)
        ps.runBlock()
        ps.waitReady()
        time.sleep(0.02)
        if data_b_std > THRESHOLD:
            start_sound()
        else:
            stop_sound()

        i += 1


def add_to_path():
    picoinstallpath = os.path.normpath(r"C:\Program Files\Pico Technology\PicoScope 7 T&M Stable")
    if picoinstallpath not in os.environ['PATH']:
        print("Adding Pico Install to Path")
        os.environ['PATH'] = picoinstallpath + os.pathsep + os.environ['PATH']
    else:
        print("Pico Install Already on Path")


def main():
    pass


if __name__ == '__main__':
    # play_sound_thread()
    add_to_path()

    if os.path.isfile("measurements.json"):
        with open("measurements.json", "r") as f:
            data = json.load(f)
            for obj, measurement in measurements.items():
                if obj not in data:
                    continue
                measurement.mean = data[obj]["mean"]
                measurement.std = data[obj]["std"]
                measurement.key = data[obj]["key"]

    try:
        examplePS2000()
    except Exception as e:
        raise e
    finally:
        json_data = {}
        for obj, measurement in measurements.items():
            json_data[obj] = {"mean": measurement.mean,
                              "std": measurement.std,
                              "key": measurement.key}
        with open("measurements.json", "w") as f:
            json.dump(json_data, f)
