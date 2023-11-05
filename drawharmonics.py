import numpy as np
import matplotlib.pyplot as plt


def plot_individual_sine_waves(times, frequencies, amplitudes):
    assert len(times) == len(frequencies) == len(amplitudes), "All input lists must have the same length"

    # Create a time array
    time_values = np.linspace(0, max(times), 500)  # 500 points in time

    plt.figure(figsize=(10, 4))

    for time, frequency, amplitude in zip(times, frequencies, amplitudes):
        # Compute the values of the sine wave at the given points in time
        wave_values = amplitude * np.cos(2 * np.pi * frequency * time_values)

        # Plot this sine wave
        plt.plot(time_values, wave_values, label=f'{frequency} Hz')

    plt.title('Individual sine waves')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    # plt.legend()
    # plt.grid(True)
    plt.show()


def plot_sine_waves(times, frequencies, amplitudes):
    assert len(times) == len(frequencies) == len(amplitudes), "All input lists must have the same length"

    # Create a time array
    time_values = np.linspace(0, max(times), 500)  # 500 points in time

    # Create an array to store the total wave values
    total_wave_values = np.zeros_like(time_values)

    for time, frequency, amplitude in zip(times, frequencies, amplitudes):
        # Compute the values of the sine wave at the given points in time
        wave_values = amplitude * np.cos(2 * np.pi * frequency * time_values)

        # Add these wave values to the total
        total_wave_values += wave_values

    # Plot the combined sine wave
    plt.figure(figsize=(10, 4))
    plt.plot(time_values, total_wave_values)
    plt.title('Combined sine waves')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    # plt.grid(True)
    plt.show()


def plot_sine_wave(time_in_seconds, frequency):
    # Compute the values of the sine wave at the given points in time
    time_values = np.linspace(0, time_in_seconds, 500) # 500 points in time
    wave_values = np.cos(2 * np.pi * frequency * time_values)

    # Plot the sine wave
    plt.figure(figsize=(10, 4))
    plt.plot(time_values, wave_values)
    plt.title(f'Sine wave with frequency {frequency} Hz')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    # plt.grid(True)
    plt.show()


def plot_custom_wave(times, amplitudes, frequency1, frequency2):
    assert len(times) == len(amplitudes), "Times and amplitudes lists must have the same length"

    # Create a time array
    time_values = np.linspace(0, max(times), 500)  # 500 points in time

    plt.figure(figsize=(10, 4))

    for time, amplitude in zip(times, amplitudes):
        # Compute the values of the wave at the given points in time
        wave_values = amplitude * (1 + np.cos(2 * np.pi * frequency1 * time_values)) * np.cos(
            2 * np.pi * frequency2 * time_values)

        # Plot this wave
        plt.plot(time_values, wave_values)

    plt.title('Custom wave')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    # plt.grid(True)
    plt.show()


plot_individual_sine_waves([0.5, 0.5], [20, 40], [1, 0.25])
# plot_sine_waves([0.5, 0.5], [20000, 20], [1, 1])
plot_custom_wave([0.5], [1], 20, 20000)
