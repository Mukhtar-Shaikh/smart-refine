
import numpy as np
import matplotlib.pyplot as plt

# 1. Simulate 1 second of vibration measurements
sampling_rate = 1000  # Samples per second
duration = 1.0

time = np.arange(0, duration, 1 / sampling_rate)

# 2. Create a vibration-like signal
frequency = 50  # Hz
signal = np.sin(2 * np.pi * frequency * time)

# 3. Add a small amount of random noise
rng = np.random.default_rng(42)
noise = rng.normal(0, 0.15, size=len(time))
vibration = signal + noise

# 4. Calculate basic signal features
rms = np.sqrt(np.mean(vibration ** 2))
std_dev = np.std(vibration)
peak_to_peak = np.ptp(vibration)

print(f"Number of samples: {len(vibration)}")
print(f"RMS: {rms:.4f}")
print(f"Standard deviation: {std_dev:.4f}")
print(f"Peak-to-peak amplitude: {peak_to_peak:.4f}")

# 5. Plot the signal
plt.figure(figsize=(10, 4))
plt.plot(time, vibration, linewidth=0.8)
plt.title("Smart-Refine: Simulated Vibration Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()
