import numpy as np
import librosa
import soundfile as sf
from scipy.signal import lfilter

# Load an audio file
def load_audio(file_path):
    audio, sample_rate = librosa.load('output1.wav', sr=None)
    return audio, sample_rate

# Save audio to file
def save_audio(output_path, audio, sample_rate):
    sf.write(output_path, audio, sample_rate)

# Pitch shifting
def pitch_shift(audio, sample_rate, pitch_factor):
    return librosa.effects.pitch_shift(audio, sample_rate, n_steps=pitch_factor)

# Voice modulation (adding tremolo effect)
def tremolo(audio, sample_rate, mod_freq=5.0, mod_depth=0.5):
    t = np.linspace(0, len(audio) / sample_rate, num=len(audio))
    modulator = 1 + mod_depth * np.sin(2 * np.pi * mod_freq * t)
    return audio * modulator

# High-pass filter for robotic effect
def high_pass_filter(audio, sample_rate, cutoff=1000):
    # Design a high-pass filter
    nyquist = 0.5 * sample_rate
    norm_cutoff = cutoff / nyquist
    b = [1, -1]  # Simple high-pass filter coefficients
    a = [1, -norm_cutoff]
    return lfilter(b, a, audio)

# Full voice changer pipeline
def voice_changer(input_file, output_file, pitch_factor=2, mod_freq=5.0, mod_depth=0.5, apply_filter=False):
    # Load the audio
    audio, sample_rate = load_audio(input_file)

    # Apply pitch shifting
    pitched_audio = pitch_shift(audio, sample_rate, pitch_factor)

    # Apply voice modulation (tremolo effect)
    modulated_audio = tremolo(pitched_audio, sample_rate, mod_freq, mod_depth)

    # Apply a high-pass filter for a robotic effect, if needed
    if apply_filter:
        modulated_audio = high_pass_filter(modulated_audio, sample_rate)

    # Save the processed audio
    save_audio(output_file, modulated_audio, sample_rate)

# Usage example:
# voice_changer('input_audio.wav', 'output_audio.wav', pitch_factor=2, mod_freq=4.0, mod_depth=0.6, apply_filter=True)

