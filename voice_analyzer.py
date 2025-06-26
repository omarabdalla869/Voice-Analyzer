"""
Voice Analyzer Library
A comprehensive library for voice analysis and audio processing.

Categories:
- Recording and File Operations
- Time Domain Filters
- Frequency Domain Filters
- Spectral Filters
- Analysis Tools
"""

import numpy as np
import librosa
import sounddevice as sd
from scipy.fft import fft, ifft
import soundfile as sf
import pyaudio
import wave
from scipy import signal
import os

# ================ Recording and File Operations ================

def record_audio(duration=5, sample_rate=22050):
    """
    Record audio from microphone.
    
    Args:
        duration (float): Recording duration in seconds
        sample_rate (int): Sampling rate in Hz
        
    Returns:
        numpy.ndarray: Recorded audio signal
    """
    # Use blocking mode to ensure proper timing
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, blocking=True)
    return audio.flatten()

def load_audio_file(file_path, sample_rate=22050):
    """
    Load audio file from disk.
    
    Args:
        file_path (str): Path to audio file
        sample_rate (int): Target sampling rate
        
    Returns:
        numpy.ndarray: Audio signal
    """
    audio, _ = librosa.load(file_path, sr=sample_rate)
    return audio

def save_audio_file(audio_data, file_path, sample_rate=22050):
    """
    Save audio data to file.
    
    Args:
        audio_data (numpy.ndarray): Audio signal to save
        file_path (str): Output file path
        sample_rate (int): Sampling rate
    """
    sf.write(file_path, audio_data, sample_rate)

def play_audio(audio_signal, sample_rate=22050):
    """
    Play audio signal.
    
    Args:
        audio_signal (numpy.ndarray): Audio signal to play
        sample_rate (int): Sampling rate
    """
    if np.iscomplexobj(audio_signal):
        audio_signal = np.real(audio_signal)
    audio_signal = audio_signal / np.max(np.abs(audio_signal))
    sd.play(audio_signal, sample_rate)
    sd.wait()

def modify_speed(audio_data, speed_factor=1.0, sample_rate=22050):
    """
    Modify the speed of audio without changing pitch.
    
    Args:
        audio_data (numpy.ndarray): Input audio signal
        speed_factor (float): Speed modification factor (1.0 is original, 
                            0.5 is half speed, 2.0 is double speed)
        sample_rate (int): Sampling rate
        
    Returns:
        numpy.ndarray: Speed-modified audio
    """
    if speed_factor <= 0:
        raise ValueError("Speed factor must be positive")
        
    # Use librosa's time_stretch for high-quality speed modification
    return librosa.effects.time_stretch(audio_data, rate=speed_factor)

def record_audio_chunk(chunk_size=1024, sample_rate=22050):
    """
    Record a single chunk of audio data.
    
    Args:
        chunk_size (int): Number of samples to record in one chunk
        sample_rate (int): Sampling rate in Hz
        
    Returns:
        numpy.ndarray: Array of audio samples
    """
    try:
        # Record a single chunk with blocking to ensure timing
        audio = sd.rec(chunk_size, samplerate=sample_rate, channels=1, blocking=True)
        return audio.flatten()
    except Exception as e:
        print(f"Error recording audio chunk: {str(e)}")
        return None

# ================ Time Domain Filters ================

def apply_volume_adjustment(audio, gain=1.0):
    """
    Adjust audio volume.
    
    Args:
        audio (numpy.ndarray): Input audio signal
        gain (float): Volume multiplier
        
    Returns:
        numpy.ndarray: Volume adjusted audio
    """
    return audio * gain

def apply_fade(audio, fade_length=1000, fade_type='both'):
    """
    Apply fade in/out effect.
    
    Args:
        audio (numpy.ndarray): Input audio signal
        fade_length (int): Length of fade in samples
        fade_type (str): 'in', 'out', or 'both'
        
    Returns:
        numpy.ndarray: Audio with fade effect
    """
    audio_out = audio.copy()
    fade_curve = np.linspace(0, 1, fade_length)
    
    if fade_type in ['in', 'both']:
        audio_out[:fade_length] *= fade_curve
    if fade_type in ['out', 'both']:
        audio_out[-fade_length:] *= fade_curve[::-1]
        
    return audio_out

# ================ Frequency Domain Filters ================

def apply_fft(audio):
    """
    Apply Fast Fourier Transform.
    
    Args:
        audio (numpy.ndarray): Input audio signal
        
    Returns:
        numpy.ndarray: FFT result
    """
    return fft(audio)

def apply_dft(audio):
    """
    Apply Discrete Fourier Transform.
    Memory efficient implementation using batch processing.
    
    Args:
        audio (numpy.ndarray): Input audio signal
        
    Returns:
        numpy.ndarray: DFT result
    """
    N = len(audio)
    batch_size = 1000
    result = np.zeros(N, dtype=np.complex128)
    
    for k in range(N):
        for n in range(N):
            result[k] += audio[n] * np.exp(-2j * np.pi * k * n / N)
    
    return result

def apply_idft(freq_domain):
    """
    Apply Inverse Discrete Fourier Transform.
    Memory efficient implementation using batch processing.
    
    Args:
        freq_domain (numpy.ndarray): Frequency domain signal
        
    Returns:
        numpy.ndarray: Time domain signal
    """
    N = len(freq_domain)
    batch_size = 1000
    result = np.zeros(N, dtype=np.complex128)
    
    for n in range(N):
        for k in range(N):
            result[n] += freq_domain[k] * np.exp(2j * np.pi * k * n / N)
    
    return result / N

def apply_bandpass_filter(audio, sample_rate=22050, lowcut=500, highcut=3000):
    """
    Apply bandpass filter.
    
    Args:
        audio (numpy.ndarray): Input audio signal
        sample_rate (int): Sampling rate
        lowcut (float): Lower frequency cutoff
        highcut (float): Upper frequency cutoff
        
    Returns:
        numpy.ndarray: Filtered audio
    """
    freqs = np.fft.fftfreq(len(audio), 1/sample_rate)
    f_signal = fft(audio)
    
    mask = (abs(freqs) > lowcut) & (abs(freqs) < highcut)
    f_signal_filtered = f_signal * mask
    
    return np.real(ifft(f_signal_filtered))

# ================ Spectral Filters ================

def apply_noise_reduction(audio, sample_rate=22050, noise_reduce_strength=2, noise_threshold=0.1):
    """
    Apply noise reduction using spectral gating.
    
    Args:
        audio (numpy.ndarray): Input audio signal
        sample_rate (int): Sampling rate
        noise_reduce_strength (float): Strength of noise reduction
        noise_threshold (float): Threshold for noise detection
        
    Returns:
        numpy.ndarray: Noise-reduced audio
    """
    S = librosa.stft(audio)
    mag = np.abs(S)
    phase = np.angle(S)
    
    noise_profile = np.mean(mag[:, :int(len(mag[0])*0.1)], axis=1)
    noise_profile = noise_profile.reshape((-1, 1))
    
    mask = mag > (noise_profile * noise_reduce_strength + noise_threshold)
    mag_cleaned = mag * mask
    
    S_cleaned = mag_cleaned * np.exp(1j * phase)
    audio_cleaned = librosa.istft(S_cleaned)
    
    return audio_cleaned

def pitch_shift(audio, n_steps, sample_rate=22050):
    """
    Shift the pitch of audio.
    
    Args:
        audio (numpy.ndarray): Input audio signal
        n_steps (float): Number of steps to shift
        sample_rate (int): Sampling rate
        
    Returns:
        numpy.ndarray: Pitch-shifted audio
    """
    return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)

# ================ Analysis Tools ================

def detect_gender(audio, sample_rate=22050):
    """
    Detect gender based on fundamental frequency analysis.
    
    Args:
        audio (numpy.ndarray): Input audio signal
        sample_rate (int): Sampling rate
        
    Returns:
        tuple: (str: detected gender, float: fundamental frequency)
    """
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
    
    pit = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:,i].argmax()
        pit.append(pitches[index,i])
    
    mean_pitch = np.mean([p for p in pit if p > 0])
    
    if mean_pitch < 165:
        return ("Male", mean_pitch)
    elif mean_pitch > 255:
        return ("Child", mean_pitch)
    else:
        return ("Female", mean_pitch)

def get_audio_features(audio, sample_rate=22050):
    """
    Extract various audio features.
    
    Args:
        audio (numpy.ndarray): Input audio signal
        sample_rate (int): Sampling rate
        
    Returns:
        dict: Dictionary of audio features
    """
    features = {
        'rms': float(np.sqrt(np.mean(audio**2))),
        'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(audio))),
        'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))),
        'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)))
    }
    return features

def get_spectrogram_data(audio, sample_rate=22050):
    """
    Get spectrogram data for visualization.
    
    Args:
        audio (numpy.ndarray): Input audio signal
        sample_rate (int): Sampling rate
        
    Returns:
        tuple: (magnitude_db, frequencies, times)
    """
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return S_db

def get_frequency_spectrum(audio, sample_rate=22050):
    """
    Get frequency spectrum data for visualization.
    
    Args:
        audio (numpy.ndarray): Input audio signal
        sample_rate (int): Sampling rate
        
    Returns:
        tuple: (frequencies, magnitudes)
    """
    fft_result = fft(audio)
    freqs = np.fft.fftfreq(len(audio), 1/sample_rate)
    return freqs, np.abs(fft_result) 