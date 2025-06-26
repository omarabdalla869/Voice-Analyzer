import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import voice_analyzer as va
import threading
import queue
import numpy as np
from scipy.fft import fft
import os
import time
from matplotlib.animation import FuncAnimation

class ModernButton(ttk.Button):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.style = ttk.Style()
        self.style.configure('Modern.TButton',
                           padding=6,
                           relief="flat",
                           background="#4CAF50",
                           foreground="white")

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        
        # Create a canvas and scrollbar
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        
        # Create the scrollable frame
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Add the frame to the canvas
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Configure canvas to expand horizontally
        self.canvas.bind('<Configure>', self.resize_frame)
        
        # Configure scrollbar
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack widgets
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Bind mouse wheel
        self.bind_mouse_wheel()
        
    def resize_frame(self, event):
        # Resize the inner frame to match canvas width
        self.canvas.itemconfig(self.canvas_frame, width=event.width)
        
    def bind_mouse_wheel(self):
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
    def unbind_mouse_wheel(self):
        self.canvas.unbind_all("<MouseWheel>")

class VoiceAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Analyzer")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)  # Set minimum window size
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure('Modern.TFrame', background='#f0f0f0')
        self.style.configure('Modern.TLabel', background='#f0f0f0', font=('Segoe UI', 10))
        self.style.configure('Modern.TButton',
                           padding=6,
                           relief="flat",
                           background="#4CAF50",
                           foreground="white")
        self.style.configure('Modern.TLabelframe', background='#f0f0f0')
        self.style.configure('Modern.TLabelframe.Label', background='#f0f0f0', font=('Segoe UI', 10, 'bold'))
        
        # Initialize variables
        self.initialize_variables()
        
        # Create main frame with modern style
        self.main_frame = ttk.Frame(self.root, style='Modern.TFrame', padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create resizable layout
        self.create_resizable_layout()
        
        # Configure grid weights for resizing
        self.configure_grid_weights()

    def initialize_variables(self):
        self.live_plot_active = False
        self.audio_data = np.array([])
        self.processed_audio = None
        self.max_points = 10000
        self.recording = False
        self.queue = queue.Queue()
        self.ani = None
        self.plot_line = None
        self.speed_factor = tk.StringVar(value="1.0")

    def create_resizable_layout(self):
        # Create left panel (controls) with scrolling
        self.left_panel = ttk.Frame(self.main_frame, style='Modern.TFrame')
        self.left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Create scrollable frame for controls
        self.scrollable_controls = ScrollableFrame(self.left_panel)
        self.scrollable_controls.pack(fill="both", expand=True)
        
        # Create control panel inside scrollable frame
        self.create_control_panel(self.scrollable_controls.scrollable_frame)
        
        # Create right panel (display)
        self.right_panel = ttk.Frame(self.main_frame, style='Modern.TFrame')
        self.right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create display area
        self.create_display_area()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, style='Modern.TLabel')
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=10, pady=5)

    def configure_grid_weights(self):
        # Configure main window grid weights
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Configure main frame grid weights
        self.main_frame.grid_columnconfigure(1, weight=3)  # Display area takes more space
        self.main_frame.grid_columnconfigure(0, weight=1)  # Controls take less space
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # Configure panels grid weights
        self.left_panel.grid_columnconfigure(0, weight=1)
        self.left_panel.grid_rowconfigure(0, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(0, weight=1)

    def create_control_panel(self, parent):
        control_frame = ttk.LabelFrame(parent, text="Controls", style='Modern.TLabelframe', padding="10")
        control_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add controls with pack instead of grid for better scrolling behavior
        sections = [
            self.create_recording_section,
            self.create_file_operations_section,
            self.create_time_domain_section,
            self.create_frequency_section,
            self.create_noise_reduction_section,
            self.create_pitch_section,
            self.create_speed_section,
            self.create_analysis_section,
            self.create_visualization_section,
            self.create_playback_section
        ]
        
        for section in sections:
            section(control_frame)
            ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

    def create_recording_section(self, parent):
        section = ttk.Frame(parent)
        section.pack(fill='x', pady=5)
        
        ttk.Label(section, text="Recording", style='Modern.TLabel', font=('Segoe UI', 10, 'bold')).pack(pady=5)
        
        self.record_button = ModernButton(section, text="Record", command=self.toggle_recording)
        self.record_button.pack(fill='x', pady=2)
        
        duration_frame = ttk.Frame(section)
        duration_frame.pack(fill='x', pady=2)
        ttk.Label(duration_frame, text="Duration (s):", style='Modern.TLabel').pack(side=tk.LEFT)
        self.duration_var = tk.StringVar(value="5")
        ttk.Entry(duration_frame, textvariable=self.duration_var, width=10).pack(side=tk.LEFT, padx=5)

    def create_file_operations_section(self, parent):
        section = ttk.Frame(parent)
        section.pack(fill='x', pady=5)
        
        ttk.Label(section, text="File Operations", style='Modern.TLabel', font=('Segoe UI', 10, 'bold')).pack(pady=5)
        ModernButton(section, text="Upload Audio", command=self.upload_audio).pack(fill='x', pady=2)
        ModernButton(section, text="Save Audio", command=self.save_audio).pack(fill='x', pady=2)

    def create_time_domain_section(self, parent):
        section = ttk.Frame(parent)
        section.pack(fill='x', pady=5)
        
        ttk.Label(section, text="Time Domain Filters", style='Modern.TLabel', font=('Segoe UI', 10, 'bold')).pack(pady=5)
        
        # Volume control
        volume_frame = ttk.Frame(section)
        volume_frame.pack(fill='x', pady=2)
        ttk.Label(volume_frame, text="Volume Gain:", style='Modern.TLabel').pack(side=tk.LEFT)
        self.volume_var = tk.StringVar(value="1.0")
        ttk.Entry(volume_frame, textvariable=self.volume_var, width=10).pack(side=tk.LEFT, padx=5)
        ModernButton(section, text="Adjust Volume", command=self.apply_volume).pack(fill='x', pady=2)
        
        # Fade control
        fade_frame = ttk.Frame(section)
        fade_frame.pack(fill='x', pady=2)
        ttk.Label(fade_frame, text="Fade Type:", style='Modern.TLabel').pack(side=tk.LEFT)
        self.fade_var = tk.StringVar(value="both")
        ttk.Combobox(fade_frame, textvariable=self.fade_var, values=["in", "out", "both"], width=8).pack(side=tk.LEFT, padx=5)
        ModernButton(section, text="Apply Fade", command=self.apply_fade).pack(fill='x', pady=2)

    def create_frequency_section(self, parent):
        section = ttk.Frame(parent)
        section.pack(fill='x', pady=5)
        
        ttk.Label(section, text="Frequency Filters", style='Modern.TLabel', font=('Segoe UI', 10, 'bold')).pack(pady=5)
        
        # Bandpass filter
        filter_frame = ttk.Frame(section)
        filter_frame.pack(fill='x', pady=2)
        ttk.Label(filter_frame, text="Bandpass (Hz):", style='Modern.TLabel').pack(side=tk.LEFT)
        self.lowcut_var = tk.StringVar(value="500")
        self.highcut_var = tk.StringVar(value="3000")
        ttk.Entry(filter_frame, textvariable=self.lowcut_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(filter_frame, text="-", style='Modern.TLabel').pack(side=tk.LEFT, padx=2)
        ttk.Entry(filter_frame, textvariable=self.highcut_var, width=6).pack(side=tk.LEFT, padx=2)
        ModernButton(section, text="Apply Bandpass", command=self.apply_bandpass).pack(fill='x', pady=2)

    def create_noise_reduction_section(self, parent):
        section = ttk.Frame(parent)
        section.pack(fill='x', pady=5)
        
        ttk.Label(section, text="Noise Reduction", style='Modern.TLabel', font=('Segoe UI', 10, 'bold')).pack(pady=5)
        
        # Strength control
        strength_frame = ttk.Frame(section)
        strength_frame.pack(fill='x', pady=2)
        ttk.Label(strength_frame, text="Strength:", style='Modern.TLabel').pack(side=tk.LEFT)
        self.noise_strength_var = tk.StringVar(value="2.0")
        ttk.Entry(strength_frame, textvariable=self.noise_strength_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # Threshold control
        threshold_frame = ttk.Frame(section)
        threshold_frame.pack(fill='x', pady=2)
        ttk.Label(threshold_frame, text="Threshold:", style='Modern.TLabel').pack(side=tk.LEFT)
        self.noise_threshold_var = tk.StringVar(value="0.1")
        ttk.Entry(threshold_frame, textvariable=self.noise_threshold_var, width=10).pack(side=tk.LEFT, padx=5)
        
        ModernButton(section, text="Apply Noise Reduction", command=self.apply_noise_reduction).pack(fill='x', pady=2)

    def create_pitch_section(self, parent):
        section = ttk.Frame(parent)
        section.pack(fill='x', pady=5)
        
        ttk.Label(section, text="Pitch Shift", style='Modern.TLabel', font=('Segoe UI', 10, 'bold')).pack(pady=5)
        
        pitch_frame = ttk.Frame(section)
        pitch_frame.pack(fill='x', pady=2)
        ttk.Label(pitch_frame, text="Steps:", style='Modern.TLabel').pack(side=tk.LEFT)
        self.pitch_var = tk.StringVar(value="2")
        ttk.Entry(pitch_frame, textvariable=self.pitch_var, width=10).pack(side=tk.LEFT, padx=5)
        
        ModernButton(section, text="Apply Pitch Shift", command=lambda: self.process_audio('pitch')).pack(fill='x', pady=2)

    def create_speed_section(self, parent):
        section = ttk.Frame(parent)
        section.pack(fill='x', pady=5)
        
        ttk.Label(section, text="Speed Control", style='Modern.TLabel', font=('Segoe UI', 10, 'bold')).pack(pady=5)
        
        speed_frame = ttk.Frame(section)
        speed_frame.pack(fill='x', pady=2)
        ttk.Label(speed_frame, text="Speed Factor:", style='Modern.TLabel').pack(side=tk.LEFT)
        self.speed_entry = ttk.Entry(speed_frame, textvariable=self.speed_factor, width=6)
        self.speed_entry.pack(side=tk.LEFT, padx=5)
        
        buttons_frame = ttk.Frame(section)
        buttons_frame.pack(fill='x', pady=2)
        ModernButton(buttons_frame, text="½×", command=lambda: self.set_speed(0.5), width=4).pack(side=tk.LEFT, padx=2)
        ModernButton(buttons_frame, text="1×", command=lambda: self.set_speed(1.0), width=4).pack(side=tk.LEFT, padx=2)
        ModernButton(buttons_frame, text="2×", command=lambda: self.set_speed(2.0), width=4).pack(side=tk.LEFT, padx=2)
        
        ModernButton(section, text="Apply Speed Change", command=self.apply_speed_change).pack(fill='x', pady=2)

    def create_analysis_section(self, parent):
        section = ttk.Frame(parent)
        section.pack(fill='x', pady=5)
        
        ttk.Label(section, text="Analysis", style='Modern.TLabel', font=('Segoe UI', 10, 'bold')).pack(pady=5)
        ModernButton(section, text="Detect Gender", command=self.detect_gender).pack(fill='x', pady=2)
        ModernButton(section, text="Show Features", command=self.show_features).pack(fill='x', pady=2)

    def create_visualization_section(self, parent):
        section = ttk.Frame(parent)
        section.pack(fill='x', pady=5)
        
        ttk.Label(section, text="Visualization", style='Modern.TLabel', font=('Segoe UI', 10, 'bold')).pack(pady=5)
        ModernButton(section, text="Plot Waveform", command=lambda: self.plot_analysis('waveform')).pack(fill='x', pady=2)
        ModernButton(section, text="Plot Spectrogram", command=lambda: self.plot_analysis('spectrogram')).pack(fill='x', pady=2)
        ModernButton(section, text="Plot Frequency Spectrum", command=lambda: self.plot_analysis('frequency')).pack(fill='x', pady=2)

    def create_playback_section(self, parent):
        section = ttk.Frame(parent)
        section.pack(fill='x', pady=5)
        
        ttk.Label(section, text="Playback", style='Modern.TLabel', font=('Segoe UI', 10, 'bold')).pack(pady=5)
        ModernButton(section, text="Play Original", command=lambda: self.play_audio('original')).pack(fill='x', pady=2)
        ModernButton(section, text="Play Processed", command=lambda: self.play_audio('processed')).pack(fill='x', pady=2)

    def create_display_area(self):
        self.display_frame = ttk.LabelFrame(self.right_panel, text="Display", style='Modern.TLabelframe', padding="10")
        self.display_frame.pack(fill="both", expand=True, padx=5)
        
        # Create figure with subplots
        self.fig = plt.Figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create multiple subplots
        self.axes = {
            'main': self.fig.add_subplot(211),
            'secondary': self.fig.add_subplot(212)
        }
        
        # Configure the live plot axis
        self.axes['main'].set_title('Live Audio Waveform')
        self.axes['main'].set_xlabel('Time (s)')
        self.axes['main'].set_ylabel('Amplitude')
        self.axes['main'].grid(True)
        
        self.fig.tight_layout(pad=3.0)

    def update_live_plot(self, frame):
        if not self.live_plot_active:
            return [self.plot_line] if self.plot_line else []
            
        # Update the plot with new data
        if len(self.audio_data) > 0:
            time = np.linspace(0, len(self.audio_data)/22050, len(self.audio_data))
            
            # Clear previous line
            self.axes['main'].clear()
            
            # Plot new data
            self.plot_line, = self.axes['main'].plot(time, self.audio_data)
            self.axes['main'].set_title('Live Audio Waveform')
            self.axes['main'].set_xlabel('Time (s)')
            self.axes['main'].set_ylabel('Amplitude')
            self.axes['main'].grid(True)
            
            # Adjust x-axis limits to show the most recent data
            if len(time) > 0:
                self.axes['main'].set_xlim(max(0, time[-1] - 1), time[-1])
            
            # Update layout
            self.fig.tight_layout()
            
        return [self.plot_line] if self.plot_line else []

    def plot_freq_amp_analysis(self):
        if self.audio_data is None:
            self.status_var.set("No audio data! Please record first.")
            return
            
        # Clear both plots
        for ax in self.axes.values():
            ax.clear()
            
        # Plot waveform in top subplot
        time = np.linspace(0, len(self.audio_data)/22050, len(self.audio_data))
        self.axes['main'].plot(time, self.audio_data)
        self.axes['main'].set_title('Waveform')
        self.axes['main'].set_xlabel('Time (s)')
        self.axes['main'].set_ylabel('Amplitude')
        
        # Calculate and plot frequency-amplitude relationship in bottom subplot
        fft_result = fft(self.audio_data)
        freqs = np.fft.fftfreq(len(self.audio_data), 1/22050)
        amplitudes = np.abs(fft_result)
        
        # Plot only positive frequencies up to 5000 Hz for better visualization
        mask = (freqs >= 0) & (freqs <= 5000)
        self.axes['secondary'].plot(freqs[mask], amplitudes[mask])
        self.axes['secondary'].set_title('Frequency-Amplitude Relationship')
        self.axes['secondary'].set_xlabel('Frequency (Hz)')
        self.axes['secondary'].set_ylabel('Amplitude')
        
        # Update the plot
        self.fig.tight_layout()
        self.canvas.draw()
        self.status_var.set("Frequency-Amplitude analysis complete")
            
    def plot_analysis(self, plot_type):
        if self.audio_data is None:
            self.status_var.set("No audio data! Please record first.")
            return
            
        # Clear both plots
        for ax in self.axes.values():
            ax.clear()
            
        if plot_type == 'waveform':
            time = np.linspace(0, len(self.audio_data)/22050, len(self.audio_data))
            self.axes['main'].plot(time, self.audio_data)
            self.axes['main'].set_title('Waveform')
            self.axes['main'].set_xlabel('Time (s)')
            self.axes['main'].set_ylabel('Amplitude')
            
        elif plot_type == 'spectrogram':
            D = va.librosa.stft(self.audio_data)
            S_db = va.librosa.amplitude_to_db(np.abs(D), ref=np.max)
            img = self.axes['main'].imshow(S_db, aspect='auto', origin='lower')
            self.axes['main'].set_title('Spectrogram')
            self.axes['main'].set_ylabel('Frequency Bin')
            self.axes['main'].set_xlabel('Time Frame')
            plt.colorbar(img, ax=self.axes['main'])
            
        elif plot_type == 'frequency':
            fft_result = fft(self.audio_data)
            freqs = np.fft.fftfreq(len(self.audio_data), 1/22050)
            self.axes['main'].plot(freqs[:len(freqs)//2], np.abs(fft_result)[:len(freqs)//2])
            self.axes['main'].set_title('Frequency Spectrum')
            self.axes['main'].set_xlabel('Frequency (Hz)')
            self.axes['main'].set_ylabel('Magnitude')
            
        # Update the plot
        self.fig.tight_layout()
        self.canvas.draw()
        self.status_var.set(f"{plot_type.capitalize()} plotted")
        
    def toggle_recording(self):
        if not self.recording:
            try:
                duration = float(self.duration_var.get())
                if duration <= 0:
                    raise ValueError("Duration must be positive")
                    
                # Disable all buttons during recording
                self.disable_buttons()
                
                self.record_button.config(text="Recording...")
                self.status_var.set("Recording in progress...")
                self.recording = True
                self.live_plot_active = True
                self.audio_data = np.array([])  # Reset audio data
                
                # Start recording in a separate thread
                self.record_thread = threading.Thread(target=self.record_audio, args=(duration,), daemon=True)
                self.record_thread.start()
                
                # Start live plotting animation
                self.ani = FuncAnimation(
                    self.fig,
                    self.update_live_plot,
                    interval=50,
                    blit=True,
                    cache_frame_data=False,
                    save_count=None
                )
                self.canvas.draw()
                
                # Start checking the recording status
                self.root.after(100, self.check_recording_status)
                
            except ValueError as e:
                self.status_var.set(f"Error: {str(e)}")
                self.recording = False
                self.live_plot_active = False
                self.record_button.config(text="Record")
                self.enable_buttons()
        else:
            self.recording = False
            self.live_plot_active = False
            if self.ani is not None:
                self.ani.event_source.stop()
                self.ani = None
            self.record_button.config(text="Record")
            self.status_var.set("Recording cancelled")
            self.enable_buttons()

    def record_audio(self, duration):
        try:
            chunk_size = 1024
            total_samples = int(duration * 22050)
            chunks_per_update = 4  # Number of chunks to collect before updating plot
            
            for i in range(0, total_samples, chunk_size * chunks_per_update):
                if not self.recording:
                    break
                    
                # Record multiple chunks at once
                chunks = []
                for _ in range(chunks_per_update):
                    if i + chunk_size > total_samples:
                        break
                    chunk = va.record_audio_chunk(chunk_size)
                    if chunk is not None:
                        chunks.append(chunk)
                
                if chunks:
                    new_data = np.concatenate(chunks)
                    self.audio_data = np.append(self.audio_data, new_data)
                
            if len(self.audio_data) > 0:
                self.queue.put(("Recording complete!", None))
            else:
                self.queue.put(("Error during recording!", "No audio data captured"))
        except Exception as e:
            self.queue.put(("Error during recording!", str(e)))

    def check_recording_status(self):
        try:
            if self.recording and self.record_thread.is_alive():
                # Continue checking while recording is in progress
                self.root.after(100, self.check_recording_status)
            else:
                self.check_queue()
        except Exception as e:
            self.status_var.set(f"Error checking recording status: {str(e)}")
            self.recording = False
            self.record_button.config(text="Record")
            self.enable_buttons()

    def check_queue(self):
        try:
            message, error = self.queue.get_nowait()
            if error:
                self.status_var.set(f"Error: {error}")
            else:
                self.status_var.set(message)
                # Plot the recorded waveform automatically
                self.plot_analysis('waveform')
                
            self.recording = False
            self.record_button.config(text="Record")
            self.enable_buttons()
            
        except queue.Empty:
            if self.recording:
                # Keep checking if still recording
                self.root.after(100, self.check_queue)

    def disable_buttons(self):
        """Disable all buttons except Record during recording"""
        for widget in self.main_frame.winfo_children():
            if isinstance(widget, ttk.LabelFrame):  # Control frame
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Button) and child != self.record_button:
                        child.config(state='disabled')
                    if isinstance(child, ttk.Entry):
                        child.config(state='disabled')

    def enable_buttons(self):
        """Re-enable all buttons after recording"""
        for widget in self.main_frame.winfo_children():
            if isinstance(widget, ttk.LabelFrame):  # Control frame
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Button):
                        child.config(state='normal')
                    if isinstance(child, ttk.Entry):
                        child.config(state='normal')

    def process_audio(self, process_type):
        if self.audio_data is None:
            self.status_var.set("No audio data! Please record or upload first.")
            return
            
        if process_type == 'fft':
            processed = va.apply_fft(self.audio_data)
            self.processed_audio = np.real(np.fft.ifft(processed))
        elif process_type == 'dft':
            processed = va.apply_dft(self.audio_data)
            self.processed_audio = va.apply_idft(processed)
        elif process_type == 'pitch':
            try:
                steps = float(self.pitch_var.get())
                self.processed_audio = va.pitch_shift(self.audio_data, n_steps=steps)
            except ValueError:
                self.status_var.set("Invalid pitch shift value!")
                return
                
        va.play_audio(self.processed_audio)
        self.status_var.set(f"{process_type.upper()} processing complete")
        
    def detect_gender(self):
        if self.audio_data is None:
            self.status_var.set("No audio data! Please record first.")
            return
            
        gender, frequency = va.detect_gender(self.audio_data)
        self.status_var.set(f"Detected gender: {gender} (Fundamental frequency: {frequency:.2f} Hz)")

    def apply_noise_reduction(self):
        if self.audio_data is None:
            self.status_var.set("No audio data! Please record or upload first.")
            return
            
        try:
            strength = float(self.noise_strength_var.get())
            threshold = float(self.noise_threshold_var.get())
            
            # Apply noise reduction
            self.processed_audio = va.apply_noise_reduction(
                self.audio_data,
                noise_reduce_strength=strength,
                noise_threshold=threshold
            )
            
            # Plot original vs cleaned audio
            for ax in self.axes.values():
                ax.clear()
                
            # Plot original audio in top subplot
            time = np.linspace(0, len(self.audio_data)/22050, len(self.audio_data))
            self.axes['main'].plot(time, self.audio_data, label='Original')
            self.axes['main'].set_title('Original Audio Waveform')
            self.axes['main'].set_xlabel('Time (s)')
            self.axes['main'].set_ylabel('Amplitude')
            self.axes['main'].legend()
            
            # Plot cleaned audio in bottom subplot
            time = np.linspace(0, len(self.processed_audio)/22050, len(self.processed_audio))
            self.axes['secondary'].plot(time, self.processed_audio, label='Noise Reduced', color='green')
            self.axes['secondary'].set_title('Noise Reduced Audio Waveform')
            self.axes['secondary'].set_xlabel('Time (s)')
            self.axes['secondary'].set_ylabel('Amplitude')
            self.axes['secondary'].legend()
            
            # Update plot
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Play the cleaned audio
            va.play_audio(self.processed_audio)
            
            self.status_var.set("Noise reduction applied successfully")
            
        except ValueError:
            self.status_var.set("Invalid noise reduction parameters!")

    def upload_audio(self):
        """Handle audio file upload"""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.ogg *.flac"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.audio_data = va.load_audio_file(file_path)
                self.status_var.set(f"Loaded audio file: {os.path.basename(file_path)}")
                self.plot_analysis('waveform')
            except Exception as e:
                self.status_var.set(f"Error loading audio file: {str(e)}")

    def save_audio(self):
        """Save processed audio to file"""
        if self.processed_audio is None:
            self.status_var.set("No processed audio to save!")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Audio File",
            defaultextension=".wav",
            filetypes=[
                ("WAV Audio", "*.wav"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            try:
                va.save_audio_file(self.processed_audio, file_path)
                self.status_var.set(f"Saved audio to: {os.path.basename(file_path)}")
            except Exception as e:
                self.status_var.set(f"Error saving audio file: {str(e)}")

    def play_audio(self, audio_type='original'):
        """Play either original or processed audio"""
        if audio_type == 'original' and self.audio_data is not None:
            va.play_audio(self.audio_data)
        elif audio_type == 'processed' and self.processed_audio is not None:
            va.play_audio(self.processed_audio)
        else:
            self.status_var.set(f"No {audio_type} audio available!")

    def apply_volume(self):
        if self.audio_data is None:
            self.status_var.set("No audio data! Please record or upload first.")
            return
        try:
            gain = float(self.volume_var.get())
            self.processed_audio = va.apply_volume_adjustment(self.audio_data, gain)
            self.plot_comparison("Volume Adjusted")
            va.play_audio(self.processed_audio)
            self.status_var.set("Volume adjustment complete")
        except ValueError:
            self.status_var.set("Invalid volume gain value!")

    def apply_fade(self):
        if self.audio_data is None:
            self.status_var.set("No audio data! Please record or upload first.")
            return
        fade_type = self.fade_var.get()
        self.processed_audio = va.apply_fade(self.audio_data, fade_type=fade_type)
        self.plot_comparison("Fade Applied")
        va.play_audio(self.processed_audio)
        self.status_var.set("Fade effect complete")

    def apply_bandpass(self):
        if self.audio_data is None:
            self.status_var.set("No audio data! Please record or upload first.")
            return
        try:
            lowcut = float(self.lowcut_var.get())
            highcut = float(self.highcut_var.get())
            self.processed_audio = va.apply_bandpass_filter(self.audio_data, lowcut=lowcut, highcut=highcut)
            self.plot_comparison("Bandpass Filtered")
            va.play_audio(self.processed_audio)
            self.status_var.set("Bandpass filter complete")
        except ValueError:
            self.status_var.set("Invalid frequency values!")

    def show_features(self):
        if self.audio_data is None:
            self.status_var.set("No audio data! Please record or upload first.")
            return
        features = va.get_audio_features(self.audio_data)
        feature_window = tk.Toplevel(self.root)
        feature_window.title("Audio Features")
        feature_window.geometry("300x200")
        
        for i, (feature, value) in enumerate(features.items()):
            ttk.Label(feature_window, text=f"{feature}: {value:.4f}").grid(row=i, column=0, padx=10, pady=5)

    def plot_comparison(self, title):
        """Plot original vs processed audio comparison"""
        for ax in self.axes.values():
            ax.clear()
            
        # Plot original audio in top subplot
        time = np.linspace(0, len(self.audio_data)/22050, len(self.audio_data))
        self.axes['main'].plot(time, self.audio_data, label='Original')
        self.axes['main'].set_title('Original Audio Waveform')
        self.axes['main'].set_xlabel('Time (s)')
        self.axes['main'].set_ylabel('Amplitude')
        self.axes['main'].legend()
        
        # Plot processed audio in bottom subplot
        time = np.linspace(0, len(self.processed_audio)/22050, len(self.processed_audio))
        self.axes['secondary'].plot(time, self.processed_audio, label=title, color='green')
        self.axes['secondary'].set_title(f'{title} Audio Waveform')
        self.axes['secondary'].set_xlabel('Time (s)')
        self.axes['secondary'].set_ylabel('Amplitude')
        self.axes['secondary'].legend()
        
        # Update plot
        self.fig.tight_layout()
        self.canvas.draw()

    def set_speed(self, speed):
        """Set the speed factor value."""
        self.speed_factor.set(str(speed))

    def apply_speed_change(self):
        """Apply speed change to the audio."""
        if self.audio_data is None or len(self.audio_data) == 0:
            self.status_var.set("No audio data to modify!")
            return
            
        try:
            speed = float(self.speed_factor.get())
            if speed <= 0:
                raise ValueError("Speed factor must be positive")
                
            self.status_var.set("Applying speed change...")
            self.processed_audio = va.modify_speed(self.audio_data, speed_factor=speed)
            
            # Plot the modified audio
            self.plot_comparison("Speed Modified")
            
            # Update status
            self.status_var.set(f"Speed changed to {speed}×")
            
        except ValueError as e:
            self.status_var.set(f"Error: {str(e)}")
        except Exception as e:
            self.status_var.set(f"Error modifying speed: {str(e)}")

def main():
    root = tk.Tk()
    app = VoiceAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 