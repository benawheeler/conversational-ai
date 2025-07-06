import torch
import numpy as np
import pyaudio
import time
import threading
import queue
from pathlib import Path
from pynput import keyboard
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# --- Configuration ---
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
SPEAKER_WAV_PATH = "speaker_ref.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optimized parameters for <200ms latency with better quality
STREAM_CHUNK_SIZE = 20  # Smaller chunks for lower latency
DECODER_ITERATIONS_CHUNKS = 1  # Process 1 chunk at a time
ENABLE_TEXT_SPLITTING = False  # Disable text splitting for lower latency

class InteractiveTTS:
    def __init__(self):
        self.model = None
        self.gpt_cond_latent = None
        self.speaker_embedding = None
        self.audio_queue = queue.Queue(maxsize=10)
        self.is_generating = False
        self.stream_is_stopped = True
        self.pyaudio_stream = None  # Reference to the stream for instant stop
        self.stop_event = threading.Event()
        self.playback_complete = threading.Event()
        self.playback_complete.set()  # Set initially to allow first input
        
        print("🚀 Interactive TTS with Low Latency and Hard Stop")
        print("=" * 60)
        
        # Load model
        self._load_model()
        
        # Start audio player thread
        self.audio_thread = threading.Thread(target=self._audio_player, daemon=True)
        self.audio_thread.start()

        # Start keyboard listener
        self.keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
        self.keyboard_listener.start()
        
    def _load_model(self):
        """Load XTTS model with optimizations."""
        print("Loading XTTS model...")
        start_time = time.time()
        
        try:
            model_path = Path.home() / "AppData" / "Local" / "tts" / TTS_MODEL_NAME.replace("/", "--")
            config = XttsConfig()
            config.load_json(str(model_path / "config.json"))
            
            self.model = Xtts.init_from_config(config)
            self.model.load_checkpoint(config, checkpoint_dir=str(model_path), eval=True)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure you have run 'TTS --model_name \"tts_models/multilingual/multi-dataset/xtts_v2\"' to download the model.")
            exit()
        
        if DEVICE == "cuda":
            self.model.cuda()
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
        print("Computing speaker embeddings...")
        if not Path(SPEAKER_WAV_PATH).exists():
            print(f"Error: Speaker reference file not found at '{SPEAKER_WAV_PATH}'")
            print("Please provide a speaker_ref.wav file.")
            exit()

        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
            audio_path=SPEAKER_WAV_PATH,
            gpt_cond_len=8,
            gpt_cond_chunk_len=4,
        )
        
        print("Warming up model...")
        self._warmup()
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f}s\n")
        
    def _warmup(self):
        """Warm up the model with a dummy generation."""
        try:
            list(self.model.inference_stream(
                "Hello",
                "en",
                self.gpt_cond_latent,
                self.speaker_embedding,
                stream_chunk_size=STREAM_CHUNK_SIZE,
                enable_text_splitting=False,
            ))
        except:
            pass
            
    def _on_key_press(self, key):
        """Handle global key presses for stopping."""
        try:
            if key.char.lower() == 's':
                self.stop()
        except AttributeError:
            pass  # Ignore special keys
            
    def _audio_player(self):
        """Audio player that can be hard-stopped from another thread."""
        p = pyaudio.PyAudio()
        self.pyaudio_stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True,
            frames_per_buffer=1024,
        )
        stream = self.pyaudio_stream
        self.stream_is_stopped = False
        active_chunk = None

        while True:
            try:
                item = self.audio_queue.get(timeout=0.02) # Faster polling

                if item is None:  # Cleanup signal
                    break

                if item == "INTERRUPT":
                    # Stream is already stopped by the stop() method.
                    # This is just a signal to clean up and reset.
                    active_chunk = None
                    # Drain queue
                    while not self.audio_queue.empty():
                        try: self.audio_queue.get_nowait()
                        except queue.Empty: break
                    
                    print("\nPlayback stopped.")
                    self.playback_complete.set()
                    continue

                if item == "END_STREAM":
                    if active_chunk is not None and not self.stream_is_stopped:
                        audio_data = (active_chunk * 32767).astype(np.int16)
                        stream.write(audio_data.tobytes(), exception_on_underflow=False)
                        active_chunk = None
                    self.playback_complete.set()
                    continue
                
                if self.stream_is_stopped:
                    stream.start_stream()
                    self.stream_is_stopped = False

                if active_chunk is not None:
                    audio_data = (active_chunk * 32767).astype(np.int16)
                    stream.write(audio_data.tobytes(), exception_on_underflow=False)

                active_chunk, _ = item
                
            except queue.Empty:
                if active_chunk is not None and not self.stream_is_stopped:
                    audio_data = (active_chunk * 32767).astype(np.int16)
                    stream.write(audio_data.tobytes(), exception_on_underflow=False)
                    active_chunk = None
                continue
            except IOError:
                # This is expected if the stream is stopped abruptly from another thread
                active_chunk = None
                continue # The INTERRUPT signal will be handled next
            except Exception as e:
                print(f"Audio error: {e}")
                self.playback_complete.set()
                
        if not stream.is_stopped():
            stream.stop_stream()
        stream.close()
        p.terminate()
        
    def stop(self):
        """Interrupts generation and audio playback immediately."""
        if self.is_generating:
            print("\nStopping generation and playback...")
            self.stop_event.set()
            
            # Stop the audio stream immediately from this thread
            if self.pyaudio_stream and not self.stream_is_stopped:
                try:
                    self.pyaudio_stream.stop_stream()
                    self.stream_is_stopped = True
                except Exception as e:
                    print(f"Error stopping stream: {e}")

            # Clear the queue and send the final signal for cleanup
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.audio_queue.put("INTERRUPT") # Signal audio thread to clean up

    def generate_speech(self, text):
        """Generate speech with optimized settings."""
        if self.is_generating:
            print("Already generating...")
            return
            
        self.playback_complete.clear()
        self.is_generating = True
        self.stop_event.clear()
        generation_start = time.time()
        
        try:
            while not self.audio_queue.empty():
                try: self.audio_queue.get_nowait()
                except queue.Empty: break
                    
            for chunk in self.model.inference_stream(
                text, "en", self.gpt_cond_latent, self.speaker_embedding,
                stream_chunk_size=STREAM_CHUNK_SIZE,
                enable_text_splitting=ENABLE_TEXT_SPLITTING,
                decoder_iterations_chunks=DECODER_ITERATIONS_CHUNKS,
                temperature=0.75, length_penalty=1.0,
                repetition_penalty=2.5, top_p=0.9, top_k=50,
            ):
                if self.stop_event.is_set():
                    break
                
                chunk_cpu = chunk.cpu().numpy()
                try:
                    self.audio_queue.put((chunk_cpu, time.time()), timeout=0.1)
                except queue.Full:
                    pass
                    
            if not self.stop_event.is_set():
                self.audio_queue.put("END_STREAM")
                
        except Exception as e:
            print(f"Generation error: {e}")
            self.playback_complete.set()
        finally:
            self.is_generating = False
            
    def run_interactive_mode(self):
        """Interactive test with keyboard input."""
        print("\n" + "=" * 60)
        print("INTERACTIVE TTS DEMO")
        print("Features: Low-latency streaming, immediate 'pull-the-plug' stop.")
        print("⚡ Press 's' at any time to immediately stop the current audio.")
        print("Type 'quit' or press Ctrl+C to exit.")
        print("=" * 60)
        
        while True:
            self.playback_complete.wait()

            try:
                text = input("Enter text to synthesize: ")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break
            
            if text.lower() == 'quit':
                break
            if not text.strip():
                continue
                
            print(f"Generating: \"{text}\"")
            self.generate_speech(text)
            
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up resources...")
        self.keyboard_listener.stop()
        if self.audio_thread.is_alive():
            self.audio_queue.put(None)
            self.audio_thread.join(timeout=2.0)

def main():
    tts = InteractiveTTS()
    try:
        tts.run_interactive_mode()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        tts.cleanup()
        
    print("\nApplication terminated.")

if __name__ == "__main__":
    main() 