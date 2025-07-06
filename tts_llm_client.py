import torch
import numpy as np
import pyaudio
import time
import threading
import queue
import asyncio
import websockets
import json
import random
import sys
import os
from pathlib import Path
from pynput import keyboard
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from enum import Enum

# --- Configuration ---
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
SPEAKER_WAV_PATH = "speaker_ref.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLM_WEBSOCKET_URL = "ws://localhost:8765"

# Optimized parameters for <200ms latency
STREAM_CHUNK_SIZE = 20
DECODER_ITERATIONS_CHUNKS = 1
ENABLE_TEXT_SPLITTING = False
SENTENCE_DELAY_MS = 500  # Shorter delay for more energetic conversation
SENTENCE_DELAY_VARIANCE = 200  # Less variance for snappier pacing
MAX_CONVERSATION_HISTORY = 10  # Keep last N exchanges

class State(Enum):
    WRITING = "writing"  # User is typing input
    LISTENING = "listening"  # LLM is generating/speaking

class LLMTTSClient:
    def __init__(self):
        self.model = None
        self.gpt_cond_latent = None
        self.speaker_embedding = None
        self.audio_queue = queue.Queue(maxsize=200)  # Larger queue to reduce blocking
        self.text_queue = queue.Queue(maxsize=50)
        self.is_generating = False
        self.stream_is_stopped = True
        self.pyaudio_stream = None
        self.stop_event = threading.Event()
        self.barge_in_event = threading.Event()  # New event for immediate interruption
        self.playback_complete = threading.Event()
        self.playback_complete.set()
        self.websocket = None
        self.tts_thread = None
        self.websocket_task = None
        
        # State management
        self.state = State.WRITING
        self.keyboard_listener = None
        
        # Conversation history
        self.conversation_history = []
        self.current_response = ""
        
        # Latency tracking
        self.query_start_time = None
        self.first_llm_token_time = None
        self.first_audio_time = None
        
        print("🚀 LLM-powered TTS with Low Latency Streaming")
        print("=" * 60)
        
        # Load TTS model
        self._load_model()
        
        # Start audio player thread
        self.audio_thread = threading.Thread(target=self._audio_player, daemon=True)
        self.audio_thread.start()
        
        # Start TTS generator thread
        self.tts_thread = threading.Thread(target=self._tts_generator, daemon=True)
        self.tts_thread.start()
        
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
            gpt_cond_len=12,  # Longer conditioning to capture more voice dynamics
            gpt_cond_chunk_len=6,  # Larger chunks for richer tone capture
            max_ref_length=15,  # Allow longer reference for better prosody
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
    
    def _start_keyboard_listener(self):
        """Start keyboard listener for 's' key during listening state."""
        if self.keyboard_listener is None:
            self.keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
            self.keyboard_listener.start()
    
    def _stop_keyboard_listener(self):
        """Stop keyboard listener."""
        if self.keyboard_listener is not None:
            self.keyboard_listener.stop()
            self.keyboard_listener = None
            
    def _on_key_press(self, key):
        """Handle global key presses for stopping - only active during LISTENING state."""
        if self.state == State.LISTENING:
            try:
                if hasattr(key, 'char') and key.char and key.char.lower() == 's':
                    # Immediate barge-in - run in separate thread to avoid blocking
                    threading.Thread(target=self.barge_in, daemon=True).start()
            except AttributeError:
                pass
            
    def _audio_player(self):
        p = pyaudio.PyAudio()
        stream = None

        def create_stream():
            nonlocal stream
            if stream:
                try: stream.abort()
                except: pass
                try: stream.close()
                except: pass
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True, frames_per_buffer=1024)
            self.pyaudio_stream = stream
            return stream

        stream = create_stream()

        while not self.stop_event.is_set():
            try:
                if self.barge_in_event.is_set():
                    time.sleep(0.01)
                    continue

                item = self.audio_queue.get(timeout=0.1)

                if item == "END_STREAM":
                    self.playback_complete.set()
                elif isinstance(item, tuple):
                    if self.barge_in_event.is_set(): continue
                    chunk, _ = item
                    audio_data = (chunk * 32767).astype(np.int16)
                    stream.write(audio_data.tobytes())
            except queue.Empty:
                continue
            except (IOError, AttributeError) as e:
                if self.barge_in_event.is_set(): continue
                print(f"Audio stream error: {e}. Recreating stream...")
                try:
                    stream = create_stream()
                except Exception as e_rec:
                    print(f"Fatal: Could not recreate audio stream: {e_rec}")
                    break
        
        if stream:
            try: stream.abort()
            except: pass
        p.terminate()

    def barge_in(self):
        if self.barge_in_event.is_set(): return
        print("\n[BARGE-IN! Stopping everything...]")
        self.barge_in_event.set()
        
        if self.websocket_task:
            self.websocket_task.cancel()
        
        if self.pyaudio_stream:
            try: self.pyaudio_stream.abort()
            except: pass
        
        self.audio_queue = queue.Queue(maxsize=200)
        self.text_queue = queue.Queue(maxsize=50)

        self.playback_complete.set()

        def clear_barge_in_event():
            time.sleep(0.2)
            self.barge_in_event.clear()
        threading.Thread(target=clear_barge_in_event, daemon=True).start()

        if sys.platform == 'win32':
            import msvcrt
            time.sleep(0.05)
            while msvcrt.kbhit(): msvcrt.getch()

    def _tts_generator(self):
        while not self.stop_event.is_set():
            if self.barge_in_event.is_set():
                time.sleep(0.01)
                continue
            try:
                text = self.text_queue.get(timeout=0.1)
                if self.barge_in_event.is_set(): continue
                
                print(f"Speaking: {text}")
                
                for chunk in self.model.inference_stream(
                    text, "en", self.gpt_cond_latent, self.speaker_embedding,
                    stream_chunk_size=STREAM_CHUNK_SIZE,
                    enable_text_splitting=ENABLE_TEXT_SPLITTING,
                    decoder_iterations_chunks=DECODER_ITERATIONS_CHUNKS,
                    temperature=0.75,
                    length_penalty=1.0,
                    repetition_penalty=2.5,
                    top_p=0.85,
                    top_k=50,
                    speed=1.15,
                ):
                    if self.barge_in_event.is_set(): break
                    chunk_cpu = chunk.cpu().numpy()
                    try:
                        self.audio_queue.put_nowait((chunk_cpu, time.time()))
                    except queue.Full:
                        pass
            except queue.Empty:
                continue

    async def process_llm_stream(self, prompt):
        """Connect to LLM server and process streaming response."""
        text_buffer = ""
        sentence_endings = ['.', '!', '?', '...']
        self.current_response = ""  # Reset current response
        
        try:
            async with websockets.connect(LLM_WEBSOCKET_URL) as websocket:
                self.websocket = websocket
                
                # Send chat request with conversation history
                await websocket.send(json.dumps({
                    "type": "chat",
                    "content": prompt,
                    "history": self.conversation_history
                }))
                
                # Process streaming response
                async for message in websocket:
                    if self.stop_event.is_set() or self.barge_in_event.is_set():
                        break
                        
                    data = json.loads(message)
                    
                    if data["type"] == "token":
                        # Track first LLM token
                        if self.first_llm_token_time is None:
                            self.first_llm_token_time = time.time()
                            latency_ms = (self.first_llm_token_time - self.query_start_time) * 1000
                            print(f"🎯 LLM first token latency: {latency_ms:.1f}ms")
                        
                        # Add token to buffer and current response
                        text_buffer += data["content"]
                        self.current_response += data["content"]
                        
                        # Check if we have a complete sentence
                        for ending in sentence_endings:
                            if ending in text_buffer:
                                # Split at sentence boundary
                                parts = text_buffer.split(ending, 1)
                                if len(parts) > 1:
                                    sentence = parts[0] + ending
                                    text_buffer = parts[1]
                                    
                                    # Queue sentence for TTS
                                    if not self.stop_event.is_set() and not self.barge_in_event.is_set():
                                        self.text_queue.put(sentence.strip())
                    
                    elif data["type"] == "complete":
                        # Process any remaining text
                        if text_buffer.strip() and not self.stop_event.is_set() and not self.barge_in_event.is_set():
                            self.text_queue.put(text_buffer.strip())
                        
                        # Print stats
                        if not self.stop_event.is_set() and not self.barge_in_event.is_set():
                            print(f"\nLLM Stats:")
                            print(f"  Total time: {data['total_time']:.2f}s")
                            print(f"  Tokens: {data['token_count']}")
                            print(f"  Tokens/sec: {data['tokens_per_second']:.1f}")
                            
                            # Add to conversation history
                            self.conversation_history.append({
                                "role": "user",
                                "content": prompt
                            })
                            self.conversation_history.append({
                                "role": "assistant", 
                                "content": self.current_response.strip()
                            })
                            
                            # Keep history size manageable
                            if len(self.conversation_history) > MAX_CONVERSATION_HISTORY * 2:
                                self.conversation_history = self.conversation_history[-(MAX_CONVERSATION_HISTORY * 2):]
                        break
                        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not self.stop_event.is_set() and not self.barge_in_event.is_set():
                print(f"WebSocket error: {e}")
        finally:
            self.websocket = None
            if not self.stop_event.is_set() and not self.barge_in_event.is_set():
                # Wait for TTS queue to empty
                timeout_start = time.time()
                while not self.text_queue.empty() and not self.stop_event.is_set() and not self.barge_in_event.is_set():
                    if time.time() - timeout_start > 10:  # 10 second timeout
                        break
                    await asyncio.sleep(0.05)
            
            # Always send END_STREAM to signal completion
            try:
                self.audio_queue.put("END_STREAM", timeout=0.5)
            except:
                pass

    async def run_interactive_mode_async(self):
        """Async interactive mode."""
        print("\n" + "=" * 60)
        print("LLM-POWERED TTS DEMO - BARGE-IN ENABLED")
        print("Features: Ultra-low latency streaming with instant 's' key interruption")
        print("States: WRITING (typing) | LISTENING (press 's' for instant stop)")
        print("Type 'quit' or press Ctrl+C to exit.")
        print("Type 'clear' to clear conversation history.")
        print("=" * 60)
        
        while True:
            # Wait for any ongoing playback to complete
            self.playback_complete.wait()
            
            # Ensure we're in writing state and keyboard listener is stopped
            self.state = State.WRITING
            self._stop_keyboard_listener()
            
            # Reset ALL state variables
            self.stop_event.clear()
            self.barge_in_event.clear()
            self.query_start_time = None
            self.first_llm_token_time = None
            self.first_audio_time = None
            self.is_generating = False
            self.current_response = ""
            self.websocket = None
            self.websocket_task = None
            
            # Ensure queues are empty
            while not self.audio_queue.empty():
                try: self.audio_queue.get_nowait()
                except: break
            while not self.text_queue.empty():
                try: self.text_queue.get_nowait()
                except: break
            
            # Small delay to ensure clean state transition
            time.sleep(0.3)  # Slightly longer delay after barge-in
            
            # Flush any remaining input
            if sys.platform == 'win32':
                import msvcrt
                while msvcrt.kbhit():
                    msvcrt.getch()

            try:
                # Get user input (keyboard listener is OFF during this)
                prompt = input("\n[WRITING] Enter your query: ")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break
            
            if prompt.lower() == 'quit':
                break
            if prompt.lower() == 'clear':
                self.conversation_history = []
                print("Conversation history cleared.")
                continue
            if not prompt.strip():
                continue
            
            # Switch to listening state and start keyboard listener
            self.state = State.LISTENING
            self._start_keyboard_listener()
            
            # Start processing
            self.query_start_time = time.time()
            print(f"\n[LISTENING] Processing query: \"{prompt}\"")
            print("(Press 's' for INSTANT stop)")
            
            self.playback_complete.clear()
            self.is_generating = True
            
            # Process LLM stream
            self.websocket_task = asyncio.create_task(self.process_llm_stream(prompt))
            
            try:
                await self.websocket_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[ERROR] WebSocket task error: {e}")
            
            self.is_generating = False
            
            # Switch back to writing state
            self.state = State.WRITING
            self._stop_keyboard_listener()
            
    def run_interactive_mode(self):
        """Run the async interactive mode."""
        asyncio.run(self.run_interactive_mode_async())
            
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up resources...")
        self.stop_event.set()
        self._stop_keyboard_listener()
        
        if self.websocket_task:
            self.websocket_task.cancel()

        if self.tts_thread and self.tts_thread.is_alive():
            self.tts_thread.join(timeout=1.0)
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)

def main():
    client = LLMTTSClient()
    try:
        client.run_interactive_mode()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        client.cleanup()
        
    print("\nApplication terminated.")

if __name__ == "__main__":
    main() 