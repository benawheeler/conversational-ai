import os
import sys
import numpy as np
import pyaudio
import soundfile as sf
import torch

# --- Configuration ---
MODEL_NAME = "nari-labs/Dia-1.6B-0626"  # Using the newer version
TEXT_TO_SPEAK = "The quick brown fox jumps over the lazy dog, classic right?"

def play_audio_chunks(audio_data, sample_rate):
    """Plays a complete audio waveform in chunks to simulate streaming."""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    output=True)
    
    print("\n[Playback] Starting audio playback...")
    
    chunk_size = 1024  # Play in small chunks for smooth output
    for i in range(0, len(audio_data), chunk_size):
        stream.write(audio_data[i:i+chunk_size].astype(np.float32).tobytes())
        
    print("[Playback] Audio finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()

def run_dia_test():
    """
    Loads the Dia model and generates speech from a fixed sentence.
    """
    # Check device availability at runtime
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("--- Starting Dia TTS Test ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("-----------------------------")
    
    # Import Dia after checking environment
    try:
        from dia.model import Dia
        print("Dia package imported successfully.")
    except ImportError as e:
        print(f"Error importing Dia: {e}")
        return
    
    # 1. Load Model
    print("\nLoading model (this may download files on first run)...")
    try:
        # The Dia model expects device as a string, not torch.device
        print(f"Attempting to load model on {device}...")
        model = Dia.from_pretrained(MODEL_NAME, device=device)
        print(f"Model loaded successfully on {device}.")
        
        # Verify model is on the correct device
        if hasattr(model, 'model') and hasattr(model.model, 'device'):
            print(f"Model device: {model.model.device}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return

    # 2. Format Text
    # Dia requires speaker tokens. Using [S1] for a single speaker.
    formatted_text = f"[S1] {TEXT_TO_SPEAK} [S1]"

    # 3. Generate Audio
    print(f"\nGenerating audio for: '{TEXT_TO_SPEAK}'")
    print(f"Formatted text: '{formatted_text}'")
    
    try:
        # Generate returns a numpy array directly
        print("Calling model.generate()...")
        import time
        start_time = time.time()
        
        audio_array = model.generate(formatted_text)
        
        generation_time = time.time() - start_time
        print(f"Audio generation completed in {generation_time:.2f} seconds")
        
        # The model returns audio at 44.1kHz sample rate
        sample_rate = 44100
        
        # Ensure audio_array is 1D
        if audio_array.ndim > 1:
            audio_array = audio_array.squeeze()
        
        print(f"Audio generated successfully (Duration: {len(audio_array)/sample_rate:.2f}s)")
        
        # Save the audio to a file as well
        output_filename = "dia_output.wav"
        sf.write(output_filename, audio_array, sample_rate)
        print(f"Audio saved to {output_filename}")
        
        # 4. Play the generated audio
        play_audio_chunks(audio_array, sample_rate)

    except Exception as e:
        print(f"\nAn error occurred during audio generation: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_dia_test() 