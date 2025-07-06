import time
import torch
import numpy as np
import pyaudio
from transformers import AutoModelForCausalLM, AutoTokenizer
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from pathlib import Path
import os

# --- Configuration ---
LLM_MODEL_NAME = "Qwen/Qwen3-1.7B"
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
SPEAKER_WAV_PATH = "speaker_ref.wav"  # <--- IMPORTANT: Create this file. Record a 6-10 second sample of your voice.
PROMPT = "Hello! Introduce yourself and tell me a short, interesting fact."
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def play_audio_stream(audio_stream):
    """Plays an audio stream from the TTS model using PyAudio."""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=24000,  # XTTS model output sample rate is 24kHz
                    output=True)
    
    first_chunk = True
    ttfa_time = 0
    
    for chunk in audio_stream:
        if first_chunk:
            # This is the moment the first audio chunk is ready to be played
            ttfa_time = time.perf_counter()
            first_chunk = False
        
        # Convert numpy array to bytes for PyAudio
        stream.write(chunk.tobytes())
        
    stream.stop_stream()
    stream.close()
    p.terminate()
    return ttfa_time

def run_full_pipeline_test():
    """
    Loads both LLM and TTS models, generates text, synthesizes it to speech,
    and measures the end-to-end latency.
    """
    print("--- Starting Full Pipeline Latency Test ---")
    print(f"LLM: {LLM_MODEL_NAME} | TTS: {TTS_MODEL_NAME} | Device: {DEVICE}")
    print("------------------------------------------")

    # 1. Load Models
    print("Loading LLM...")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME, torch_dtype="auto", device_map="auto"
    )

    print("Loading TTS model (this may download files on first run)...")
    tts = TTS(model_name=TTS_MODEL_NAME, gpu=(DEVICE == "cuda"))
    print("Models loaded successfully.")

    # 2. Prepare Prompt for LLM
    messages = [{"role": "user", "content": PROMPT}]
    text_prompt = llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    model_inputs = llm_tokenizer([text_prompt], return_tensors="pt").to(DEVICE)

    # 3. Generate Text from LLM (non-streaming for this test to isolate generation time)
    print(f"\nUser Prompt: {PROMPT}")
    print("Generating LLM response...")
    
    start_time = time.perf_counter()
    generated_ids = llm_model.generate(
        **model_inputs,
        max_new_tokens=60,
        do_sample=True,
        top_p=0.8,
        temperature=0.7,
    )
    llm_response_time = time.perf_counter()
    
    response_text = llm_tokenizer.decode(generated_ids[0][len(model_inputs["input_ids"][0]):], skip_special_tokens=True)
    print(f"LLM Response: {response_text}")

    # 4. Synthesize Speech with TTS (Streaming)
    print("\nSynthesizing speech and playing audio...")
    
    # --- Use the correct tts_stream() method ---
    audio_stream = tts.tts_stream(
        text=response_text,
        speaker_wav=SPEAKER_WAV_PATH,
        language="en",
    )

    # This function will start playing the audio as soon as the first chunk arrives
    first_audio_time = play_audio_stream(audio_stream)
    end_time = time.perf_counter()

    # 5. Print Latency Report
    print("\n\n--- Latency Report ---")
    llm_latency = (llm_response_time - start_time) * 1000
    ttfa_latency = (first_audio_time - start_time) * 1000
    full_latency = (end_time - start_time) * 1000
    
    print(f"LLM Text Generation Time: {llm_latency:.2f} ms")
    print(f"Time to First Audio (TTFA): {ttfa_latency:.2f} ms")
    print(f"Total Pipeline Time (Text Gen + Full Audio Playback): {full_latency:.2f} ms")
    print("----------------------")


if __name__ == "__main__":
    # Ensure you have a speaker reference file
    try:
        with open(SPEAKER_WAV_PATH, "rb") as f:
            pass
    except FileNotFoundError:
        print("\n\nERROR: Speaker reference file not found!")
        print(f"Please create a WAV file named '{SPEAKER_WAV_PATH}' with 6-10 seconds of clear speech.")
        print("You can record one yourself using a tool like Audacity or your operating system's voice recorder.")
    else:
        run_full_pipeline_test() 