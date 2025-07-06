import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-1.7B"
PROMPT = "Hello, this is a test, it is nice to speak with you."
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LatencyStreamer(TextStreamer):
    """
    A custom streamer to measure Time to First Token (TTFT) and
    Time to First Sentence (TTFS) during generation.
    """
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.start_time = 0
        self.first_token_time = None
        self.first_sentence_time = None
        self.full_response = ""
        self.first_token_received = False
        self.first_sentence_found = False

    def on_finalized_text(self, text: str, stream_end: bool = False):
        # This method is called by the streamer for each new piece of text.
        
        # --- Measure Time to First Token ---
        if not self.first_token_received:
            self.first_token_time = time.perf_counter()
            self.first_token_received = True

        # Append the new text to the full response
        self.full_response += text
        
        # --- Measure Time to First Sentence ---
        if not self.first_sentence_found:
            if any(punct in self.full_response for punct in ".!?"):
                self.first_sentence_time = time.perf_counter()
                self.first_sentence_found = True

        # Print the token to the console (the default behavior)
        print(text, end="", flush=True)

    def print_latency(self):
        """Prints the measured latencies after generation is complete."""
        if self.first_token_time:
            ttft = (self.first_token_time - self.start_time) * 1000
            print(f"\n\n--- Latency Report ---")
            print(f"Time to First Token (TTFT): {ttft:.2f} ms")
        
        if self.first_sentence_time:
            ttfs = (self.first_sentence_time - self.start_time) * 1000
            print(f"Time to First Sentence (TTFS): {ttfs:.2f} ms")
        
        print("----------------------")


def run_latency_test():
    """
    Loads the Qwen3 model and runs a latency test with streaming.
    """
    print(f"--- Starting LLM Latency Test ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print("---------------------------------")

    # 1. Load Model and Tokenizer
    print("Loading model and tokenizer (this may take a while on first run)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )
    print("Model loaded successfully.")

    # 2. Prepare the Prompt
    messages = [{"role": "user", "content": PROMPT}]
    text_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Use non-thinking mode for direct responses
    )
    model_inputs = tokenizer([text_prompt], return_tensors="pt").to(DEVICE)
    
    # 3. Setup Streamer and Generation
    streamer = LatencyStreamer(tokenizer, skip_prompt=True)
    
    generation_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.8,
        temperature=0.7,
    )

    print(f"\nUser Prompt: {PROMPT}")
    print("Bot Response: ", end="", flush=True)
    
    # --- Start Timer and Run Generation ---
    streamer.start_time = time.perf_counter()
    model.generate(**generation_kwargs)
    
    # 4. Print Latency Report
    streamer.print_latency()


if __name__ == "__main__":
    run_latency_test() 