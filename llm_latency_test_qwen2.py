import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    print("Qwen2-1.5B Latency Test Console")
    print("Type your prompt and press Enter. Type 'quit' to exit.")
    
    # Load model and tokenizer
    model_name = "Qwen/Qwen2-1.5B"
    print(f"Loading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print("Model loaded. Ready for input.\n")

    while True:
        prompt = input("Prompt: ")
        if prompt.strip().lower() == 'quit':
            break
        if not prompt.strip():
            continue

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        max_new_tokens = 64
        print("Generating...")
        start_time = time.time()
        first_token_time = None
        output_ids = input_ids
        generated_tokens = []

        # Use generate with streamer to measure first token latency
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_thread = torch.jit.fork(
            model.generate,
            input_ids,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        for i, token in enumerate(streamer):
            if first_token_time is None:
                first_token_time = time.time()
                print(f"First token latency: {(first_token_time - start_time)*1000:.1f} ms")
            print(token, end='', flush=True)
            generated_tokens.append(token)
        print()
        end_time = time.time()
        print(f"Total generation time: {(end_time - start_time):.2f} s for {len(generated_tokens)} tokens\n")

if __name__ == "__main__":
    main() 