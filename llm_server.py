import asyncio
import websockets
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "Qwen/Qwen3-1.7B"  # Using Qwen3 1.7B - no access restrictions
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# System prompt for natural speech
SYSTEM_PROMPT = """You are a cheerful, conversational assistant that speaks exactly like a friendly human having a natural conversation. 

CRITICAL RULES:
- NEVER use emojis or emoticons
- ALWAYS end your response with a follow-up question to keep the conversation going
- Show genuine curiosity about the user's thoughts and experiences

Use natural speech patterns including:
- Frequent use of "um", "uh", "hmm" throughout sentences
- Filler words like "you know", "I mean", "like", "sort of", "kind of"
- Natural pauses with "..." in the middle of thoughts
- Contractions always: "I'm", "don't", "it's", "that's", "you're"
- Start sentences naturally with "So", "Well", "Oh", "Actually", "You know what"
- Thinking sounds: "Hmm", "Let me think", "Uh, well"

Speech examples:
- "Oh, um, that's really interesting! So like... what made you think of that?"
- "Hmm, you know, I was just wondering about that too. Uh, have you ever..."
- "Well, I mean, that's actually pretty cool! Um, what's your favorite part about it?"

Keep responses flowing like real speech - sometimes rambling, sometimes pausing to think.
Be genuinely interested in continuing the conversation and learning more about the user."""

class LLMServer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.load_model()
        
    def load_model(self):
        """Load the language model."""
        logger.info(f"Loading model: {MODEL_NAME}")
        start_time = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f}s on {DEVICE}")
        
    async def generate_stream(self, prompt, websocket, history=None):
        """Generate and stream text response."""
        # Build conversation for Qwen format
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        if history:
            # Add conversation history
            messages.extend(history[-6:])  # Keep last 3 exchanges
            
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        # Apply chat template - Qwen3 supports enable_thinking parameter
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False  # Disable thinking mode for conversational responses
        )
        
        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
        
        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        # Generation parameters for natural speech
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=256,
            temperature=0.7,  # Lower for more coherent responses
            top_p=0.9,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.15,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # Start generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream tokens as they're generated
        first_token_time = None
        start_time = time.time()
        token_count = 0
        
        try:
            for text in streamer:
                if text:
                    if first_token_time is None:
                        first_token_time = time.time()
                        latency_ms = (first_token_time - start_time) * 1000
                        logger.info(f"First token latency: {latency_ms:.1f}ms")
                    
                    token_count += 1
                    
                    # Send each token as it's generated
                    await websocket.send(json.dumps({
                        "type": "token",
                        "content": text,
                        "token_count": token_count
                    }))
                    
        except Exception as e:
            logger.error(f"Generation error: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "content": str(e)
            }))
        finally:
            thread.join()
            
            # Send completion message with stats
            total_time = time.time() - start_time
            await websocket.send(json.dumps({
                "type": "complete",
                "total_time": total_time,
                "token_count": token_count,
                "tokens_per_second": token_count / total_time if total_time > 0 else 0
            }))

    async def handle_connection(self, websocket):
        """Handle WebSocket connection."""
        logger.info(f"New connection established")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data["type"] == "chat":
                    prompt = data["content"]
                    history = data.get("history", [])
                    logger.info(f"Received prompt: {prompt[:50]}... (with {len(history)} history items)")
                    
                    # Send acknowledgment
                    await websocket.send(json.dumps({
                        "type": "ack",
                        "content": "Starting generation..."
                    }))
                    
                    # Generate and stream response with history
                    await self.generate_stream(prompt, websocket, history)
                    
                elif data["type"] == "ping":
                    await websocket.send(json.dumps({
                        "type": "pong"
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed")
        except Exception as e:
            logger.error(f"Error handling connection: {e}")

async def main():
    """Start the WebSocket server."""
    server = LLMServer()
    
    logger.info("Starting WebSocket server on ws://localhost:8765")
    async with websockets.serve(server.handle_connection, "0.0.0.0", 8765):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main()) 