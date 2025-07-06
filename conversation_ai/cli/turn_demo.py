import asyncio
from pathlib import Path

from conversation_ai.audio import AudioStream
from conversation_ai.vad import SileroVAD
from conversation_ai.vad.semantic import SemanticTurnDetector
from conversation_ai.core.turn_manager import TurnManager
from conversation_ai.asr.transcriber import WhisperTranscriber


async def main():
    semantic = SemanticTurnDetector()
    transcriber = WhisperTranscriber(model_name="base")
    turn_mgr = TurnManager(
        semantic,
        soft_timeout_sec=None,
        hard_timeout_sec=None,
        context_max_sec=6.0,
        debug_dir="context_inputs",
        transcriber=transcriber,
    )
    vad = SileroVAD(pre_roll_sec=1.0)

    async with AudioStream(block_size=512) as mic:
        print("[turn_demo] Listening… Speak and pause to create turns. Press Ctrl+C to exit.")
        try:
            async for frame in mic.frames():
                events = vad.feed(frame)

                curr_active = vad.is_active()
                if curr_active and not prev_active:
                    turn_mgr.reset_decay()
                prev_active = curr_active

                for evt in events:
                    turn_mgr.handle_segment(evt)
                turn_mgr.tick()
        except KeyboardInterrupt:
            print("[turn_demo] Stopped by user")


if __name__ == "__main__":
    asyncio.run(main()) 