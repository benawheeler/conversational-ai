import asyncio
from pathlib import Path

import numpy as np

from conversation_ai.audio import AudioStream
from conversation_ai.vad import SileroVAD


async def main():
    vad = SileroVAD(pre_roll_sec=1.0, output_dir=Path("segments"))

    async with AudioStream(block_size=512) as mic:
        print("[vad_demo] Listening… Press Ctrl+C to stop")
        try:
            async for frame in mic.frames():
                events = vad.feed(frame)
                for evt in events:
                    print(f"[vad_demo] Saved speech segment: {evt.wav_path} ({(evt.end_time_ms-evt.start_time_ms)/1000:.2f}s)")
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            print("[vad_demo] Stopped by user")


if __name__ == "__main__":
    asyncio.run(main()) 