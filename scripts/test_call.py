"""Quick verification script for Phase 1.

Run this with:
    uv run python scripts/test_call.py

It verifies:
1. .env is correctly loaded by Pydantic
2. The Gemini API key is present
3. An actual LLM call succeeds
"""

import sys
from pathlib import Path

# Add 'src' directory to Python search path so imports work cleanly
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from core.config import get_settings
from core.llm import check_llm_connection, get_llm


def main():
    print("=" * 60)
    print("Multi-Agent Research Assistant - Phase 1 Verification")
    print("=" * 60)

    # Step 1: Check Settings
    print("\n[1/3] Loading configuration from .env...")
    settings = get_settings()
    print(f"  Target Model: {settings.gemini_model}")
    print(f"  Qdrant URL:   {settings.qdrant_url}")
    print(f"  Redis URL:    {settings.redis_url}")

    # Check API key presence
    if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
        print("\n[!] GEMINI_API_KEY is not set yet in your .env file.")
        print("    1. Go to https://aistudio.google.com/ and create a free API key.")
        print("    2. Open .env and paste: GEMINI_API_KEY=AIzaSy...")
        print("    3. Run this script again: uv run python scripts/test_call.py")
        sys.exit(1)

    masked_key = settings.gemini_api_key[:6] + "..." + settings.gemini_api_key[-4:]
    print(f"  Gemini Key:   {masked_key} (Valid format)")

    # Step 2: Initialize LLM
    print("\n[2/3] Initializing unified LLM provider...")
    llm = get_llm()
    print(f"  Provider initialized: {llm.__class__.__name__}")

    # Step 3: Test LLM Call
    print("\n[3/3] Sending test prompt to Gemini...")
    try:
        reply = check_llm_connection()
        print(f"\n[SUCCESS] Response from Gemini:\n{reply}\n")
        print("Phase 1 verified successfully!")
    except Exception as e:  # noqa: BLE001
        print(f"\n[ERROR] Connection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
