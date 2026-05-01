"""
LLM Client: Abstraction layer for LLM calls (Gemini or Groq).
Prioritizes Groq if GROQ_API_KEY is available.
"""
import json
import os
import time
from typing import Type, TypeVar, Optional, Any

from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
API_KEY = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")

# Configuration
GROQ_MODEL = "llama-3.3-70b-versatile"  # Latest available 70B model
GEMINI_MODEL = "gemini-2.0-flash"

_groq_client = None
_gemini_initialized = False

def _get_groq_client():
    global _groq_client
    if _groq_client is None and GROQ_KEY:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_KEY)
    return _groq_client

def _ensure_gemini():
    global _gemini_initialized
    if not _gemini_initialized and API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=API_KEY)
        _gemini_initialized = True
    return _gemini_initialized


T = TypeVar("T", bound=BaseModel)

def generate_structured(prompt: str, schema: Type[T], retries: int = 2) -> Optional[T]:
    """
    Generate structured output. Prioritizes Groq (JSON mode) over Gemini.
    """
    # 1. Try Groq
    client = _get_groq_client()
    if client:
        # Append schema instructions for JSON mode
        full_prompt = f"{prompt}\n\nIMPORTANT: Return ONLY a JSON object matching this schema: {schema.model_json_schema()}"
        for attempt in range(retries + 1):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": full_prompt}],
                    model=GROQ_MODEL,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                content = chat_completion.choices[0].message.content
                return schema.model_validate_json(content)
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    time.sleep(2 ** (attempt + 2))
                    continue
                print(f"[Groq] Structured error: {e}")
                break

    # 2. Try Gemini Fallback
    if _ensure_gemini():
        import google.generativeai as genai
        model = genai.GenerativeModel(GEMINI_MODEL)
        for attempt in range(retries + 1):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.0,
                    ),
                )
                return schema.model_validate_json(response.text)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    time.sleep(2 ** (attempt + 2))
                    continue
                print(f"[Gemini] Structured error: {e}")
                break

    return None


def generate_text(prompt: str, retries: int = 2) -> Optional[str]:
    """
    Generate free text. Prioritizes Groq over Gemini.
    """
    # 1. Try Groq
    client = _get_groq_client()
    if client:
        for attempt in range(retries + 1):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=GROQ_MODEL,
                    temperature=0.15,
                    max_tokens=1024,
                )
                return chat_completion.choices[0].message.content.strip()
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    time.sleep(2 ** (attempt + 2))
                    continue
                print(f"[Groq] Text error: {e}")
                break

    # 2. Try Gemini Fallback
    if _ensure_gemini():
        import google.generativeai as genai
        model = genai.GenerativeModel(GEMINI_MODEL)
        for attempt in range(retries + 1):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.15,
                        max_output_tokens=1024,
                    ),
                )
                return response.text.strip()
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    time.sleep(2 ** (attempt + 2))
                    continue
                print(f"[Gemini] Text error: {e}")
                break

    return None
