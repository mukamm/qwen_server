import asyncio
import requests
import logging
from typing import Dict, Any, List
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# LLM Configuration
QWEN_HOST = "localhost"
QWEN_PORT = 8001
QWEN_MODEL = "qwen-7b-instruct"
QWEN_TIMEOUT = 60

# Token limits
MAX_RESPONSE_TOKENS = 300
MAX_SUMMARY_TOKENS = 150

# Concurrency
LLM_CONCURRENCY = 2

# Semaphore for concurrency control
llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

# Base URL
BASE_URL = f"http://{QWEN_HOST}:{QWEN_PORT}/v1/chat/completions"


async def send_to_llm(messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = MAX_RESPONSE_TOKENS) -> Dict[str, Any]:
    """
    Send request to LLM
    Returns: {"content": str, "tokens_used": int}
    """
    async with llm_semaphore:
        logger.info(f"🔄 LLM request (queue: {LLM_CONCURRENCY - llm_semaphore._value}/{LLM_CONCURRENCY})")
        
        try:
            payload = {
                "messages": messages,
                "model": QWEN_MODEL,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(BASE_URL, json=payload, timeout=QWEN_TIMEOUT)
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            tokens_used = result.get("usage", {}).get("total_tokens", len(content.split()))
            
            logger.info(f"✓ LLM response: {len(content)} chars, ~{tokens_used} tokens")
            
            return {
                "content": content,
                "tokens_used": tokens_used
            }
            
        except requests.exceptions.Timeout:
            logger.error("✗ LLM timeout")
            raise HTTPException(status_code=504, detail="LLM timeout")
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ LLM error: {e}")
            raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")
        except Exception as e:
            logger.error(f"✗ Unexpected error: {e}")
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


def get_llm_queue_status() -> Dict[str, int]:
    """Get current LLM queue status"""
    return {
        "concurrency_limit": LLM_CONCURRENCY,
        "active": LLM_CONCURRENCY - llm_semaphore._value,
        "available": llm_semaphore._value
    }