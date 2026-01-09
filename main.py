from fastapi import FastAPI, APIRouter, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import redis
import uuid
import json
import requests
import logging
import asyncio
from datetime import datetime
from collections import defaultdict
import time

##################### LOGGING ####################

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

##################### CONFIGURATION ####################

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = None
REDIS_DB = 0

QWEN_HOST = "localhost"
QWEN_PORT = 8001
QWEN_MODEL = "qwen-7b-instruct"
QWEN_TIMEOUT = 60

SESSION_PREFIX = "session:"
SUMMARY_PREFIX = "summary:"
SESSION_TTL = 86400 * 7  # 7 days

# 🔥 КРИТИЧЕСКИЕ ОГРАНИЧЕНИЯ ДЛЯ СТАБИЛЬНОСТИ
MAX_HISTORY_LENGTH = 6  # Только последние 6 сообщений
MAX_RESPONSE_TOKENS = 300  # Короткие ответы = быстрее
MAX_SUMMARY_TOKENS = 200  # Компактные summaries
SUMMARY_THRESHOLD = 8  # Обновлять summary после 8 сообщений

# 🔒 ЗАЩИТА ОТ ПЕРЕГРУЗКИ
LLM_CONCURRENCY = 2  # Максимум 2 одновременных запроса к LLM
RATE_LIMIT_REQUESTS = 1  # 1 запрос
RATE_LIMIT_WINDOW = 3  # за 3 секунды на session

##################### REDIS CONNECTION ####################

r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    db=REDIS_DB,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_keepalive=True
)

##################### CONCURRENCY CONTROL ####################

# Семафор для ограничения одновременных запросов к LLM
llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

# Rate limiting по session_id
rate_limit_tracker: Dict[str, List[float]] = defaultdict(list)

def check_rate_limit(session_id: str) -> bool:
    """Проверка rate limit для session"""
    now = time.time()
    
    # Удаляем старые запросы
    rate_limit_tracker[session_id] = [
        req_time for req_time in rate_limit_tracker[session_id]
        if now - req_time < RATE_LIMIT_WINDOW
    ]
    
    # Проверяем лимит
    if len(rate_limit_tracker[session_id]) >= RATE_LIMIT_REQUESTS:
        return False
    
    # Добавляем текущий запрос
    rate_limit_tracker[session_id].append(now)
    return True

##################### PYDANTIC MODELS ####################

class Message(BaseModel):
    role: str
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    timestamp: str
    summary: Optional[str] = None
    tokens_used: Optional[int] = None

class SummaryUpdate(BaseModel):
    new_summary: str

class SessionCreate(BaseModel):
    metadata: Optional[Dict[str, Any]] = None

##################### UTILITIES ####################

def generate_id() -> str:
    return str(uuid.uuid4())

##################### SESSION FUNCTIONS ####################

def create_session(metadata: Optional[Dict] = None) -> str:
    session_id = generate_id()
    key = f"{SESSION_PREFIX}{session_id}"
    
    r.hset(key, mapping={
        "messages": json.dumps([]),
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "metadata": json.dumps(metadata or {}),
        "message_count": "0"
    })
    r.expire(key, SESSION_TTL)
    
    logger.info(f"✓ Session created: {session_id}")
    return session_id

def get_session(session_id: str) -> Optional[Dict]:
    key = f"{SESSION_PREFIX}{session_id}"
    data = r.hgetall(key)
    
    if not data:
        return None
    
    return {
        "session_id": session_id,
        "messages": json.loads(data.get("messages", "[]")),
        "created_at": data.get("created_at", ""),
        "updated_at": data.get("updated_at", ""),
        "metadata": json.loads(data.get("metadata", "{}")),
        "message_count": int(data.get("message_count", 0))
    }

def update_session(session_id: str, messages: List[Dict]) -> bool:
    key = f"{SESSION_PREFIX}{session_id}"
    
    if not r.exists(key):
        return False
    
    # Храним только последние MAX_HISTORY_LENGTH сообщений
    if len(messages) > MAX_HISTORY_LENGTH:
        messages = messages[-MAX_HISTORY_LENGTH:]
    
    r.hset(key, mapping={
        "messages": json.dumps(messages),
        "updated_at": datetime.utcnow().isoformat(),
        "message_count": str(len(messages))
    })
    r.expire(key, SESSION_TTL)
    
    return True

def delete_session(session_id: str) -> bool:
    key = f"{SESSION_PREFIX}{session_id}"
    summary_key = f"{SUMMARY_PREFIX}{session_id}"
    
    deleted = r.delete(key, summary_key)
    
    # Очистка rate limit
    if session_id in rate_limit_tracker:
        del rate_limit_tracker[session_id]
    
    logger.info(f"✓ Session deleted: {session_id}")
    return deleted > 0

def list_sessions(limit: int = 100) -> List[str]:
    pattern = f"{SESSION_PREFIX}*"
    keys = r.keys(pattern)
    session_ids = [k.replace(SESSION_PREFIX, "") for k in keys]
    return session_ids[:limit]

##################### SUMMARY FUNCTIONS ####################

def get_summary(session_id: str) -> str:
    key = f"{SUMMARY_PREFIX}{session_id}"
    data = r.get(key)
    return data if data else ""

def update_summary(session_id: str, summary: str) -> None:
    key = f"{SUMMARY_PREFIX}{session_id}"
    r.set(key, summary, ex=SESSION_TTL)
    logger.info(f"✓ Summary updated: {session_id}")

def delete_summary(session_id: str) -> bool:
    key = f"{SUMMARY_PREFIX}{session_id}"
    return r.delete(key) > 0

##################### LLM CLIENT ####################

BASE_URL = f"http://{QWEN_HOST}:{QWEN_PORT}/v1/chat/completions"

async def send_to_llm(messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = MAX_RESPONSE_TOKENS) -> Dict[str, Any]:
    """Асинхронная отправка запроса к LLM с ограничением concurrency"""
    
    async with llm_semaphore:
        logger.info(f"🔄 LLM request started (semaphore: {llm_semaphore._value}/{LLM_CONCURRENCY})")
        
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
            
            # Подсчёт токенов (примерно)
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

async def generate_summary(old_summary: str, messages: List[Dict]) -> str:
    """Генерация краткого summary"""
    
    # Берём только последние 4 сообщения для summary
    recent_messages = "\n".join([
        f"{m['role'].capitalize()}: {m['content'][:100]}"
        for m in messages[-4:]
    ])
    
    prompt = f"""Create a brief summary (max 150 words) of this conversation.

Previous summary: {old_summary if old_summary else "None"}

Recent messages:
{recent_messages}

Focus on: key topics, important facts, user preferences.
Be concise and factual."""

    llm_messages = [{"role": "user", "content": prompt}]
    
    try:
        result = await send_to_llm(llm_messages, temperature=0.3, max_tokens=MAX_SUMMARY_TOKENS)
        return result["content"].strip()
    except Exception as e:
        logger.error(f"✗ Summary generation failed: {e}")
        return old_summary

##################### CHAT PROCESSING ####################

def build_compact_context(summary: str, messages: List[Dict]) -> str:
    """Компактный контекст: summary + последние 4 сообщения"""
    
    context_parts = []
    
    if summary:
        context_parts.append(f"Previous context: {summary[:300]}")
    
    if messages:
        # Только последние 4 сообщения
        recent = messages[-4:] if len(messages) > 4 else messages
        history = "\n".join([
            f"{m['role']}: {m['content'][:200]}"
            for m in recent
        ])
        context_parts.append(f"Recent:\n{history}")
    
    return "\n\n".join(context_parts)

async def process_chat_message(session_id: str, user_message: str) -> ChatResponse:
    """Обработка сообщения с защитой от перегрузки"""
    
    # Rate limit проверка
    if not check_rate_limit(session_id):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit: max {RATE_LIMIT_REQUESTS} request per {RATE_LIMIT_WINDOW}s"
        )
    
    # Получаем сессию
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = session.get("messages", [])
    old_summary = get_summary(session_id)
    
    # Компактный контекст
    context = build_compact_context(old_summary, messages)
    
    # LLM запрос
    llm_messages = []
    
    if context:
        llm_messages.append({
            "role": "system",
            "content": f"You are a helpful AI assistant.\n\nContext:\n{context}"
        })
    
    llm_messages.append({
        "role": "user",
        "content": user_message
    })
    
    # Получаем ответ
    result = await send_to_llm(llm_messages)
    ai_response = result["content"]
    
    # Сохраняем сообщения
    timestamp = datetime.utcnow().isoformat()
    messages.append({
        "role": "user",
        "content": user_message,
        "timestamp": timestamp
    })
    messages.append({
        "role": "assistant",
        "content": ai_response,
        "timestamp": timestamp
    })
    
    # Обновляем сессию
    update_session(session_id, messages)
    
    # Обновляем summary если нужно
    new_summary = None
    if len(messages) >= SUMMARY_THRESHOLD:
        logger.info(f"📝 Generating summary for {session_id}")
        new_summary = await generate_summary(old_summary, messages)
        update_summary(session_id, new_summary)
    
    return ChatResponse(
        session_id=session_id,
        response=ai_response,
        timestamp=timestamp,
        summary=new_summary,
        tokens_used=result.get("tokens_used")
    )

##################### FASTAPI APP ####################

app = FastAPI(
    title="AI Chat Server - Production",
    description="Optimized AI chat server with rate limiting and concurrency control",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
chat_router = APIRouter(prefix="/chat", tags=["chat"])
session_router = APIRouter(prefix="/session", tags=["session"])
summary_router = APIRouter(prefix="/summary", tags=["summary"])

##################### HEALTH CHECK ####################

@app.get("/health")
async def health_check():
    """Health check с детальной информацией"""
    try:
        r.ping()
        redis_status = "ok"
    except Exception as e:
        redis_status = f"error: {str(e)}"
    
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: requests.get(f"http://{QWEN_HOST}:{QWEN_PORT}/health", timeout=2)
        )
        llm_status = "ok" if response.status_code == 200 else "error"
    except:
        llm_status = "unreachable"
    
    return {
        "status": "ok" if redis_status == "ok" and llm_status == "ok" else "degraded",
        "redis": redis_status,
        "llm": llm_status,
        "concurrency": {
            "current": LLM_CONCURRENCY - llm_semaphore._value,
            "max": LLM_CONCURRENCY,
            "available": llm_semaphore._value
        },
        "config": {
            "max_tokens": MAX_RESPONSE_TOKENS,
            "max_history": MAX_HISTORY_LENGTH,
            "rate_limit": f"{RATE_LIMIT_REQUESTS}/{RATE_LIMIT_WINDOW}s"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

##################### CHAT ENDPOINTS ####################

@chat_router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send message and get AI response (with rate limiting)"""
    try:
        return await process_chat_message(req.session_id, req.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

##################### SESSION ENDPOINTS ####################

@session_router.post("/create")
def create_new_session(req: SessionCreate = SessionCreate()):
    """Create new chat session"""
    session_id = create_session(req.metadata)
    return {
        "session_id": session_id,
        "created_at": datetime.utcnow().isoformat()
    }

@session_router.get("/{session_id}")
def get_session_info(session_id: str):
    """Get session info"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@session_router.delete("/{session_id}")
def remove_session(session_id: str):
    """Delete session"""
    if not delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted", "session_id": session_id}

@session_router.get("/")
def get_all_sessions(limit: int = 100):
    """List all sessions"""
    sessions = list_sessions(limit)
    return {"sessions": sessions, "count": len(sessions)}

##################### SUMMARY ENDPOINTS ####################

@summary_router.get("/{session_id}")
def read_summary(session_id: str):
    """Get session summary"""
    summary = get_summary(session_id)
    return {
        "session_id": session_id,
        "summary": summary if summary else "No summary available"
    }

@summary_router.post("/{session_id}")
def edit_summary(session_id: str, req: SummaryUpdate):
    """Update session summary manually"""
    if not get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    update_summary(session_id, req.new_summary)
    return {
        "session_id": session_id,
        "summary": req.new_summary,
        "updated_at": datetime.utcnow().isoformat()
    }

@summary_router.delete("/{session_id}")
def remove_summary(session_id: str):
    """Delete session summary"""
    if not delete_summary(session_id):
        raise HTTPException(status_code=404, detail="Summary not found")
    return {"message": "Summary deleted", "session_id": session_id}

@summary_router.post("/{session_id}/regenerate")
async def regenerate_summary(session_id: str):
    """Regenerate summary from messages"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = session.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="No messages to summarize")
    
    old_summary = get_summary(session_id)
    new_summary = await generate_summary(old_summary, messages)
    update_summary(session_id, new_summary)
    
    return {
        "session_id": session_id,
        "summary": new_summary,
        "updated_at": datetime.utcnow().isoformat()
    }

##################### STATS ENDPOINT ####################

@app.get("/stats")
def get_stats():
    """Server statistics"""
    sessions = list_sessions(1000)
    
    return {
        "total_sessions": len(sessions),
        "active_rate_limits": len(rate_limit_tracker),
        "llm_queue": LLM_CONCURRENCY - llm_semaphore._value,
        "config": {
            "max_tokens": MAX_RESPONSE_TOKENS,
            "max_history": MAX_HISTORY_LENGTH,
            "concurrency": LLM_CONCURRENCY,
            "rate_limit": f"{RATE_LIMIT_REQUESTS}/{RATE_LIMIT_WINDOW}s"
        }
    }

##################### INCLUDE ROUTERS ####################

app.include_router(chat_router)
app.include_router(session_router)
app.include_router(summary_router)

##################### STARTUP/SHUTDOWN ####################

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 50)
    logger.info("🚀 AI CHAT SERVER STARTING")
    logger.info("=" * 50)
    
    try:
        r.ping()
        logger.info("✓ Redis: CONNECTED")
    except Exception as e:
        logger.error(f"✗ Redis: FAILED - {e}")
    
    logger.info(f"⚙️  Config:")
    logger.info(f"   • Max tokens: {MAX_RESPONSE_TOKENS}")
    logger.info(f"   • Max history: {MAX_HISTORY_LENGTH}")
    logger.info(f"   • Concurrency: {LLM_CONCURRENCY}")
    logger.info(f"   • Rate limit: {RATE_LIMIT_REQUESTS}/{RATE_LIMIT_WINDOW}s")
    logger.info("=" * 50)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("👋 Shutting down server...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # ВАЖНО: только 1 worker для LLM
        log_level="info"
    )