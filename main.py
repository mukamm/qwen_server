from fastapi import FastAPI, APIRouter, HTTPException
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
SESSION_PREFIX = "session:"
SUMMARY_PREFIX = "summary:"
PROFILE_PREFIX = "profile:"  # 🆕 USER PROFILE
USER_PROFILE_PREFIX = "user_profile:"  # 🆕 ОБЩИЙ ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ
SESSION_TTL = 86400 * 7  # 7 days

# 🔥 КРИТИЧЕСКИЕ ОГРАНИЧЕНИЯ
MAX_HISTORY_LENGTH = 6
MAX_RESPONSE_TOKENS = 300
MAX_SUMMARY_TOKENS = 150  # Уменьшили для компактности
SUMMARY_THRESHOLD = 8

# 🔒 ЗАЩИТА ОТ ПЕРЕГРУЗКИ
LLM_CONCURRENCY = 2
RATE_LIMIT_REQUESTS = 1
RATE_LIMIT_WINDOW = 3

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

llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY)
rate_limit_tracker: Dict[str, List[float]] = defaultdict(list)

def check_rate_limit(user_id: str, session_id: str) -> bool:
    """Rate limit по user_id:session_id"""
    key = f"{user_id}:{session_id}"
    now = time.time()
    
    rate_limit_tracker[key] = [
        req_time for req_time in rate_limit_tracker[key]
        if now - req_time < RATE_LIMIT_WINDOW
    ]
    
    if len(rate_limit_tracker[key]) >= RATE_LIMIT_REQUESTS:
        return False
    
    rate_limit_tracker[key].append(now)
    return True

##################### PYDANTIC MODELS ####################

class ChatRequest(BaseModel):
    user_id: str  # 🆕 Обязательное поле
    session_id: str
    message: str

class ChatResponse(BaseModel):
    user_id: str  # 🆕
    session_id: str
    response: str
    timestamp: str
    summary: Optional[str] = None
    tokens_used: Optional[int] = None

class SessionCreate(BaseModel):
    user_id: str  # 🆕 Обязательное поле
    metadata: Optional[Dict[str, Any]] = None

class ProfileUpdate(BaseModel):
    user_id: str  # 🆕
    profile_data: Dict[str, Any]  # Без session_id - это общий профиль

class SummaryUpdate(BaseModel):
    new_summary: str

##################### UTILITIES ####################

def generate_id() -> str:
    return str(uuid.uuid4())

##################### 🆕 PROFILE FUNCTIONS (ШАГ 1) ####################

def get_user_profile(user_id: str) -> Dict[str, Any]:
    """
    🧠 ОБЩИЙ USER PROFILE - не привязан к session
    Один профиль на всего пользователя
    """
    key = f"{USER_PROFILE_PREFIX}{user_id}"
    data = r.get(key)
    
    if not data:
        # Default profile
        return {
            "name": None,
            "language": "en",
            "role": None,
            "stack": [],
            "preferences": {}
        }
    
    return json.loads(data)

def update_user_profile(user_id: str, profile_data: Dict[str, Any]) -> None:
    """Обновление общего User Profile"""
    key = f"{USER_PROFILE_PREFIX}{user_id}"
    r.set(key, json.dumps(profile_data), ex=SESSION_TTL * 4)  # Дольше живёт
    logger.info(f"✓ User profile updated: {user_id}")

def get_profile(user_id: str, session_id: str) -> Dict[str, Any]:
    """
    🧠 SESSION PROFILE - специфичный для сессии
    Но по умолчанию берём из общего профиля
    """
    # Сначала пытаемся получить session-specific profile
    key = f"{PROFILE_PREFIX}{user_id}:{session_id}"
    data = r.get(key)
    
    if data:
        return json.loads(data)
    
    # Если нет - берём общий профиль пользователя
    return get_user_profile(user_id)

def update_profile(user_id: str, session_id: str, profile_data: Dict[str, Any]) -> None:
    """Обновление Session Profile (редко)"""
    key = f"{PROFILE_PREFIX}{user_id}:{session_id}"
    r.set(key, json.dumps(profile_data), ex=SESSION_TTL)
    logger.info(f"✓ Profile updated: {user_id}:{session_id}")

def delete_profile(user_id: str, session_id: str) -> bool:
    """Удаление профиля"""
    key = f"{PROFILE_PREFIX}{user_id}:{session_id}"
    return r.delete(key) > 0

##################### SESSION FUNCTIONS ####################

def create_session(user_id: str, metadata: Optional[Dict] = None) -> str:
    session_id = generate_id()
    key = f"{SESSION_PREFIX}{user_id}:{session_id}"
    
    r.hset(key, mapping={
        "messages": json.dumps([]),
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "metadata": json.dumps(metadata or {}),
        "message_count": "0"
    })
    r.expire(key, SESSION_TTL)
    
    logger.info(f"✓ Session created: {user_id}:{session_id}")
    return session_id

def get_session(user_id: str, session_id: str) -> Optional[Dict]:
    key = f"{SESSION_PREFIX}{user_id}:{session_id}"
    data = r.hgetall(key)
    
    if not data:
        return None
    
    return {
        "user_id": user_id,
        "session_id": session_id,
        "messages": json.loads(data.get("messages", "[]")),
        "created_at": data.get("created_at", ""),
        "updated_at": data.get("updated_at", ""),
        "metadata": json.loads(data.get("metadata", "{}")),
        "message_count": int(data.get("message_count", 0))
    }

def update_session(user_id: str, session_id: str, messages: List[Dict]) -> bool:
    key = f"{SESSION_PREFIX}{user_id}:{session_id}"
    
    if not r.exists(key):
        return False
    
    # 💬 RECENT MESSAGES - только последние 6
    if len(messages) > MAX_HISTORY_LENGTH:
        messages = messages[-MAX_HISTORY_LENGTH:]
    
    r.hset(key, mapping={
        "messages": json.dumps(messages),
        "updated_at": datetime.utcnow().isoformat(),
        "message_count": str(len(messages))
    })
    r.expire(key, SESSION_TTL)
    
    return True

def delete_session(user_id: str, session_id: str) -> bool:
    session_key = f"{SESSION_PREFIX}{user_id}:{session_id}"
    summary_key = f"{SUMMARY_PREFIX}{user_id}:{session_id}"
    profile_key = f"{PROFILE_PREFIX}{user_id}:{session_id}"
    
    deleted = r.delete(session_key, summary_key, profile_key)
    
    # Очистка rate limit
    rate_key = f"{user_id}:{session_id}"
    if rate_key in rate_limit_tracker:
        del rate_limit_tracker[rate_key]
    
    logger.info(f"✓ Session deleted: {user_id}:{session_id}")
    return deleted > 0

def list_sessions(user_id: str, limit: int = 100) -> List[str]:
    """Список сессий конкретного пользователя"""
    pattern = f"{SESSION_PREFIX}{user_id}:*"
    keys = r.keys(pattern)
    session_ids = [k.replace(f"{SESSION_PREFIX}{user_id}:", "") for k in keys]
    return session_ids[:limit]

##################### 📜 SUMMARY FUNCTIONS (ШАГ 2 - УЛУЧШЕННАЯ ЛОГИКА) ####################

def get_summary(user_id: str, session_id: str) -> str:
    """Получение Conversation Summary"""
    key = f"{SUMMARY_PREFIX}{user_id}:{session_id}"
    data = r.get(key)
    return data if data else ""

def update_summary(user_id: str, session_id: str, summary: str) -> None:
    """Обновление summary"""
    key = f"{SUMMARY_PREFIX}{user_id}:{session_id}"
    r.set(key, summary, ex=SESSION_TTL)
    logger.info(f"✓ Summary updated: {user_id}:{session_id}")

def delete_summary(user_id: str, session_id: str) -> bool:
    key = f"{SUMMARY_PREFIX}{user_id}:{session_id}"
    return r.delete(key) > 0

##################### LLM CLIENT ####################

BASE_URL = f"http://{QWEN_HOST}:{QWEN_PORT}/v1/chat/completions"

async def send_to_llm(messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = MAX_RESPONSE_TOKENS) -> Dict[str, Any]:
    """LLM запрос с ограничением concurrency"""
    
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

async def generate_summary(old_summary: str, messages: List[Dict]) -> str:
    """
    📜 НОВАЯ ЛОГИКА SUMMARY (ШАГ 2)
    
    ❌ НЕ: "кратко перескажи весь диалог"
    ✅ А: "опиши цель разговора и текущий этап"
    
    Summary = состояние задачи, а не история
    """
    
    # Берём только последние 4 сообщения
    recent_messages = "\n".join([
        f"{m['role'].capitalize()}: {m['content'][:150]}"
        for m in messages[-4:]
    ])
    
    # 🎯 НОВЫЙ PROMPT - ФОКУС НА ЗАДАЧЕ, НЕ НА ИСТОРИИ
    prompt = f"""Analyze this conversation and describe the current state.

Previous summary: {old_summary if old_summary else "New conversation"}

Recent messages:
{recent_messages}

Create a brief summary (max 100 words) focusing ONLY on:
1. Main goal/topic of conversation
2. Current progress or stage
3. Open questions or next steps
4. Key decisions made

DO NOT include:
- User's name, language, or personal details
- Full conversation history
- Technical constants or system info

Be task-focused and concise."""

    llm_messages = [{"role": "user", "content": prompt}]
    
    try:
        result = await send_to_llm(llm_messages, temperature=0.3, max_tokens=MAX_SUMMARY_TOKENS)
        return result["content"].strip()
    except Exception as e:
        logger.error(f"✗ Summary generation failed: {e}")
        return old_summary

##################### 🎯 CHAT PROCESSING (УЛУЧШЕННАЯ СБОРКА PROMPT) ####################

def build_system_prompt(profile: Dict[str, Any], summary: str, messages: List[Dict]) -> str:
    """
    🔑 КЛЮЧЕВАЯ ФУНКЦИЯ - ПРАВИЛЬНАЯ СБОРКА КОНТЕКСТА
    
    Структура:
    1. USER PROFILE (стабильное)
    2. CONVERSATION SUMMARY (динамическое)
    3. RECENT MESSAGES (живой контекст)
    """
    
    parts = ["You are a helpful AI assistant."]
    
    # 🧠 A. USER PROFILE (если есть данные)
    profile_parts = []
    if profile.get("name"):
        profile_parts.append(f"User name: {profile['name']}")
    if profile.get("language"):
        profile_parts.append(f"Preferred language: {profile['language']}")
    if profile.get("role"):
        profile_parts.append(f"Role: {profile['role']}")
    if profile.get("stack"):
        profile_parts.append(f"Tech stack: {', '.join(profile['stack'])}")
    
    if profile_parts:
        parts.append("\nUSER PROFILE:\n" + "\n".join(profile_parts))
    
    # 📜 B. CONVERSATION SUMMARY
    if summary:
        parts.append(f"\nCONVERSATION SUMMARY:\n{summary[:300]}")
    
    # 💬 C. RECENT MESSAGES (последние 4)
    if messages:
        recent = messages[-4:] if len(messages) > 4 else messages
        history = "\n".join([
            f"{m['role'].capitalize()}: {m['content'][:200]}"
            for m in recent
        ])
        parts.append(f"\nRECENT MESSAGES:\n{history}")
    
    return "\n".join(parts)

async def process_chat_message(user_id: str, session_id: str, user_message: str) -> ChatResponse:
    """Обработка сообщения с новой архитектурой памяти"""
    
    # Rate limit
    if not check_rate_limit(user_id, session_id):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit: max {RATE_LIMIT_REQUESTS} req per {RATE_LIMIT_WINDOW}s"
        )
    
    # Получаем сессию
    session = get_session(user_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = session.get("messages", [])
    
    # 🆕 Получаем ВСЕ 3 типа памяти
    profile = get_profile(user_id, session_id)
    summary = get_summary(user_id, session_id)
    
    # 🎯 Собираем правильный system prompt
    system_content = build_system_prompt(profile, summary, messages)
    
    # LLM запрос
    llm_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_message}
    ]
    
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
    update_session(user_id, session_id, messages)
    
    # Обновляем summary если нужно
    new_summary = None
    if len(messages) >= SUMMARY_THRESHOLD:
        logger.info(f"📝 Generating summary for {user_id}:{session_id}")
        new_summary = await generate_summary(summary, messages)
        update_summary(user_id, session_id, new_summary)
    
    return ChatResponse(
        user_id=user_id,
        session_id=session_id,
        response=ai_response,
        timestamp=timestamp,
        summary=new_summary,
        tokens_used=result.get("tokens_used")
    )

##################### FASTAPI APP ####################

app = FastAPI(
    title="AI Chat Server - Memory Architecture v2",
    description="3-tier memory: Profile + Summary + Recent Messages",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_router = APIRouter(prefix="/chat", tags=["chat"])
session_router = APIRouter(prefix="/session", tags=["session"])
summary_router = APIRouter(prefix="/summary", tags=["summary"])
profile_router = APIRouter(prefix="/profile", tags=["profile"])

##################### HEALTH CHECK ####################

@app.get("/health")
async def health_check():
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
        "memory_architecture": {
            "profile": "stable facts",
            "summary": "conversation state",
            "recent": "last 6 messages"
        },
        "concurrency": {
            "current": LLM_CONCURRENCY - llm_semaphore._value,
            "max": LLM_CONCURRENCY
        },
        "timestamp": datetime.utcnow().isoformat()
    }

##################### CHAT ENDPOINTS ####################

@chat_router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send message (requires user_id + session_id)"""
    try:
        return await process_chat_message(req.user_id, req.session_id, req.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

##################### SESSION ENDPOINTS ####################

@session_router.post("/create")
def create_new_session(req: SessionCreate):
    """Create new session (requires user_id)"""
    session_id = create_session(req.user_id, req.metadata)
    return {
        "user_id": req.user_id,
        "session_id": session_id,
        "created_at": datetime.utcnow().isoformat()
    }

@session_router.get("/{user_id}/{session_id}")
def get_session_info(user_id: str, session_id: str):
    """Get session info"""
    session = get_session(user_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@session_router.delete("/{user_id}/{session_id}")
def remove_session(user_id: str, session_id: str):
    """Delete session"""
    if not delete_session(user_id, session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted", "user_id": user_id, "session_id": session_id}

@session_router.get("/{user_id}")
def get_user_sessions(user_id: str, limit: int = 100):
    """List all sessions for user"""
    sessions = list_sessions(user_id, limit)
    return {"user_id": user_id, "sessions": sessions, "count": len(sessions)}

##################### 🆕 PROFILE ENDPOINTS ####################

@profile_router.get("/{user_id}")
def read_user_profile(user_id: str):
    """Get общий user profile (не привязан к session)"""
    profile = get_user_profile(user_id)
    return {
        "user_id": user_id,
        "profile": profile
    }

@profile_router.get("/{user_id}/{session_id}")
def read_session_profile(user_id: str, session_id: str):
    """Get session-specific profile (или общий если нет)"""
    profile = get_profile(user_id, session_id)
    return {
        "user_id": user_id,
        "session_id": session_id,
        "profile": profile
    }

@profile_router.post("/update")
def edit_user_profile(req: ProfileUpdate):
    """Update общий user profile"""
    update_user_profile(req.user_id, req.profile_data)
    return {
        "user_id": req.user_id,
        "profile": req.profile_data,
        "updated_at": datetime.utcnow().isoformat()
    }

@profile_router.delete("/{user_id}/{session_id}")
def remove_profile(user_id: str, session_id: str):
    """Delete session profile"""
    if not delete_profile(user_id, session_id):
        raise HTTPException(status_code=404, detail="Profile not found")
    return {"message": "Profile deleted", "user_id": user_id, "session_id": session_id}

##################### SUMMARY ENDPOINTS ####################

@summary_router.get("/{user_id}/{session_id}")
def read_summary(user_id: str, session_id: str):
    """Get conversation summary"""
    summary = get_summary(user_id, session_id)
    return {
        "user_id": user_id,
        "session_id": session_id,
        "summary": summary if summary else "No summary available"
    }

@summary_router.post("/{user_id}/{session_id}")
def edit_summary(user_id: str, session_id: str, req: SummaryUpdate):
    """Update summary manually"""
    if not get_session(user_id, session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    update_summary(user_id, session_id, req.new_summary)
    return {
        "user_id": user_id,
        "session_id": session_id,
        "summary": req.new_summary,
        "updated_at": datetime.utcnow().isoformat()
    }

@summary_router.delete("/{user_id}/{session_id}")
def remove_summary(user_id: str, session_id: str):
    """Delete summary"""
    if not delete_summary(user_id, session_id):
        raise HTTPException(status_code=404, detail="Summary not found")
    return {"message": "Summary deleted", "user_id": user_id, "session_id": session_id}

@summary_router.post("/{user_id}/{session_id}/regenerate")
async def regenerate_summary(user_id: str, session_id: str):
    """Regenerate summary"""
    session = get_session(user_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = session.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="No messages")
    
    old_summary = get_summary(user_id, session_id)
    new_summary = await generate_summary(old_summary, messages)
    update_summary(user_id, session_id, new_summary)
    
    return {
        "user_id": user_id,
        "session_id": session_id,
        "summary": new_summary,
        "updated_at": datetime.utcnow().isoformat()
    }

##################### STATS ####################

@app.get("/stats")
def get_stats():
    """Server statistics"""
    return {
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
app.include_router(profile_router)

##################### STARTUP/SHUTDOWN ####################

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("🚀 AI CHAT SERVER - MEMORY ARCHITECTURE v2")
    logger.info("=" * 60)
    
    try:
        r.ping()
        logger.info("✓ Redis: CONNECTED")
    except Exception as e:
        logger.error(f"✗ Redis: FAILED - {e}")
    
    logger.info("📚 Memory Architecture:")
    logger.info("   🧠 Profile: stable user facts")
    logger.info("   📜 Summary: conversation state")
    logger.info("   💬 Recent: last 6 messages")
    logger.info("")
    logger.info(f"⚙️  Config:")
    logger.info(f"   • Max tokens: {MAX_RESPONSE_TOKENS}")
    logger.info(f"   • Max history: {MAX_HISTORY_LENGTH}")
    logger.info(f"   • Concurrency: {LLM_CONCURRENCY}")
    logger.info(f"   • Rate limit: {RATE_LIMIT_REQUESTS}/{RATE_LIMIT_WINDOW}s")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("👋 Shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info"
    )