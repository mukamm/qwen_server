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
import re

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

# Redis prefixes
SESSION_PREFIX = "session:"
SUMMARY_PREFIX = "summary:"
USER_PROFILE_PREFIX = "user_profile:"
DIALOG_STATE_PREFIX = "dialog_state:"  # 🆕 INTENT STORAGE
SESSION_TTL = 86400 * 7  # 7 days

# Memory limits
MAX_HISTORY_LENGTH = 30
CONTEXT_WINDOW = 6
MAX_RESPONSE_TOKENS = 300
MAX_SUMMARY_TOKENS = 150
SUMMARY_THRESHOLD = 8
PROFILE_UPDATE_THRESHOLD = 5

# Concurrency
LLM_CONCURRENCY = 2
RATE_LIMIT_REQUESTS = 1
RATE_LIMIT_WINDOW = 3

# 🆕 USER LEVELS
USER_LEVELS = ["beginner", "junior", "middle", "senior", "expert"]
RESPONSE_MODES = ["learn", "debug", "inspect", "design", "quick"]

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
    user_id: str
    session_id: str
    message: str

class ChatResponse(BaseModel):
    user_id: str
    session_id: str
    response: str
    timestamp: str
    intent: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    profile_updated: Optional[bool] = None
    tokens_used: Optional[int] = None
    rules_applied: Optional[List[str]] = None  # 🆕

class SessionCreate(BaseModel):
    user_id: str
    metadata: Optional[Dict[str, Any]] = None

class ProfileUpdate(BaseModel):
    user_id: str
    profile_data: Dict[str, Any]

##################### UTILITIES ####################

def generate_id() -> str:
    return str(uuid.uuid4())

##################### 🆕 STEP 2: USER PROFILE (PASSPORT) ####################

def get_user_profile(user_id: str) -> Dict[str, Any]:
    """
    📋 USER PROFILE = ПАСПОРТ ПОЛЬЗОВАТЕЛЯ
    
    Хранит ТОЛЬКО стабильные факты:
    - имя
    - возраст
    - роль (student/engineer/designer)
    - уровень (beginner/junior/middle/senior/expert)
    - технологии
    - язык общения
    
    НЕ ХРАНИТ:
    - текущую задачу
    - эмоции
    - последние вопросы
    """
    key = f"{USER_PROFILE_PREFIX}{user_id}"
    data = r.get(key)
    
    if not data:
        return {
            "name": None,
            "age": None,
            "role": None,  # student/engineer/designer/etc
            "level": "junior",  # beginner/junior/middle/senior/expert
            "tech_stack": [],
            "language": "en",  # en/ru/etc
            "interests": [],
            "learning_goals": []
        }
    
    return json.loads(data)

def update_user_profile(user_id: str, profile_data: Dict[str, Any]) -> None:
    key = f"{USER_PROFILE_PREFIX}{user_id}"
    r.set(key, json.dumps(profile_data), ex=SESSION_TTL * 4)
    logger.info(f"✓ Profile updated: {user_id}")

def merge_profile_facts(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    merged = old.copy()
    
    for key, value in new.items():
        if key not in merged:
            merged[key] = value
        elif value is None:
            continue
        elif isinstance(value, list):
            if not isinstance(merged[key], list):
                merged[key] = []
            merged[key] = list(set(merged[key] + value))
        elif isinstance(value, dict):
            if not isinstance(merged[key], dict):
                merged[key] = {}
            merged[key].update(value)
        else:
            if value:
                merged[key] = value
    
    return merged

##################### 🆕 STEP 3: DIALOG STATE (INTENT) ####################

def get_dialog_state(user_id: str, session_id: str) -> Dict[str, Any]:
    """
    🎯 DIALOG STATE = ТЕКУЩАЯ ЦЕЛЬ И КОНТЕКСТ
    
    Отвечает на вопросы:
    - Что пользователь хочет СЕЙЧАС?
    - В каком режиме работаем? (learn/debug/inspect/design)
    - Какой уровень детализации нужен?
    - Что УЖЕ понятно?
    - Что ЗАПРЕЩЕНО объяснять?
    
    БЕЗ ЭТОГО LLM ВСЕГДА БУДЕТ ТУПИТЬ
    """
    key = f"{DIALOG_STATE_PREFIX}{user_id}:{session_id}"
    data = r.get(key)
    
    if not data:
        return {
            "current_goal": None,  # "learn React", "debug error", "design system"
            "mode": "learn",  # learn/debug/inspect/design/quick
            "detail_level": "normal",  # brief/normal/detailed
            "understood_concepts": [],  # что уже понятно
            "forbidden_topics": [],  # что НЕ объяснять
            "context_type": None,  # code/theory/architecture/practice
            "last_updated": None
        }
    
    return json.loads(data)

def update_dialog_state(user_id: str, session_id: str, state: Dict[str, Any]) -> None:
    key = f"{DIALOG_STATE_PREFIX}{user_id}:{session_id}"
    state["last_updated"] = datetime.utcnow().isoformat()
    r.set(key, json.dumps(state), ex=SESSION_TTL)
    logger.info(f"✓ Dialog state updated: {state.get('current_goal', 'unknown')}")

##################### SESSION FUNCTIONS ####################

def create_session(user_id: str, metadata: Optional[Dict] = None) -> str:
    session_id = generate_id()
    key = f"{SESSION_PREFIX}{user_id}:{session_id}"
    
    r.hset(key, mapping={
        "messages": json.dumps([]),
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "metadata": json.dumps(metadata or {}),
        "message_count": "0",
        "last_profile_check": "0"
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
        "message_count": int(data.get("message_count", 0)),
        "last_profile_check": int(data.get("last_profile_check", 0))
    }

def update_session(user_id: str, session_id: str, messages: List[Dict], last_profile_check: Optional[int] = None) -> bool:
    key = f"{SESSION_PREFIX}{user_id}:{session_id}"
    
    if not r.exists(key):
        return False
    
    if len(messages) > MAX_HISTORY_LENGTH:
        messages = messages[-MAX_HISTORY_LENGTH:]
    
    update_data = {
        "messages": json.dumps(messages),
        "updated_at": datetime.utcnow().isoformat(),
        "message_count": str(len(messages))
    }
    
    if last_profile_check is not None:
        update_data["last_profile_check"] = str(last_profile_check)
    
    r.hset(key, mapping=update_data)
    r.expire(key, SESSION_TTL)
    
    return True

def delete_session(user_id: str, session_id: str) -> bool:
    session_key = f"{SESSION_PREFIX}{user_id}:{session_id}"
    summary_key = f"{SUMMARY_PREFIX}{user_id}:{session_id}"
    state_key = f"{DIALOG_STATE_PREFIX}{user_id}:{session_id}"
    
    deleted = r.delete(session_key, summary_key, state_key)
    
    rate_key = f"{user_id}:{session_id}"
    if rate_key in rate_limit_tracker:
        del rate_limit_tracker[rate_key]
    
    logger.info(f"✓ Session deleted: {user_id}:{session_id}")
    return deleted > 0

##################### 🆕 STEP 4: SUMMARY (ТОЧКА НА КАРТЕ) ####################

def get_summary(user_id: str, session_id: str) -> str:
    key = f"{SUMMARY_PREFIX}{user_id}:{session_id}"
    data = r.get(key)
    return data if data else ""

def update_summary(user_id: str, session_id: str, summary: str) -> None:
    key = f"{SUMMARY_PREFIX}{user_id}:{session_id}"
    r.set(key, summary, ex=SESSION_TTL)
    logger.info(f"✓ Summary updated: {user_id}:{session_id}")

async def generate_summary(old_summary: str, messages: List[Dict]) -> str:
    """
    📍 SUMMARY = ТОЧКА НА КАРТЕ
    
    Отвечает ТОЛЬКО на:
    - О чём диалог СЕЙЧАС?
    - К какому результату идём?
    - Что уже решено?
    
    НЕ ДОЛЖЕН:
    - Пересказывать всю историю
    - Содержать старые темы
    """
    
    recent = "\n".join([
        f"{m['role'].capitalize()}: {m['content'][:150]}"
        for m in messages[-4:]
    ])
    
    prompt = f"""Analyze current dialog state.

Previous state: {old_summary if old_summary else "New conversation"}

Recent exchange:
{recent}

Create concise summary (max 80 words) answering ONLY:
1. What is the CURRENT topic/goal?
2. What result are we working toward?
3. What has been DECIDED/SOLVED?

FORBIDDEN:
- Full history retelling
- Old/resolved topics
- User personal details

Focus on CURRENT STATE, not past."""

    llm_messages = [{"role": "user", "content": prompt}]
    
    try:
        result = await send_to_llm(llm_messages, temperature=0.3, max_tokens=MAX_SUMMARY_TOKENS)
        return result["content"].strip()
    except Exception as e:
        logger.error(f"✗ Summary generation failed: {e}")
        return old_summary

##################### 🆕 STEP 3: INTENT EXTRACTION ####################

async def extract_intent(user_message: str, current_state: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    🎯 ИЗВЛЕЧЕНИЕ DIALOG STATE ИЗ СООБЩЕНИЯ
    
    Определяет:
    - current_goal: что хочет пользователь
    - mode: learn/debug/inspect/design/quick
    - detail_level: brief/normal/detailed
    - understood_concepts: что УЖЕ понятно
    - forbidden_topics: что НЕ объяснять
    """
    
    current_goal = current_state.get("current_goal", "unknown")
    understood = ", ".join(current_state.get("understood_concepts", [])[:5])
    
    prompt = f"""Extract dialog intent from user message.

USER LEVEL: {profile.get('level', 'junior')}
CURRENT GOAL: {current_goal}
UNDERSTOOD: {understood}

USER MESSAGE: "{user_message}"

Determine and return JSON:
{{
  "current_goal": "brief description of what user wants NOW",
  "mode": "learn|debug|inspect|design|quick",
  "detail_level": "brief|normal|detailed",
  "understood_concepts": ["concept1", "concept2"],
  "forbidden_topics": ["basics", "already explained"],
  "context_type": "code|theory|architecture|practice"
}}

RULES:
- If user says "I know X" → add X to understood_concepts, forbidden_topics
- If user asks "how to debug" → mode: "debug"
- If user asks "explain" → mode: "learn"
- If user asks "show code" → mode: "inspect"
- If user says "briefly" → detail_level: "brief"
- Return ONLY valid JSON, no text."""

    llm_messages = [{"role": "user", "content": prompt}]
    
    try:
        result = await send_to_llm(llm_messages, temperature=0.1, max_tokens=200)
        response = result["content"].strip()
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            intent = json.loads(json_match.group())
            logger.info(f"✓ Intent extracted: {intent.get('mode')} - {intent.get('current_goal')}")
            return intent
        
        logger.warning("✗ No JSON in intent extraction")
        return current_state
        
    except Exception as e:
        logger.error(f"✗ Intent extraction failed: {e}")
        return current_state

##################### 🆕 STEP 6: CONTROLLER (ПРАВИЛА) ####################

def build_response_rules(profile: Dict[str, Any], state: Dict[str, Any]) -> List[str]:
    """
    🚦 CONTROLLER - ПРАВИЛА ОТВЕТА
    
    Решает перед КАЖДЫМ ответом:
    - Можно ли давать код?
    - Можно ли повторять?
    - Можно ли объяснять базу?
    - Можно ли уходить в сторону?
    
    LLM = ИСПОЛНИТЕЛЬ
    BACKEND = МОЗГ
    """
    
    rules = []
    
    level = profile.get("level", "junior")
    mode = state.get("mode", "learn")
    detail_level = state.get("detail_level", "normal")
    understood = state.get("understood_concepts", [])
    forbidden = state.get("forbidden_topics", [])
    
    # 1. Правила по уровню
    if level == "beginner":
        rules.append("Explain like to a beginner, use simple terms")
        rules.append("Include basic examples")
        rules.append("NO assumptions about prior knowledge")
    elif level in ["middle", "senior", "expert"]:
        rules.append("Skip basic explanations")
        rules.append("Use technical terms freely")
        rules.append("Focus on advanced concepts")
    
    # 2. Правила по режиму
    if mode == "debug":
        rules.append("Focus on finding the error")
        rules.append("Provide specific solution, not theory")
        rules.append("Show corrected code")
    elif mode == "quick":
        rules.append("Answer in 1-2 sentences maximum")
        rules.append("NO long explanations")
    elif mode == "design":
        rules.append("Focus on architecture and patterns")
        rules.append("Explain trade-offs")
    
    # 3. Правила по детализации
    if detail_level == "brief":
        rules.append("Keep response under 100 words")
        rules.append("Only essential information")
    elif detail_level == "detailed":
        rules.append("Provide thorough explanation")
        rules.append("Include examples and edge cases")
    
    # 4. Запреты
    if understood:
        rules.append(f"DO NOT explain these (user knows): {', '.join(understood[:3])}")
    
    if forbidden:
        rules.append(f"FORBIDDEN topics: {', '.join(forbidden[:3])}")
    
    # 5. Общие правила
    rules.append("NO repetition of previous answers")
    rules.append("Stay on topic, no tangents")
    rules.append("If asked something off-topic, politely redirect")
    
    return rules

##################### 🆕 STEP 7: SMART PROMPT ASSEMBLY ####################

def build_system_prompt(profile: Dict[str, Any], state: Dict[str, Any], summary: str, messages: List[Dict]) -> str:
    """
    🧠 УМНАЯ СБОРКА ПРОМПТА
    
    Собирает контекст из:
    1. Краткий профиль (паспорт)
    2. Dialog state (текущая цель)
    3. Summary (где мы сейчас)
    4. Жёсткие правила ответа
    5. Только последние CONTEXT_WINDOW сообщений
    """
    
    parts = ["You are a helpful AI assistant."]
    
    # 1. USER PROFILE (PASSPORT)
    profile_lines = []
    if profile.get("name"):
        profile_lines.append(f"Name: {profile['name']}")
    if profile.get("role"):
        profile_lines.append(f"Role: {profile['role']}")
    profile_lines.append(f"Level: {profile.get('level', 'junior')}")
    if profile.get("tech_stack"):
        profile_lines.append(f"Tech: {', '.join(profile['tech_stack'][:3])}")
    
    if profile_lines:
        parts.append(f"\nUSER PROFILE:\n" + "\n".join(profile_lines))
    
    # 2. DIALOG STATE (INTENT)
    if state.get("current_goal"):
        parts.append(f"\nCURRENT GOAL: {state['current_goal']}")
        parts.append(f"MODE: {state.get('mode', 'learn')}")
        parts.append(f"DETAIL LEVEL: {state.get('detail_level', 'normal')}")
    
    if state.get("understood_concepts"):
        parts.append(f"USER ALREADY KNOWS: {', '.join(state['understood_concepts'][:5])}")
    
    # 3. SUMMARY (WHERE WE ARE)
    if summary:
        parts.append(f"\nCONVERSATION STATE:\n{summary[:250]}")
    
    # 4. RESPONSE RULES
    rules = build_response_rules(profile, state)
    if rules:
        parts.append("\nRESPONSE RULES:")
        for rule in rules[:8]:  # max 8 rules
            parts.append(f"- {rule}")
    
    # 5. RECENT MESSAGES (only last CONTEXT_WINDOW)
    if messages:
        recent = messages[-CONTEXT_WINDOW:]
        history = "\n".join([
            f"{m['role'].capitalize()}: {m['content'][:200]}"
            for m in recent
        ])
        parts.append(f"\nRECENT EXCHANGE:\n{history}")
    
    return "\n".join(parts)

##################### LLM CLIENT ####################

BASE_URL = f"http://{QWEN_HOST}:{QWEN_PORT}/v1/chat/completions"

async def send_to_llm(messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = MAX_RESPONSE_TOKENS) -> Dict[str, Any]:
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

##################### 🆕 STEP 8: RESPONSE VALIDATION ####################

def validate_response(response: str, state: Dict[str, Any], previous_responses: List[str]) -> bool:
    """
    ✅ ПРОВЕРКА ОТВЕТА
    
    Проверяет:
    - Релевантен ли ответ?
    - Не повторяет ли предыдущие?
    - Не ушёл ли в сторону?
    
    Если плохой — НЕ СОХРАНЯТЬ КАК ИСТИНУ
    """
    
    # 1. Проверка на повторение
    response_lower = response.lower()
    for prev in previous_responses[-3:]:  # last 3
        prev_lower = prev.lower()
        # Если более 60% совпадение - это повтор
        overlap = len(set(response_lower.split()) & set(prev_lower.split()))
        total = len(set(response_lower.split()))
        if total > 0 and overlap / total > 0.6:
            logger.warning("✗ Response rejected: too similar to previous")
            return False
    
    # 2. Проверка длины
    if len(response) < 10:
        logger.warning("✗ Response rejected: too short")
        return False
    
    # 3. Проверка на off-topic (опционально)
    forbidden = state.get("forbidden_topics", [])
    if forbidden:
        for topic in forbidden:
            if topic.lower() in response_lower:
                logger.warning(f"✗ Response rejected: contains forbidden topic '{topic}'")
                return False
    
    return True

##################### 🎯 MAIN CHAT PROCESSOR ####################

async def process_chat_message(user_id: str, session_id: str, user_message: str) -> ChatResponse:
    """
    🎯 ГЛАВНЫЙ ПРОЦЕССОР СООБЩЕНИЙ
    
    Шаги:
    1. Получить Profile + State + Summary + History
    2. Извлечь Intent из сообщения
    3. Обновить Dialog State
    4. Собрать умный промпт
    5. Получить ответ от LLM
    6. Проверить ответ
    7. Сохранить или отклонить
    """
    
    # Rate limit
    if not check_rate_limit(user_id, session_id):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit: max {RATE_LIMIT_REQUESTS} req per {RATE_LIMIT_WINDOW}s"
        )
    
    # Get session
    session = get_session(user_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = session.get("messages", [])
    
    # STEP 1: Load memory tiers
    profile = get_user_profile(user_id)
    state = get_dialog_state(user_id, session_id)
    summary = get_summary(user_id, session_id)
    
    # STEP 2: Extract intent
    logger.info("🎯 Extracting intent...")
    new_state = await extract_intent(user_message, state, profile)
    
    # STEP 3: Update dialog state
    update_dialog_state(user_id, session_id, new_state)
    
    # STEP 4: Build smart prompt
    logger.info("🧠 Building smart prompt...")
    system_content = build_system_prompt(profile, new_state, summary, messages)
    
    # Get rules for response metadata
    rules = build_response_rules(profile, new_state)
    
    # STEP 5: Get LLM response
    llm_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_message}
    ]
    
    result = await send_to_llm(llm_messages)
    ai_response = result["content"]
    
    # STEP 6: Validate response
    previous_responses = [m["content"] for m in messages if m.get("role") == "assistant"]
    
    if not validate_response(ai_response, new_state, previous_responses):
        # Retry once with stronger rules
        logger.warning("⚠️  First response rejected, retrying with stricter rules...")
        rules.append("CRITICAL: This is a retry. Previous response was rejected for repetition or off-topic content.")
        rules.append("Provide a COMPLETELY DIFFERENT answer with NEW information.")
        
        system_content = build_system_prompt(profile, new_state, summary, messages)
        llm_messages[0]["content"] = system_content
        
        result = await send_to_llm(llm_messages, temperature=0.9)  # higher temp for variety
        ai_response = result["content"]
    
    # STEP 7: Save messages
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
    
    update_session(user_id, session_id, messages)
    
    # Update summary if needed
    new_summary = None
    if len(messages) >= SUMMARY_THRESHOLD:
        logger.info("📝 Updating summary...")
        new_summary = await generate_summary(summary, messages)
        update_summary(user_id, session_id, new_summary)
    
    return ChatResponse(
        user_id=user_id,
        session_id=session_id,
        response=ai_response,
        timestamp=timestamp,
        intent=new_state,
        summary=new_summary,
        tokens_used=result.get("tokens_used"),
        rules_applied=rules[:5]  # top 5 rules for debugging
    )

##################### FASTAPI APP ####################

app = FastAPI(
    title="Smart Chat Server v4.0",
    description="Intent-Driven Architecture with Dialog State",
    version="4.0.0"
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
profile_router = APIRouter(prefix="/profile", tags=["profile"])
summary_router = APIRouter(prefix="/summary", tags=["summary"])
state_router = APIRouter(prefix="/state", tags=["dialog_state"])

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
        "architecture": {
            "tier_1": "User Profile (passport - stable facts)",
            "tier_2": "Dialog State (intent - current goal)",
            "tier_3": "Summary (where we are now)",
            "tier_4": "History (raw messages for analysis)"
        },
        "features": {
            "intent_extraction": "automatic",
            "response_validation": "enabled",
            "dynamic_rules": "enabled",
            "modes": RESPONSE_MODES,
            "levels": USER_LEVELS
        },
        "timestamp": datetime.utcnow().isoformat()
    }

##################### CHAT ENDPOINTS ####################

@chat_router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    🎯 Main chat endpoint with intent-driven processing
    
    Features:
    - Automatic intent extraction
    - Dynamic response rules
    - Response validation
    - Smart context assembly
    """
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
    """Create new session"""
    session_id = create_session(req.user_id, req.metadata)
    return {
        "user_id": req.user_id,
        "session_id": session_id,
        "created_at": datetime.utcnow().isoformat()
    }

@session_router.get("/{user_id}/{session_id}")
def get_session_info(user_id: str, session_id: str):
    """Get full session info with all memory tiers"""
    session = get_session(user_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Include all memory tiers
    profile = get_user_profile(user_id)
    state = get_dialog_state(user_id, session_id)
    summary = get_summary(user_id, session_id)
    
    return {
        **session,
        "profile": profile,
        "dialog_state": state,
        "summary": summary
    }

@session_router.delete("/{user_id}/{session_id}")
def remove_session(user_id: str, session_id: str):
    """Delete session and all associated data"""
    if not delete_session(user_id, session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted", "user_id": user_id, "session_id": session_id}

##################### PROFILE ENDPOINTS ####################

@profile_router.get("/{user_id}")
def read_user_profile(user_id: str):
    """Get user profile (passport)"""
    profile = get_user_profile(user_id)
    return {
        "user_id": user_id,
        "profile": profile
    }

@profile_router.post("/update")
def edit_user_profile(req: ProfileUpdate):
    """Update user profile"""
    current = get_user_profile(req.user_id)
    merged = merge_profile_facts(current, req.profile_data)
    update_user_profile(req.user_id, merged)
    return {
        "user_id": req.user_id,
        "profile": merged,
        "updated_at": datetime.utcnow().isoformat()
    }

@profile_router.delete("/{user_id}")
def delete_user_profile(user_id: str):
    """Delete user profile"""
    key = f"{USER_PROFILE_PREFIX}{user_id}"
    deleted = r.delete(key)
    if not deleted:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {"message": "Profile deleted", "user_id": user_id}

##################### DIALOG STATE ENDPOINTS ####################

@state_router.get("/{user_id}/{session_id}")
def read_dialog_state(user_id: str, session_id: str):
    """Get current dialog state (intent)"""
    state = get_dialog_state(user_id, session_id)
    return {
        "user_id": user_id,
        "session_id": session_id,
        "dialog_state": state
    }

@state_router.post("/{user_id}/{session_id}/update")
def manual_state_update(user_id: str, session_id: str, state: Dict[str, Any]):
    """Manually update dialog state"""
    update_dialog_state(user_id, session_id, state)
    return {
        "user_id": user_id,
        "session_id": session_id,
        "dialog_state": state,
        "updated_at": datetime.utcnow().isoformat()
    }

@state_router.post("/{user_id}/{session_id}/reset")
def reset_dialog_state(user_id: str, session_id: str):
    """Reset dialog state to default"""
    default_state = {
        "current_goal": None,
        "mode": "learn",
        "detail_level": "normal",
        "understood_concepts": [],
        "forbidden_topics": [],
        "context_type": None
    }
    update_dialog_state(user_id, session_id, default_state)
    return {
        "message": "Dialog state reset",
        "user_id": user_id,
        "session_id": session_id,
        "dialog_state": default_state
    }

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

@summary_router.post("/{user_id}/{session_id}/regenerate")
async def regenerate_summary(user_id: str, session_id: str):
    """Regenerate summary from current messages"""
    session = get_session(user_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = session.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="No messages to summarize")
    
    old_summary = get_summary(user_id, session_id)
    new_summary = await generate_summary(old_summary, messages)
    update_summary(user_id, session_id, new_summary)
    
    return {
        "user_id": user_id,
        "session_id": session_id,
        "summary": new_summary,
        "updated_at": datetime.utcnow().isoformat()
    }

@summary_router.delete("/{user_id}/{session_id}")
def remove_summary(user_id: str, session_id: str):
    """Delete summary"""
    key = f"{SUMMARY_PREFIX}{user_id}:{session_id}"
    deleted = r.delete(key)
    if not deleted:
        raise HTTPException(status_code=404, detail="Summary not found")
    return {"message": "Summary deleted", "user_id": user_id, "session_id": session_id}

##################### ANALYTICS & DEBUG ####################

@app.get("/stats")
def get_stats():
    """Server statistics and configuration"""
    return {
        "active_rate_limits": len(rate_limit_tracker),
        "llm_queue": LLM_CONCURRENCY - llm_semaphore._value,
        "config": {
            "max_tokens": MAX_RESPONSE_TOKENS,
            "max_history": MAX_HISTORY_LENGTH,
            "context_window": CONTEXT_WINDOW,
            "concurrency": LLM_CONCURRENCY,
            "rate_limit": f"{RATE_LIMIT_REQUESTS}/{RATE_LIMIT_WINDOW}s",
            "profile_update_threshold": PROFILE_UPDATE_THRESHOLD
        },
        "architecture": {
            "memory_tiers": 4,
            "intent_driven": True,
            "response_validation": True,
            "dynamic_rules": True
        },
        "supported": {
            "modes": RESPONSE_MODES,
            "levels": USER_LEVELS
        }
    }

@app.get("/debug/{user_id}/{session_id}")
async def debug_session(user_id: str, session_id: str):
    """
    🔍 Debug endpoint - shows complete memory state
    
    Useful for understanding how the system works
    """
    session = get_session(user_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    profile = get_user_profile(user_id)
    state = get_dialog_state(user_id, session_id)
    summary = get_summary(user_id, session_id)
    rules = build_response_rules(profile, state)
    
    messages = session.get("messages", [])
    recent_messages = messages[-CONTEXT_WINDOW:] if messages else []
    
    return {
        "user_id": user_id,
        "session_id": session_id,
        "memory_tiers": {
            "tier_1_profile": profile,
            "tier_2_dialog_state": state,
            "tier_3_summary": summary,
            "tier_4_history_count": len(messages)
        },
        "context_sent_to_llm": {
            "profile_fields": [k for k, v in profile.items() if v],
            "dialog_state": state,
            "summary_length": len(summary) if summary else 0,
            "recent_messages_count": len(recent_messages),
            "recent_messages": recent_messages
        },
        "active_rules": rules,
        "system_prompt_preview": build_system_prompt(profile, state, summary, messages)[:500] + "..."
    }

##################### TESTING ENDPOINTS ####################

@app.post("/test/scenario")
async def test_scenario(user_id: str, scenario: str):
    """
    🧪 Test different user scenarios
    
    Scenarios:
    - beginner_learning
    - senior_debugging
    - quick_answers
    - detailed_explanation
    """
    
    # Create test session
    session_id = create_session(user_id, {"test_scenario": scenario})
    
    # Set profile based on scenario
    if scenario == "beginner_learning":
        profile = {
            "name": "TestUser",
            "level": "beginner",
            "role": "student",
            "tech_stack": [],
            "language": "en"
        }
        test_message = "How do I create a function in Python?"
        
    elif scenario == "senior_debugging":
        profile = {
            "name": "TestUser",
            "level": "senior",
            "role": "engineer",
            "tech_stack": ["Python", "FastAPI", "Redis"],
            "language": "en"
        }
        test_message = "My Redis connection keeps timing out in production"
        
    elif scenario == "quick_answers":
        profile = {
            "name": "TestUser",
            "level": "middle",
            "role": "developer",
            "tech_stack": ["JavaScript"],
            "language": "en"
        }
        test_message = "Quick: what's the difference between let and const?"
        
    elif scenario == "detailed_explanation":
        profile = {
            "name": "TestUser",
            "level": "junior",
            "role": "student",
            "tech_stack": ["React"],
            "language": "en"
        }
        test_message = "Explain React hooks in detail with examples"
        
    else:
        raise HTTPException(status_code=400, detail=f"Unknown scenario: {scenario}")
    
    # Update profile
    update_user_profile(user_id, profile)
    
    # Process message
    response = await process_chat_message(user_id, session_id, test_message)
    
    return {
        "scenario": scenario,
        "session_id": session_id,
        "test_message": test_message,
        "profile_used": profile,
        "response": response
    }

##################### INCLUDE ROUTERS ####################

app.include_router(chat_router)
app.include_router(session_router)
app.include_router(profile_router)
app.include_router(state_router)
app.include_router(summary_router)

##################### STARTUP/SHUTDOWN ####################

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 80)
    logger.info("🚀 SMART CHAT SERVER v4.0 - INTENT-DRIVEN ARCHITECTURE")
    logger.info("=" * 80)
    
    try:
        r.ping()
        logger.info("✓ Redis: CONNECTED")
    except Exception as e:
        logger.error(f"✗ Redis: FAILED - {e}")
    
    logger.info("")
    logger.info("📚 4-TIER MEMORY ARCHITECTURE:")
    logger.info("   1️⃣  USER PROFILE    → Passport (stable facts)")
    logger.info("   2️⃣  DIALOG STATE    → Intent (current goal, mode)")
    logger.info("   3️⃣  SUMMARY         → Where we are now")
    logger.info("   4️⃣  HISTORY         → Raw messages (analysis only)")
    logger.info("")
    logger.info("🎯 KEY FEATURES:")
    logger.info("   ✓ Automatic intent extraction from each message")
    logger.info("   ✓ Dynamic response rules based on context")
    logger.info("   ✓ Response validation (anti-repetition)")
    logger.info("   ✓ Smart prompt assembly (only relevant context)")
    logger.info("")
    logger.info("⚙️  CONFIGURATION:")
    logger.info(f"   • Max tokens: {MAX_RESPONSE_TOKENS}")
    logger.info(f"   • History stored: {MAX_HISTORY_LENGTH} messages")
    logger.info(f"   • Context sent to LLM: {CONTEXT_WINDOW} messages")
    logger.info(f"   • Concurrency: {LLM_CONCURRENCY} parallel requests")
    logger.info(f"   • Rate limit: {RATE_LIMIT_REQUESTS} req/{RATE_LIMIT_WINDOW}s")
    logger.info("")
    logger.info("🔧 SUPPORTED MODES:")
    logger.info(f"   • Response modes: {', '.join(RESPONSE_MODES)}")
    logger.info(f"   • User levels: {', '.join(USER_LEVELS)}")
    logger.info("")
    logger.info("🧪 TEST ENDPOINTS:")
    logger.info("   • POST /test/scenario - Test different user scenarios")
    logger.info("   • GET /debug/{user_id}/{session_id} - Full memory inspection")
    logger.info("")
    logger.info("=" * 80)
    logger.info("LLM = ИСПОЛНИТЕЛЬ | BACKEND = МОЗГ")
    logger.info("=" * 80)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("👋 Shutting down Smart Chat Server v4.0...")

##################### MAIN ####################

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info"
    )