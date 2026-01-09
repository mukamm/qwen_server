from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import redis
import uuid
import json
import requests
import logging
from datetime import datetime

####################
# LOGGING
####################

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

####################
# CONFIGURATION
####################

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = None
REDIS_DB = 0

QWEN_HOST = "localhost"
QWEN_PORT = 8001
QWEN_MODEL = "qwen-7b-instruct"
QWEN_TIMEOUT = 30

SESSION_PREFIX = "session:"
SUMMARY_PREFIX = "summary:"
SESSION_TTL = 86400 * 7  # 7 days

MAX_HISTORY_LENGTH = 20
SUMMARY_THRESHOLD = 10

####################
# REDIS CONNECTION
####################

r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    db=REDIS_DB,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_keepalive=True
)

####################
# PYDANTIC MODELS
####################

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

class SummaryUpdate(BaseModel):
    new_summary: str

class SessionCreate(BaseModel):
    metadata: Optional[Dict[str, Any]] = None

####################
# UTILITIES
####################

def generate_id() -> str:
    return str(uuid.uuid4())

def serialize_messages(messages: List[Message]) -> str:
    return json.dumps([{"role": m.role, "content": m.content, "timestamp": m.timestamp} for m in messages])

def deserialize_messages(data: str) -> List[Message]:
    return [Message(**m) for m in json.loads(data)]

####################
# SESSION FUNCTIONS
####################

def create_session(metadata: Optional[Dict] = None) -> str:
    session_id = generate_id()
    key = f"{SESSION_PREFIX}{session_id}"
    
    r.hset(key, mapping={
        "messages": json.dumps([]),
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "metadata": json.dumps(metadata or {})
    })
    r.expire(key, SESSION_TTL)
    
    logger.info(f"Session created: {session_id}")
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
        "metadata": json.loads(data.get("metadata", "{}"))
    }

def update_session(session_id: str, messages: List[Dict]) -> bool:
    key = f"{SESSION_PREFIX}{session_id}"
    
    if not r.exists(key):
        return False
    
    # Keep only recent messages
    if len(messages) > MAX_HISTORY_LENGTH:
        messages = messages[-MAX_HISTORY_LENGTH:]
    
    r.hset(key, mapping={
        "messages": json.dumps(messages),
        "updated_at": datetime.utcnow().isoformat()
    })
    r.expire(key, SESSION_TTL)
    
    return True

def delete_session(session_id: str) -> bool:
    key = f"{SESSION_PREFIX}{session_id}"
    summary_key = f"{SUMMARY_PREFIX}{session_id}"
    
    deleted = r.delete(key, summary_key)
    logger.info(f"Session deleted: {session_id}")
    return deleted > 0

def list_sessions(limit: int = 100) -> List[str]:
    pattern = f"{SESSION_PREFIX}*"
    keys = r.keys(pattern)
    session_ids = [k.replace(SESSION_PREFIX, "") for k in keys]
    return session_ids[:limit]

####################
# SUMMARY FUNCTIONS
####################

def get_summary(session_id: str) -> str:
    key = f"{SUMMARY_PREFIX}{session_id}"
    data = r.get(key)
    return data if data else ""

def update_summary(session_id: str, summary: str) -> None:
    key = f"{SUMMARY_PREFIX}{session_id}"
    r.set(key, summary, ex=SESSION_TTL)
    logger.info(f"Summary updated for session: {session_id}")

def delete_summary(session_id: str) -> bool:
    key = f"{SUMMARY_PREFIX}{session_id}"
    return r.delete(key) > 0

####################
# LLM CLIENT
####################

BASE_URL = f"http://{QWEN_HOST}:{QWEN_PORT}/v1/chat/completions"

def send_to_llm(messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 2000) -> str:
    try:
        payload = {
            "messages": messages,
            "model": QWEN_MODEL,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(BASE_URL, json=payload, timeout=QWEN_TIMEOUT)
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        logger.info(f"LLM response received, length: {len(content)}")
        return content
        
    except requests.exceptions.Timeout:
        logger.error("LLM request timeout")
        raise HTTPException(status_code=504, detail="LLM service timeout")
    except requests.exceptions.RequestException as e:
        logger.error(f"LLM request error: {e}")
        raise HTTPException(status_code=502, detail=f"LLM service error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in LLM call: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

def generate_summary(old_summary: str, new_messages: List[Dict]) -> str:
    """Generate or update conversation summary"""
    
    recent_messages = "\n".join([
        f"{m['role'].capitalize()}: {m['content']}" 
        for m in new_messages[-5:]
    ])
    
    prompt = f"""You are an AI that creates concise conversation summaries for memory purposes.

Old summary:
{old_summary if old_summary else "No previous summary"}

Recent messages:
{recent_messages}

Create a brief, factual summary that captures:
- Key topics discussed
- Important names, dates, or facts mentioned
- User's preferences or requirements
- Action items or decisions

Keep the summary under 200 words and focus only on important information."""

    messages = [{"role": "user", "content": prompt}]
    
    try:
        new_summary = send_to_llm(messages, temperature=0.3, max_tokens=500)
        return new_summary.strip()
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return old_summary

####################
# CHAT PROCESSING
####################

def build_context(summary: str, messages: List[Dict]) -> str:
    """Build conversation context from summary and history"""
    
    context_parts = []
    
    if summary:
        context_parts.append(f"Conversation Summary:\n{summary}\n")
    
    if messages:
        history = "\n".join([
            f"{m['role'].capitalize()}: {m['content']}" 
            for m in messages[-10:]
        ])
        context_parts.append(f"Recent History:\n{history}")
    
    return "\n".join(context_parts)

def process_chat_message(session_id: str, user_message: str) -> ChatResponse:
    """Process incoming chat message and generate response"""
    
    # Get session
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = session.get("messages", [])
    old_summary = get_summary(session_id)
    
    # Build context
    context = build_context(old_summary, messages)
    
    # Prepare LLM messages
    llm_messages = []
    
    if context:
        llm_messages.append({
            "role": "system",
            "content": f"You are a helpful AI assistant. Use this context from previous conversation:\n\n{context}"
        })
    
    llm_messages.append({
        "role": "user",
        "content": user_message
    })
    
    # Get AI response
    ai_response = send_to_llm(llm_messages)
    
    # Update messages
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
    
    # Save messages
    update_session(session_id, messages)
    
    # Update summary if needed
    new_summary = None
    if len(messages) >= SUMMARY_THRESHOLD:
        new_summary = generate_summary(old_summary, messages)
        update_summary(session_id, new_summary)
    
    return ChatResponse(
        session_id=session_id,
        response=ai_response,
        timestamp=timestamp,
        summary=new_summary
    )

####################
# FASTAPI APP
####################

app = FastAPI(
    title="AI Chat Server",
    description="Production-ready AI chat server with session management and memory",
    version="1.0.0"
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

####################
# HEALTH CHECK
####################

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        r.ping()
        redis_status = "ok"
    except:
        redis_status = "error"
    
    try:
        requests.get(f"http://{QWEN_HOST}:{QWEN_PORT}/health", timeout=2)
        llm_status = "ok"
    except:
        llm_status = "error"
    
    return {
        "status": "ok" if redis_status == "ok" and llm_status == "ok" else "degraded",
        "redis": redis_status,
        "llm": llm_status,
        "timestamp": datetime.utcnow().isoformat()
    }

####################
# CHAT ENDPOINTS
####################

@chat_router.post("/", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Send a message and get AI response"""
    try:
        return process_chat_message(req.session_id, req.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

####################
# SESSION ENDPOINTS
####################

@session_router.post("/create")
def create_new_session(req: SessionCreate = SessionCreate()):
    """Create a new chat session"""
    session_id = create_session(req.metadata)
    return {
        "session_id": session_id,
        "created_at": datetime.utcnow().isoformat()
    }

@session_router.get("/{session_id}")
def get_session_info(session_id: str):
    """Get session information"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@session_router.delete("/{session_id}")
def remove_session(session_id: str):
    """Delete a session"""
    if not delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted", "session_id": session_id}

@session_router.get("/")
def get_all_sessions(limit: int = 100):
    """List all sessions"""
    sessions = list_sessions(limit)
    return {"sessions": sessions, "count": len(sessions)}

####################
# SUMMARY ENDPOINTS
####################

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
    # Check if session exists
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
def regenerate_summary(session_id: str):
    """Regenerate summary from current messages"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = session.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="No messages to summarize")
    
    old_summary = get_summary(session_id)
    new_summary = generate_summary(old_summary, messages)
    update_summary(session_id, new_summary)
    
    return {
        "session_id": session_id,
        "summary": new_summary,
        "updated_at": datetime.utcnow().isoformat()
    }

####################
# INCLUDE ROUTERS
####################

app.include_router(chat_router)
app.include_router(session_router)
app.include_router(summary_router)

####################
# STARTUP/SHUTDOWN
####################

@app.on_event("startup")
async def startup_event():
    try:
        r.ping()
        logger.info("✓ Redis connection successful")
    except Exception as e:
        logger.error(f"✗ Redis connection failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down server...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)