"""
Chat processing logic - Part 1: Memory Management & Profile
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .redis_client import (
    safe_redis_get, safe_redis_set, safe_redis_hgetall, safe_redis_hset,
    safe_redis_delete, safe_redis_pipeline_execute, json_encode, json_decode,
    SESSION_PREFIX, SUMMARY_PREFIX, USER_PROFILE_PREFIX, DIALOG_STATE_PREFIX,
    SESSION_TTL
)
from .models import create_default_profile, create_default_dialog_state
from .utils import merge_profile_facts, extract_json_safe, get_representative_messages
from .llm import send_to_llm, MAX_SUMMARY_TOKENS

logger = logging.getLogger(__name__)

# Memory limits
MAX_HISTORY_LENGTH = 30
CONTEXT_WINDOW = 6
SUMMARY_THRESHOLD = 8
PROFILE_UPDATE_THRESHOLD = 5


##################### USER PROFILE ####################

def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get user profile
    Returns None if profile doesn't exist
    """
    key = f"{USER_PROFILE_PREFIX}{user_id}"
    data = safe_redis_get(key)
    
    if not data:
        return None
    
    return json_decode(data)


def create_default_user_profile(user_id: str) -> Dict[str, Any]:
    """
    Explicit profile creation
    Called only when client wants to create new user
    """
    default_profile = create_default_profile()
    
    if update_user_profile(user_id, default_profile):
        logger.info(f"✓ Created default profile for user: {user_id}")
        return default_profile
    else:
        logger.error(f"✗ Failed to create profile for {user_id}")
        return default_profile


def update_user_profile(user_id: str, profile_data: Dict[str, Any]) -> bool:
    """Update user profile"""
    key = f"{USER_PROFILE_PREFIX}{user_id}"
    success = safe_redis_set(key, json_encode(profile_data), ex=SESSION_TTL * 4)
    
    if success:
        logger.info(f"✓ Profile updated: {user_id}")
    else:
        logger.error(f"✗ Profile update failed: {user_id}")
    
    return success


def delete_user_profile(user_id: str) -> bool:
    """Delete user profile"""
    key = f"{USER_PROFILE_PREFIX}{user_id}"
    return safe_redis_delete(key)


##################### DIALOG STATE ####################

def get_dialog_state(user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
    """
    Get dialog state
    Returns None if state doesn't exist
    """
    key = f"{DIALOG_STATE_PREFIX}{user_id}:{session_id}"
    data = safe_redis_get(key)
    
    if not data:
        return None
    
    return json_decode(data)


def create_default_dialog_state_for_session(user_id: str, session_id: str) -> Dict[str, Any]:
    """Explicit state creation"""
    default_state = create_default_dialog_state()
    
    if update_dialog_state(user_id, session_id, default_state):
        logger.info(f"✓ Created default dialog state: {user_id}:{session_id}")
        return default_state
    else:
        logger.error(f"✗ Failed to create dialog state")
        return default_state


def update_dialog_state(user_id: str, session_id: str, state: Dict[str, Any]) -> bool:
    """Update dialog state"""
    key = f"{DIALOG_STATE_PREFIX}{user_id}:{session_id}"
    state["last_updated"] = datetime.utcnow().isoformat()
    
    success = safe_redis_set(key, json_encode(state), ex=SESSION_TTL)
    
    if success:
        logger.info(f"✓ Dialog state updated: {state.get('current_goal', 'unknown')}")
    else:
        logger.error(f"✗ Dialog state update failed")
    
    return success


##################### SESSION ####################

def create_session(user_id: str, metadata: Optional[Dict] = None) -> str:
    """Create new session"""
    from .utils import generate_id
    
    session_id = generate_id()
    key = f"{SESSION_PREFIX}{user_id}:{session_id}"
    
    success = safe_redis_hset(key, mapping={
        "messages": json_encode([]),
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "metadata": json_encode(metadata or {}),
        "message_count": "0",
        "last_profile_check": "0"
    }, ex=SESSION_TTL)
    
    if success:
        logger.info(f"✓ Session created: {user_id}:{session_id}")
    else:
        logger.error(f"✗ Session creation failed: {user_id}:{session_id}")
    
    return session_id


def get_session(user_id: str, session_id: str) -> Optional[Dict]:
    """Get session data"""
    key = f"{SESSION_PREFIX}{user_id}:{session_id}"
    data = safe_redis_hgetall(key)
    
    if not data or not data.get("messages"):
        return None
    
    try:
        return {
            "user_id": user_id,
            "session_id": session_id,
            "messages": json_decode(data.get("messages", "[]")),
            "created_at": data.get("created_at", ""),
            "updated_at": data.get("updated_at", ""),
            "metadata": json_decode(data.get("metadata", "{}")),
            "message_count": int(data.get("message_count", 0)),
            "last_profile_check": int(data.get("last_profile_check", 0))
        }
    except (ValueError, TypeError) as e:
        logger.error(f"Failed to parse session data: {e}")
        return None


def update_session(user_id: str, session_id: str, messages: List[Dict], last_profile_check: Optional[int] = None) -> bool:
    """Update session messages"""
    key = f"{SESSION_PREFIX}{user_id}:{session_id}"
    
    # Check if session exists
    try:
        from .redis_client import r
        if not r.exists(key):
            return False
    except:
        return False

    if len(messages) > MAX_HISTORY_LENGTH:
        messages = messages[-MAX_HISTORY_LENGTH:]
    
    update_data = {
        "messages": json_encode(messages),
        "updated_at": datetime.utcnow().isoformat(),
        "message_count": str(len(messages))
    }
    
    if last_profile_check is not None:
        update_data["last_profile_check"] = str(last_profile_check)
    
    success = safe_redis_hset(key, mapping=update_data, ex=SESSION_TTL)
    return success


def delete_session(user_id: str, session_id: str) -> bool:
    """Delete session and all associated data"""
    session_key = f"{SESSION_PREFIX}{user_id}:{session_id}"
    summary_key = f"{SUMMARY_PREFIX}{user_id}:{session_id}"
    state_key = f"{DIALOG_STATE_PREFIX}{user_id}:{session_id}"
    
    from .redis_client import RATE_LIMIT_PREFIX
    rate_key = f"{RATE_LIMIT_PREFIX}{user_id}:{session_id}"
    
    # Delete all keys
    deleted = all([
        safe_redis_delete(session_key),
        safe_redis_delete(summary_key),
        safe_redis_delete(state_key),
        safe_redis_delete(rate_key)
    ])
    
    logger.info(f"✓ Session deleted: {user_id}:{session_id}")
    return deleted


##################### SUMMARY ####################

def get_summary(user_id: str, session_id: str) -> str:
    """Get conversation summary"""
    key = f"{SUMMARY_PREFIX}{user_id}:{session_id}"
    data = safe_redis_get(key, default="")
    return data


def update_summary(user_id: str, session_id: str, summary: str) -> bool:
    """Update summary"""
    key = f"{SUMMARY_PREFIX}{user_id}:{session_id}"
    success = safe_redis_set(key, summary, ex=SESSION_TTL)
    
    if success:
        logger.info(f"✓ Summary updated: {user_id}:{session_id}")
    else:
        logger.error(f"✗ Summary update failed")
    
    return success


async def generate_summary(old_summary: str, messages: List[Dict]) -> str:
    """Generate conversation summary"""
    representative = get_representative_messages(messages, count=6)
    
    recent_text = "\n".join([
        f"{m['role'].capitalize()}: {m['content'][:150]}"
        for m in representative
    ])
    
    prompt = f"""Analyze dialog state from {len(messages)} messages.

Previous summary: {old_summary if old_summary else "New conversation"}

Representative messages:
{recent_text}

Create brief summary (max 80 words):
1. Current topic/goal?
2. Target result?
3. What's decided/solved?

FORBIDDEN:
- Full history
- Old topics
- Personal details

Focus on CURRENT STATE."""

    llm_messages = [{"role": "user", "content": prompt}]
    
    try:
        result = await send_to_llm(llm_messages, temperature=0.3, max_tokens=MAX_SUMMARY_TOKENS)
        return result["content"].strip()
    except Exception as e:
        logger.error(f"✗ Summary generation failed: {e}")
        return old_summary