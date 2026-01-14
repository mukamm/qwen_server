"""
Data models using plain dicts instead of Pydantic
Includes validation functions
"""
from typing import Dict, Any, Optional, List
from datetime import datetime


# Constants
USER_LEVELS = ["beginner", "junior", "middle", "senior", "expert"]
RESPONSE_MODES = ["learn", "debug", "inspect", "design", "quick"]


def validate_chat_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate chat request"""
    required = ["user_id", "session_id", "message"]
    for field in required:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    if not isinstance(data["message"], str) or not data["message"].strip():
        raise ValueError("Message must be a non-empty string")
    
    return {
        "user_id": str(data["user_id"]),
        "session_id": str(data["session_id"]),
        "message": data["message"].strip()
    }


def validate_session_create(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate session create request"""
    if "user_id" not in data:
        raise ValueError("Missing required field: user_id")
    
    return {
        "user_id": str(data["user_id"]),
        "metadata": data.get("metadata", {})
    }


def validate_profile_update(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate profile update request"""
    if "user_id" not in data:
        raise ValueError("Missing required field: user_id")
    
    if "profile_data" not in data:
        raise ValueError("Missing required field: profile_data")
    
    profile = data["profile_data"]
    
    # Validate level if provided
    if "level" in profile and profile["level"] not in USER_LEVELS:
        raise ValueError(f"Invalid level. Must be one of: {USER_LEVELS}")
    
    # Validate lists
    for field in ["tech_stack", "interests", "learning_goals"]:
        if field in profile and not isinstance(profile[field], list):
            raise ValueError(f"{field} must be a list")
    
    return {
        "user_id": str(data["user_id"]),
        "profile_data": profile
    }


def create_chat_response(
    user_id: str,
    session_id: str,
    response: str,
    intent: Optional[Dict[str, Any]] = None,
    summary: Optional[str] = None,
    profile_updated: Optional[bool] = None,
    tokens_used: Optional[int] = None,
    rules_applied: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create chat response dict"""
    return {
        "user_id": user_id,
        "session_id": session_id,
        "response": response,
        "timestamp": datetime.utcnow().isoformat(),
        "intent": intent,
        "summary": summary,
        "profile_updated": profile_updated,
        "tokens_used": tokens_used,
        "rules_applied": rules_applied
    }


def create_default_profile() -> Dict[str, Any]:
    """Create default user profile"""
    return {
        "name": None,
        "age": None,
        "role": None,
        "level": "junior",
        "tech_stack": [],
        "language": "en",
        "interests": [],
        "learning_goals": []
    }


def create_default_dialog_state() -> Dict[str, Any]:
    """Create default dialog state"""
    return {
        "current_goal": None,
        "mode": "learn",
        "detail_level": "normal",
        "understood_concepts": [],
        "forbidden_topics": [],
        "context_type": None,
        "last_updated": datetime.utcnow().isoformat()
    }


def create_session_dict(
    user_id: str,
    session_id: str,
    messages: List[Dict],
    created_at: str,
    updated_at: str,
    metadata: Dict,
    message_count: int,
    last_profile_check: int
) -> Dict[str, Any]:
    """Create session dict"""
    return {
        "user_id": user_id,
        "session_id": session_id,
        "messages": messages,
        "created_at": created_at,
        "updated_at": updated_at,
        "metadata": metadata,
        "message_count": message_count,
        "last_profile_check": last_profile_check
    }