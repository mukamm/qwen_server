"""
Chat processing logic - Part 2: Intent Extraction & Main Processor
"""
import logging
from typing import Dict, Any, List
from datetime import datetime
from fastapi import HTTPException

from .redis_client import (
    check_rate_limit, safe_redis_pipeline_execute, json_decode,
    SESSION_PREFIX, USER_PROFILE_PREFIX, DIALOG_STATE_PREFIX, SUMMARY_PREFIX
)
from .models import create_chat_response
from .utils import (
    extract_json_safe, should_extract_intent, validate_response,
    build_system_prompt, build_response_rules
)
from .llm import send_to_llm
from .chat_part1 import (
    get_user_profile, create_default_user_profile, update_user_profile,
    get_dialog_state, create_default_dialog_state_for_session, update_dialog_state,
    get_session, update_session, get_summary, update_summary, generate_summary,
    merge_profile_facts, SUMMARY_THRESHOLD, PROFILE_UPDATE_THRESHOLD
)

logger = logging.getLogger(__name__)

# Rate limits
RATE_LIMIT_REQUESTS = 1
RATE_LIMIT_WINDOW = 3


##################### PROFILE EXTRACTION ####################

async def detect_profile_changes(messages: List[Dict]) -> bool:
    """
    Fast detector for profile changes
    Returns True if user mentioned personal facts
    """
    # Take last 3 user messages
    user_messages = [m for m in messages if m.get("role") == "user"][-3:]
    
    if not user_messages:
        return False
    
    text = " ".join([m["content"].lower() for m in user_messages])
    
    # Simple keyword detection first
    personal_keywords = [
        "my name", "i am", "i'm", "i work", "i study", 
        "i know", "i love", "i want to learn", "my job",
        "years old", "developer", "engineer", "student"
    ]
    
    if any(kw in text for kw in personal_keywords):
        logger.info("✓ Personal facts detected in messages")
        return True
    
    return False


async def extract_user_facts(messages: List[Dict], current_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simplified profile extraction
    - Shorter prompt
    - Only extracts NEW facts
    - Better error handling
    """
    # Take only last 5 user messages
    user_messages = [m for m in messages if m.get("role") == "user"][-5:]
    
    if not user_messages:
        return {}
    
    conversation_text = "\n".join([
        f"User: {m['content'][:100]}"  # Max 100 chars per message
        for m in user_messages
    ])
    
    # Show only filled fields from current profile
    current_facts = {k: v for k, v in current_profile.items() if v}
    
    from .redis_client import json_encode
    current_facts_str = json_encode(current_facts).decode('utf-8') if isinstance(json_encode(current_facts), bytes) else str(current_facts)
    
    # SIMPLIFIED PROMPT
    prompt = f"""Extract NEW user facts from conversation.

CURRENT KNOWN: {current_facts_str}

RECENT MESSAGES:
{conversation_text}

Return JSON with ONLY NEW facts:
{{
  "name": "string or null",
  "age": number or null,
  "role": "job title",
  "level": "beginner/junior/middle/senior/expert",
  "tech_stack": ["tech1"],
  "interests": ["interest1"],
  "language": "en/ru",
  "learning_goals": ["goal1"]
}}

RULES:
- Return ONLY NEW/UPDATED fields
- Lists = only NEW items
- If "My name is X" → {{"name": "X"}}
- If "I know Python" → {{"tech_stack": ["Python"]}}
- Return valid JSON only

Example: "Hi, I'm Batyr. I want to learn FastAPI"
Return: {{"name": "Batyr", "learning_goals": ["learn FastAPI"]}}"""

    llm_messages = [{"role": "user", "content": prompt}]
    
    try:
        result = await send_to_llm(llm_messages, temperature=0.1, max_tokens=200)
        response_text = result["content"].strip()
        
        # Safe JSON extraction
        extracted_facts = extract_json_safe(response_text)
        
        if extracted_facts:
            logger.info(f"✓ Extracted facts: {list(extracted_facts.keys())}")
            return extracted_facts
        else:
            logger.warning("✗ No valid JSON in profile extraction")
            return {}
            
    except Exception as e:
        logger.error(f"✗ Profile extraction failed: {e}")
        return {}


##################### INTENT EXTRACTION ####################

async def extract_intent(user_message: str, current_state: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    """Simplified intent extraction"""
    current_goal = current_state.get("current_goal", "unknown")
    
    # SHORTENED PROMPT
    prompt = f"""Extract user intent.

USER LEVEL: {profile.get('level', 'junior')}
CURRENT GOAL: {current_goal}

MESSAGE: "{user_message[:200]}"

Return JSON:
{{
  "current_goal": "brief goal description",
  "mode": "learn|debug|inspect|design|quick",
  "detail_level": "brief|normal|detailed",
  "understood_concepts": ["concept1"],
  "forbidden_topics": ["basics"]
}}

RULES:
- "I know X" → understood_concepts, forbidden_topics
- "how to debug" → mode: "debug"
- "explain" → mode: "learn"
- "briefly" → detail_level: "brief"
- Valid JSON only"""

    llm_messages = [{"role": "user", "content": prompt}]
    
    try:
        result = await send_to_llm(llm_messages, temperature=0.1, max_tokens=150)
        response = result["content"].strip()
        
        # Safe JSON extraction
        intent = extract_json_safe(response)
        
        if intent:
            logger.info(f"✓ Intent: {intent.get('mode')} - {intent.get('current_goal', 'unknown')[:50]}")
            return intent
        
        logger.warning("✗ No valid JSON in intent extraction")
        return current_state
        
    except Exception as e:
        logger.error(f"✗ Intent extraction failed: {e}")
        return current_state


##################### MAIN CHAT PROCESSOR ####################

async def process_chat_message(user_id: str, session_id: str, user_message: str) -> Dict[str, Any]:
    """
    Optimized main processor
    
    Changes:
    - Batch Redis operations (pipeline)
    - Conditional intent extraction
    - Conditional profile extraction
    - Better error handling
    """
    
    # Rate limit check
    if not check_rate_limit(user_id, session_id, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit: max {RATE_LIMIT_REQUESTS} req per {RATE_LIMIT_WINDOW}s"
        )
    
    # STEP 1: BATCH LOAD (Redis pipeline)
    session_key = f"{SESSION_PREFIX}{user_id}:{session_id}"
    profile_key = f"{USER_PROFILE_PREFIX}{user_id}"
    state_key = f"{DIALOG_STATE_PREFIX}{user_id}:{session_id}"
    summary_key = f"{SUMMARY_PREFIX}{user_id}:{session_id}"
    
    # Execute all reads in one pipeline
    results = safe_redis_pipeline_execute([
        ('hgetall', session_key),
        ('get', profile_key),
        ('get', state_key),
        ('get', summary_key)
    ])
    
    session_data, profile_data, state_data, summary = results
    
    # Parse session
    if not session_data or not session_data.get("messages"):
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session = {
            "user_id": user_id,
            "session_id": session_id,
            "messages": json_decode(session_data.get("messages", "[]")),
            "message_count": int(session_data.get("message_count", 0)),
            "last_profile_check": int(session_data.get("last_profile_check", 0))
        }
    except (ValueError, TypeError) as e:
        logger.error(f"Failed to parse session: {e}")
        raise HTTPException(status_code=500, detail="Session data corrupted")
    
    messages = session["messages"]
    message_count = session["message_count"]
    last_profile_check = session["last_profile_check"]
    
    # Parse profile
    if profile_data:
        profile = json_decode(profile_data)
        if not profile:
            profile = create_default_user_profile(user_id)
    else:
        profile = create_default_user_profile(user_id)
    
    # Parse state
    if state_data:
        state = json_decode(state_data)
        if not state:
            state = create_default_dialog_state_for_session(user_id, session_id)
    else:
        state = create_default_dialog_state_for_session(user_id, session_id)
    
    if not summary:
        summary = ""
    
    # STEP 2: CONDITIONAL INTENT EXTRACTION
    is_first_message = message_count == 0
    
    if should_extract_intent(user_message, state, is_first_message):
        logger.info("🎯 Extracting intent...")
        new_state = await extract_intent(user_message, state, profile)
        update_dialog_state(user_id, session_id, new_state)
        state = new_state
    else:
        logger.info("⏭️  Skipping intent extraction (not needed)")
    
    # STEP 3: CONDITIONAL PROFILE EXTRACTION
    profile_updated = False
    
    # Check every 5 messages OR first 3 exchanges
    should_check_profile = (
        message_count < 6 or  # First 3 exchanges
        message_count - last_profile_check >= PROFILE_UPDATE_THRESHOLD
    )
    
    if should_check_profile:
        # Add current message to temp history
        temp_messages = messages + [{"role": "user", "content": user_message}]
        
        # Fast detection first
        if await detect_profile_changes(temp_messages):
            logger.info(f"🧠 Extracting profile facts (message #{message_count})...")
            extracted_facts = await extract_user_facts(temp_messages, profile)
            
            if extracted_facts:
                updated_profile = merge_profile_facts(profile, extracted_facts)
                update_user_profile(user_id, updated_profile)
                profile = updated_profile
                profile_updated = True
                logger.info(f"✓ Profile updated: {list(extracted_facts.keys())}")
        else:
            logger.info("⏭️  No profile changes detected")
        
        last_profile_check = message_count
    
    # STEP 4: BUILD MINIMAL PROMPT
    logger.info("🧠 Building minimal prompt...")
    system_content = build_system_prompt(profile, state, summary, messages)
    
    # Get rules for metadata
    rules = build_response_rules(profile, state)
    
    # STEP 5: GET LLM RESPONSE
    llm_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_message}
    ]
    
    result = await send_to_llm(llm_messages)
    ai_response = result["content"]
    
    # STEP 6: VALIDATE RESPONSE
    previous_responses = [m["content"] for m in messages if m.get("role") == "assistant"]
    
    if not validate_response(ai_response, state, previous_responses):
        logger.warning("⚠️  Response rejected, retrying with higher temperature...")
        
        # Add strict rule
        retry_rules = rules + ["CRITICAL: Previous response rejected for repetition. Provide COMPLETELY DIFFERENT answer."]
        system_content = build_system_prompt(profile, state, summary, messages)
        llm_messages[0]["content"] = system_content
        
        result = await send_to_llm(llm_messages, temperature=0.9)
        ai_response = result["content"]
    
    # STEP 7: SAVE MESSAGES
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
    
    update_session(user_id, session_id, messages, last_profile_check=last_profile_check)
    
    # STEP 8: UPDATE SUMMARY IF NEEDED
    new_summary = None
    if len(messages) >= SUMMARY_THRESHOLD:
        logger.info("📝 Updating summary...")
        new_summary = await generate_summary(summary, messages)
        update_summary(user_id, session_id, new_summary)
    
    return create_chat_response(
        user_id=user_id,
        session_id=session_id,
        response=ai_response,
        intent=state,
        summary=new_summary,
        profile_updated=profile_updated,
        tokens_used=result.get("tokens_used"),
        rules_applied=rules[:5]
    )