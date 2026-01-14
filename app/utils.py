import uuid
import orjson
import logging
from typing import Dict, Any, Optional, List
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


def generate_id() -> str:
    """Generate unique ID"""
    return str(uuid.uuid4())


def extract_json_safe(text: str) -> Optional[Dict]:
    """
    Safely extract JSON from LLM response
    Uses orjson decoder
    """
    try:
        # Find first opening brace
        start = text.find('{')
        if start == -1:
            logger.warning("No JSON object found in text")
            return None
        
        # Parse from that position
        obj = orjson.loads(text[start:])
        
        # Validate it's a dict
        if not isinstance(obj, dict):
            logger.warning("Parsed JSON is not a dictionary")
            return None
            
        return obj
    except orjson.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in JSON extraction: {e}")
        return None


def merge_profile_facts(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge new facts into existing profile"""
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


def get_representative_messages(messages: List[Dict], count: int = 6) -> List[Dict]:
    """
    Get representative sample from conversation
    Takes first, middle, and last messages for better context
    """
    if len(messages) <= count:
        return messages
    
    # Calculate distribution
    first_count = count // 3
    middle_count = count // 3
    last_count = count - first_count - middle_count
    
    # Get samples
    first = messages[:first_count]
    middle_idx = len(messages) // 2
    middle = messages[middle_idx - middle_count//2 : middle_idx + middle_count//2]
    last = messages[-last_count:]
    
    return first + middle + last


def validate_response(response: str, state: Dict[str, Any], previous_responses: List[str]) -> bool:
    """
    Validate response quality
    Checks order and structure, not just word overlap
    """
    # 1. Length check
    if len(response) < 10:
        logger.warning("✗ Response too short")
        return False
    
    # 2. Check similarity with previous responses (only last 2)
    response_clean = response.lower().strip()
    
    for prev in previous_responses[-2:]:
        prev_clean = prev.lower().strip()
        
        # Use SequenceMatcher (considers order)
        ratio = SequenceMatcher(None, response_clean, prev_clean).ratio()
        
        if ratio > 0.7:  # 70% similar
            logger.warning(f"✗ Response too similar to previous (ratio: {ratio:.2f})")
            return False
    
    # 3. Check forbidden topics
    forbidden = state.get("forbidden_topics", [])
    if forbidden:
        response_lower = response.lower()
        for topic in forbidden[:3]:  # Check only top 3
            if topic.lower() in response_lower:
                logger.warning(f"✗ Response contains forbidden topic: {topic}")
                return False
    
    return True


def build_response_rules(profile: Dict[str, Any], state: Dict[str, Any]) -> List[str]:
    """
    Build response rules (max 5)
    LLM can't handle too many rules effectively
    """
    rules = []
    
    level = profile.get("level", "junior")
    mode = state.get("mode", "learn")
    detail_level = state.get("detail_level", "normal")
    understood = state.get("understood_concepts", [])
    
    # Rule 1: Level-based
    if level == "beginner":
        rules.append("Explain simply, use basic terms")
    elif level in ["senior", "expert"]:
        rules.append("Skip basics, use technical terms")
    
    # Rule 2: Mode-based
    if mode == "debug":
        rules.append("Focus on solution, show corrected code")
    elif mode == "quick":
        rules.append("Answer in 1-2 sentences max")
    
    # Rule 3: Detail level
    if detail_level == "brief":
        rules.append("Keep under 100 words, essentials only")
    elif detail_level == "detailed":
        rules.append("Thorough explanation with examples")
    
    # Rule 4: Understood concepts (only top 3)
    if understood:
        top_understood = ', '.join(understood[:3])
        rules.append(f"DO NOT explain (user knows): {top_understood}")
    
    # Rule 5: General
    rules.append("Stay on topic, no repetition")
    
    return rules[:5]  # HARD LIMIT: 5 rules


def build_system_prompt(profile: Dict[str, Any], state: Dict[str, Any], summary: str, messages: List[Dict]) -> str:
    """
    Build minimal system prompt (~200 tokens instead of 1000+)
    
    Only includes:
    - Role + level (1 line)
    - Current goal (1 line)
    - Top 3 rules
    - Summary only if conversation is long
    """
    parts = []
    
    # 1. Brief role (1 line)
    role = profile.get('role', 'user')
    level = profile.get('level', 'junior')
    parts.append(f"You help {role}s (level: {level}).")
    
    # 2. Current goal (if exists)
    if state.get("current_goal"):
        parts.append(f"Goal: {state['current_goal'][:80]}")
    
    # 3. TOP 3 RULES ONLY
    rules = build_response_rules(profile, state)
    if rules:
        parts.append("Rules: " + " | ".join(rules[:3]))
    
    # 4. Summary ONLY if >10 messages
    if len(messages) > 10 and summary:
        parts.append(f"Context: {summary[:100]}")
    
    # Total: ~150-250 tokens (vs 1000+ before)
    return "\n".join(parts)


def should_extract_intent(message: str, state: Optional[Dict[str, Any]], is_first: bool) -> bool:
    """
    Smart decision - do we need to extract intent?
    
    Extract only if:
    1. First message in session
    2. User explicitly states new goal
    3. Topic changed significantly
    """
    if is_first:
        return True
    
    # Check for explicit goal keywords
    goal_keywords = [
        "want to", "need to", "help me", "how to", "how do i",
        "explain", "show me", "can you", "i'm trying to",
        "working on", "building", "creating", "learning"
    ]
    
    message_lower = message.lower()
    
    if any(kw in message_lower for kw in goal_keywords):
        logger.info(f"✓ Goal keyword detected: {message[:50]}...")
        return True
    
    # Skip for short/casual messages
    if len(message.split()) < 5:
        return False
    
    return False