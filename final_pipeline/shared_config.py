"""
Shared configuration for the therapeutic pipeline.
Single source of truth for: extraction checklist, system prompts, session phases,
and checklist tracking helpers.

Used by both the standalone inference pipeline (agent_1, agent_2, patient, pipeline)
and the VERL training agent loop (therapist_agent_loop.py).
"""

from typing import Dict, List, Optional


# ========== Extraction Checklist ==========

EXTRACTION_CHECKLIST = {
    "chief_complaint": {
        "name": "Chief Complaint",
        "description": "What brought the patient in today",
        "items": [
            "Primary concern or reason for visit",
            "When the issue started / how long it's been going on",
            "What triggered or worsened it recently",
        ]
    },
    "symptom_assessment": {
        "name": "Symptom Assessment",
        "items": [
            "Emotional symptoms (mood, anxiety, irritability, sadness)",
            "Physical symptoms (sleep, appetite, energy, aches)",
            "Cognitive symptoms (concentration, racing thoughts, memory)",
            "Behavioral changes (withdrawal, avoidance, substance use)",
        ]
    },
    "functional_impact": {
        "name": "Functional Impact",
        "items": [
            "Impact on work or school performance",
            "Impact on relationships (family, friends, partner)",
            "Impact on daily routines and self-care",
        ]
    },
    "history_context": {
        "name": "History & Context",
        "items": [
            "Previous therapy or treatment experience",
            "Family history of mental health issues",
            "Current medications or substances",
            "Major life events or stressors",
        ]
    },
    "safety_screening": {
        "name": "Safety Screening",
        "items": [
            "Suicidal ideation (thoughts of self-harm)",
            "Homicidal ideation (thoughts of harming others)",
            "Current safety and living situation",
            "Access to means of harm",
        ]
    },
    "support_system": {
        "name": "Support System",
        "items": [
            "Who the patient can turn to for support",
            "Quality of close relationships",
            "Community or social connections",
        ]
    },
    "coping_strategies": {
        "name": "Coping Strategies",
        "items": [
            "What the patient has tried so far to cope",
            "What has helped / not helped",
            "Patient's strengths and resources",
        ]
    },
    "goals_expectations": {
        "name": "Goals & Expectations",
        "items": [
            "What the patient hopes to get from therapy",
            "Short-term goals (next 1-2 weeks)",
            "Longer-term vision of improvement",
        ]
    },
}

EXTRACTION_PRIORITY_ORDER = [
    "safety_screening", "chief_complaint", "symptom_assessment",
    "functional_impact", "history_context", "support_system",
    "coping_strategies", "goals_expectations"
]


# ========== System Prompts ==========

THERAPIST_PHASE_GUIDANCE = {
    "opening": """- Welcome the patient warmly and create psychological safety
- Explain confidentiality and session boundaries
- Ask open-ended question about what brings them today
- Avoid giving advice in this phase""",

    "assessment": """- Ask open-ended questions to understand the full situation
- Explore the history, triggers, and impact on daily life
- Ask about physical symptoms, emotional patterns, relationships
- Assess for safety concerns without being alarmist
- Use reflective listening: "It sounds like..." """,

    "intervening": """- Validate the patient's emotions and experiences
- Help them identify patterns and connections
- Explore coping strategies they've tried
- Suggest evidence-based techniques if appropriate
- Encourage problem-solving and agency""",

    "closing": """- Summarize key insights from the session
- Identify concrete action steps or homework
- Schedule follow-up if needed
- Reaffirm support and belief in their ability to cope"""
}


def get_therapist_system_prompt(session_phase: str, session_type: str = "counseling") -> str:
    """Generate system prompt for the therapist based on session phase."""
    guidance = THERAPIST_PHASE_GUIDANCE.get(session_phase, THERAPIST_PHASE_GUIDANCE["assessment"])
    return f"""You are a compassionate, skilled therapist/counselor conducting a {session_type} session.

THERAPEUTIC APPROACH:
- Use Rogerian person-centered approach: empathy, unconditional positive regard, genuineness
- Be curious and non-judgmental
- Reflect and validate emotions
- Help the patient find their own solutions

CURRENT SESSION PHASE: {session_phase}
{guidance}

GENERAL GUIDELINES:
- Ask ONE clear, open-ended question per response (not multiple)
- Use reflective listening techniques
- Acknowledge feelings before moving to problem-solving
- Never dismiss or minimize the patient's concerns
- Be warm, genuine, and present
- Keep responses concise but meaningful (2-3 sentences)
- Avoid jargon; use simple, clear language
"""


def get_patient_system_prompt(patient_context: str, patient_profile: dict) -> str:
    """Generate system prompt for the patient based on context."""
    name = patient_profile.get("name", "Patient")
    age = patient_profile.get("age", "unknown")
    occupation = patient_profile.get("occupation", "not specified")
    family_structure = patient_profile.get("family_structure", "not specified")

    return f"""You are a patient in therapy. You are experiencing the following situation:

PATIENT SITUATION:
{patient_context}

PATIENT PROFILE:
Name: {name}
Age: {age}
Occupation: {occupation}
Family Structure: {family_structure}

INSTRUCTIONS:
- Respond authentically as someone experiencing this situation
- Express genuine emotions, concerns, and worries
- Be natural and conversational - not overly formal
- Share relevant details from your experience when asked
- Show vulnerability and openness as appropriate
- Maintain consistency with the described situation throughout conversation
"""


def get_supervisor_system_prompt() -> str:
    """Generate system prompt for the supervisor (agent_2)."""
    return """You are an experienced senior therapist/clinical supervisor reviewing a therapy session.
Your job is to evaluate whether the therapist is making progress on the information extraction checklist.
Provide constructive, actionable feedback. Focus on coaching, not criticism.
Tell the therapist specifically what to ask about next based on the pending checklist items.
"""


# ========== Session Phase Logic ==========

def get_session_phase(turn: int, max_turns: int) -> str:
    """Determine therapy session phase based on turn number."""
    if turn <= 1:
        return "opening"
    elif turn <= max_turns - 2:
        return "assessment"
    elif turn <= max_turns - 1:
        return "intervening"
    else:
        return "closing"


SESSION_ENDING_INDICATORS = [
    "see you next time", "take care", "good luck",
    "until next week", "same time next", "goodbye", "farewell"
]


def should_end_session(last_response: str) -> bool:
    """Check if session should naturally end based on the response text."""
    return any(ind in last_response.lower() for ind in SESSION_ENDING_INDICATORS)


# ========== Extraction Checklist Helpers ==========

def init_extraction_status() -> Dict:
    """Initialize extraction status — all items start as not covered."""
    status = {}
    for category, data in EXTRACTION_CHECKLIST.items():
        status[category] = {
            "name": data["name"],
            "items": {item: {"covered": False, "turn_covered": None, "evidence": None}
                      for item in data["items"]}
        }
    return status


def compute_extraction_summary(extraction_status: Dict) -> Dict:
    """Compute how much of the extraction checklist has been covered."""
    total_items = 0
    covered_items = 0
    category_coverage = {}

    for category, data in extraction_status.items():
        cat_total = len(data["items"])
        cat_covered = sum(1 for v in data["items"].values() if v["covered"])
        total_items += cat_total
        covered_items += cat_covered
        category_coverage[category] = {
            "name": data["name"],
            "covered": cat_covered,
            "total": cat_total,
            "pct": round(cat_covered / cat_total * 100, 1) if cat_total else 0
        }

    return {
        "total_items": total_items,
        "covered_items": covered_items,
        "overall_pct": round(covered_items / total_items * 100, 1) if total_items else 0,
        "categories": category_coverage,
        "pending_categories": [
            cat["name"] for cat in category_coverage.values()
            if cat["covered"] < cat["total"]
        ]
    }


def format_extraction_status(extraction_status: Dict) -> str:
    """Format extraction checklist showing covered vs pending items."""
    text = ""
    for category, data in extraction_status.items():
        covered_count = sum(1 for v in data["items"].values() if v["covered"])
        total = len(data["items"])
        text += f"\n{data['name']} [{covered_count}/{total}]:\n"
        for item, status in data["items"].items():
            mark = "✓" if status["covered"] else "☐"
            text += f"  {mark} {item}\n"
    return text


def get_pending_items(extraction_status: Dict) -> List[str]:
    """Get a flat list of extraction items still not covered."""
    pending = []
    for category, data in extraction_status.items():
        for item, status in data["items"].items():
            if not status["covered"]:
                pending.append(f"[{data['name']}] {item}")
    return pending


def get_next_priority(extraction_status: Dict) -> Optional[str]:
    """Suggest which extraction area the therapist should focus on next."""
    for category in EXTRACTION_PRIORITY_ORDER:
        data = extraction_status.get(category, {})
        items = data.get("items", {})
        pending = [item for item, status in items.items() if not status["covered"]]
        if pending:
            return f"[{data['name']}] Focus on: {pending[0]}"
    return None
