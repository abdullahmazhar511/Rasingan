"""PHQ-9 (Patient Health Questionnaire-9) — depression screening checklist.

Source: Spitzer, Williams, Kroenke, and colleagues.
The PHQ-9 asks how often, over the last 2 weeks, the patient has been bothered
by each problem. Each item is rated 0 (Not at all) to 3 (Nearly every day).
"""

CHECKLIST = {
    "core_mood": {
        "name": "PHQ-9: Core Mood",
        "description": "Cardinal depression symptoms (last 2 weeks)",
        "items": [
            "Little interest or pleasure in doing things (anhedonia)",
            "Feeling down, depressed, or hopeless",
        ],
    },
    "somatic": {
        "name": "PHQ-9: Somatic Symptoms",
        "description": "Physical/neurovegetative symptoms",
        "items": [
            "Trouble falling or staying asleep, or sleeping too much",
            "Feeling tired or having little energy",
            "Poor appetite or overeating",
            "Moving or speaking slowly, or being fidgety/restless (psychomotor change)",
        ],
    },
    "cognitive_selfworth": {
        "name": "PHQ-9: Cognitive & Self-Worth",
        "items": [
            "Feeling bad about yourself, a failure, or having let family down",
            "Trouble concentrating on things (e.g., reading, watching TV)",
        ],
    },
    "safety": {
        "name": "PHQ-9: Safety / Suicidality",
        "description": "Critical safety screen — always assess",
        "items": [
            "Thoughts that you would be better off dead, or of hurting yourself",
        ],
    },
    "functional_impact": {
        "name": "PHQ-9: Functional Impact",
        "items": [
            "Difficulty doing work, home tasks, or getting along with others due to these problems",
        ],
    },
}

PRIORITY_ORDER = [
    "safety",
    "core_mood",
    "somatic",
    "cognitive_selfworth",
    "functional_impact",
]

SCORING = {
    "frequency_options": [
        "Not at all (0)",
        "Several days (1)",
        "More than half the days (2)",
        "Nearly every day (3)",
    ],
    "total_range": (0, 27),
    "interpretation": [
        ((0, 4), "Minimal depression"),
        ((5, 9), "Mild depression"),
        ((10, 14), "Moderate depression"),
        ((15, 19), "Moderately severe depression"),
        ((20, 27), "Severe depression"),
    ],
}
