"""GAD-7 (Generalized Anxiety Disorder-7) — anxiety screening checklist.

Source: Spitzer, R. L., Kroenke, K., Williams, J. B. W., & Löwe, B. (2006).
The GAD-7 asks how often, over the last 2 weeks, the patient has been bothered
by each problem. Each item is rated 0 (Not at all) to 3 (Nearly every day).
"""

CHECKLIST = {
    "core_anxiety": {
        "name": "GAD-7: Core Anxiety / Worry",
        "description": "Cardinal worry and anxiety symptoms (last 2 weeks)",
        "items": [
            "Feeling nervous, anxious, or on edge",
            "Not being able to stop or control worrying",
            "Worrying too much about different things",
        ],
    },
    "physical_tension": {
        "name": "GAD-7: Physical Tension",
        "items": [
            "Trouble relaxing",
            "Being so restless that it's hard to sit still",
        ],
    },
    "irritability_fear": {
        "name": "GAD-7: Irritability & Apprehension",
        "items": [
            "Becoming easily annoyed or irritable",
            "Feeling afraid as if something awful might happen",
        ],
    },
}

PRIORITY_ORDER = [
    "core_anxiety",
    "physical_tension",
    "irritability_fear",
]

SCORING = {
    "frequency_options": [
        "Not at all (0)",
        "Several days (1)",
        "More than half the days (2)",
        "Nearly every day (3)",
    ],
    "total_range": (0, 21),
    "interpretation": [
        ((0, 4), "Minimal anxiety"),
        ((5, 9), "Mild anxiety"),
        ((10, 14), "Moderate anxiety"),
        ((15, 21), "Severe anxiety"),
    ],
}
