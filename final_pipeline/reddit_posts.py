"""
Example reddit posts dataset for therapy simulation.
These are anonymized and modified posts from Reddit that provide context for patient roleplay.
"""

from typing import Dict, List

REDDIT_POSTS = {
    "anxiety_workplace": [
        """I (28M) have been struggling with severe anxiety at work, particularly during meetings. 
        I'm a software engineer and generally good at my job, but I have this overwhelming fear 
        that my coworkers will judge my ideas or think I'm incompetent. Last week in a standup, 
        I froze up trying to explain my technical approach and my boss had to jump in. I feel like 
        I'm failing and everyone can see it. My heart races before meetings and I sometimes call 
        in sick just to avoid them. I don't know how much longer I can keep doing this."""
    ],
    
    "depression_isolation": [
        """I (35F) feel increasingly isolated and empty inside. Over the past few months, 
        nothing seems to bring me joy anymore. I used to love painting and hiking, but now I 
        can't find the energy to do anything. I go through the motions at work, come home, 
        and just scroll my phone until bed. My friends have stopped inviting me to things 
        because I always say no. I sleep 10-12 hours but still feel exhausted. My family 
        thinks I'm just being lazy, but I don't know how to explain that I just can't."""
    ],
    
    "relationship_conflict": [
        """My partner (42M) and I (40F) have been together for 15 years but things have 
        deteriorated significantly. We barely talk anymore except about logistics. When we do, 
        it turns into criticism and resentment. They spend most evenings on their laptop and 
        when I try to express how lonely I feel, they say I'm being needy. I've suggested 
        therapy multiple times but they dismiss it. I hate that I'm even considering separation 
        after all these years, but I feel like I'm losing myself. Is this just normal for a 
        long-term relationship or is something broken?"""
    ],
    
    "grief_loss": [
        """My mother passed away 6 months ago and I'm struggling to process it. 
        She was my best friend and I wasn't ready to lose her. Logically I know grief takes time, 
        but I feel guilty that I'm not grieving the "right way." Some days I'm fine, other days 
        I'm completely lost. I've gone back to work but I'm just going through motions. 
        I avoid talking about her because I don't want to burden people with my sadness. 
        Sometimes I pick up the phone to call her before I remember she's gone. 
        I don't know if this is normal or if I need help."""
    ],
    
    "self_esteem_perfectionism": [
        """I (26F) struggle with constant self-criticism and perfectionism. 
        Whatever I achieve never feels good enough. I'll do amazing work on projects but only 
        see the flaws. My romantic relationships fail because a small imperfection makes me 
        think I'm not good enough for my partner. I've sabotaged interviews and opportunities 
        because I convinced myself I'd fail anyway. My therapist says I have impostor syndrome 
        but it feels so real. How do I stop being so hard on myself? When did my brain become 
        my enemy?"""
    ],
    
    "family_trauma": [
        """I (31M) grew up in a chaotic household where my father was emotionally and sometimes 
        physically abusive. My mother didn't intervene. I'm now in my 30s and I'm realizing how 
        much this has molded me. I have trust issues, I struggle with anger, and I'm terrified 
        of becoming like my father. I'm in a good relationship now but I'm sabotaging it because 
        I'm always waiting for the other shoe to drop. My family acts like nothing happened and 
        wants me to "just move on." I don't know how to heal from this or if my relationship 
        can survive my baggage."""
    ]
}

PATIENT_PROFILES = {
    "anxiety_workplace": {
        "name": "Alex",
        "age": 28,
        "gender": "Male",
        "occupation": "Software Engineer",
        "family_structure": "Married, no children",
        "primary_concern": "Workplace anxiety, fear of judgment",
        "history": "No prior mental health treatment"
    },
    
    "depression_isolation": {
        "name": "Sarah",
        "age": 35,
        "gender": "Female",
        "occupation": "Project Manager",
        "family_structure": "Single, close friends",
        "primary_concern": "Depression, anhedonia, isolation",
        "history": "Had one therapy session 5 years ago"
    },
    
    "relationship_conflict": {
        "name": "Michael",
        "age": 40,
        "gender": "Male",
        "occupation": "Accountant",
        "family_structure": "Married 15 years",
        "primary_concern": "Relationship deterioration, loneliness",
        "history": "Never seen a therapist"
    },
    
    "grief_loss": {
        "name": "Emma",
        "age": 32,
        "gender": "Female",
        "occupation": "Teacher",
        "family_structure": "Single, close to mother",
        "primary_concern": "Grief, loss of mother",
        "history": "First time seeking help"
    },
    
    "self_esteem_perfectionism": {
        "name": "Jessica",
        "age": 26,
        "gender": "Female",
        "occupation": "Marketing Specialist",
        "family_structure": "Dating, estranged from family",
        "primary_concern": "Perfectionism, self-criticism, impostor syndrome",
        "history": "Recently started therapy"
    },
    
    "family_trauma": {
        "name": "James",
        "age": 31,
        "gender": "Male",
        "occupation": "Writer",
        "family_structure": "In relationship, distant from family",
        "primary_concern": "Childhood trauma, trust issues",
        "history": "First therapy"
    }
}


def get_reddit_post(scenario: str) -> List[str]:
    """Get Reddit post for a specific scenario."""
    if scenario in REDDIT_POSTS:
        return REDDIT_POSTS[scenario]
    raise ValueError(f"Unknown scenario: {scenario}. Choose from: {list(REDDIT_POSTS.keys())}")


def get_patient_profile(scenario: str) -> Dict:
    """Get patient profile for a specific scenario."""
    if scenario in PATIENT_PROFILES:
        return PATIENT_PROFILES[scenario]
    raise ValueError(f"Unknown scenario: {scenario}. Choose from: {list(PATIENT_PROFILES.keys())}")


def list_scenarios() -> List[str]:
    """List all available scenarios."""
    return list(REDDIT_POSTS.keys())
