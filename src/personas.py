"""
Persona definitions for LP2.
Refined for stronger separation (JS > 0.3, overlap < 50%).
"""

PERSONAS = {
    "worker": {
        "description": "A person doing office/productive work (using the table as a desk)",
        "preferred_labels": [
            "table", "commode", "trash can", "light", "heater"
        ],
        "avoided_labels": [
            "sofa", "tv", "beanbag", "cushion", "plant", "window", "shades", "windowsill"
        ],
        "interaction_duration_mean_seconds": 45,
        "walk_speed_m_per_s": 1.4,
        "n_interactions": 5
    },
    "relaxer": {
        "description": "A person resting or leisurely exploring the space",
        "preferred_labels": [
            "sofa", "tv", "beanbag", "cushion", "plant", "window", "shades", "windowsill"
        ],
        "avoided_labels": [
            "table", "commode", "trash can", "heater", "light"
        ],
        "interaction_duration_mean_seconds": 90,
        "walk_speed_m_per_s": 0.9,
        "n_interactions": 4
    }
}
