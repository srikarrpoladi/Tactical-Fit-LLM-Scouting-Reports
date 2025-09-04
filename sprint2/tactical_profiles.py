# Predefined tactical profiles for different player roles
# Values are normalized 0-1 for composite attributes:
# attack, passing, dribbling, defense, physical, goalkeeping

# Predefined tactical profiles for different player roles


TACTICAL_PROFILES = {
    "attacking_forward": {
        "attack": 0.95,
        "passing": 0.7,
        "dribbling": 0.9,
        "defense": 0.3,
        "physical": 0.7,
        "goalkeeping": 0.0
    },
    "creative_midfielder": {
        "attack": 0.75,
        "passing": 0.95,
        "dribbling": 0.85,
        "defense": 0.4,
        "physical": 0.6,
        "goalkeeping": 0.0
    },
    "box_to_box_midfielder": {
        "attack": 0.75,
        "passing": 0.8,
        "dribbling": 0.75,
        "defense": 0.75,
        "physical": 0.8,
        "goalkeeping": 0.0
    },
    "defensive_fullback": {
        "attack": 0.5,
        "passing": 0.7,
        "dribbling": 0.65,
        "defense": 0.9,
        "physical": 0.85,
        "goalkeeping": 0.0
    },
    "central_defender": {
        "attack": 0.2,
        "passing": 0.6,
        "dribbling": 0.4,
        "defense": 0.95,
        "physical": 0.9,
        "goalkeeping": 0.0
    },
    "goalkeeper": {
        "attack": 0.0,
        "passing": 0.5,
        "dribbling": 0.3,
        "defense": 0.6,
        "physical": 0.7,
        "goalkeeping": 1.0
    }
}
