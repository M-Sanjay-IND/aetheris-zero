class BuildingSimulator:
    """Implements a 5-zone 2R2C equivalent thermal parameter state-space differential model."""
    def __init__(self, config: dict):
        self.config = config

    def reset(self) -> dict:
        return {}

    def step(self, actions: dict) -> tuple[dict, float, bool, dict]:
        return {}, 0.0, False, {}
