import yaml
import json
from pathlib import Path

class Config:
    """Config loader supporting YAML or JSON."""

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Config file not found: {self.path}")

        if self.path.suffix in [".yaml", ".yml"]:
            with open(self.path, "r") as f:
                self.cfg = yaml.safe_load(f)
        elif self.path.suffix == ".json":
            with open(self.path, "r") as f:
                self.cfg = json.load(f)
        else:
            raise ValueError("Unsupported config format. Use .yaml or .json")

    def get(self, key, default=None):
        return self.cfg.get(key, default)

    # -------------------------
    # Standard fields
    # -------------------------
    @property
    def model_name(self):
        return self.cfg["model_name"]

    @property
    def api_key(self):
        return self.cfg["api_key"]

    @property
    def device(self):
        return self.cfg.get("device", "auto")

    @property
    def save_dir(self):
        return self.cfg.get("save_dir", "outputs")

    # -------------------------
    # Dataset config block
    # -------------------------
    def get_dataset_config(self):
        """
        Returns dataset configuration block:
            dataset:
                type: synthetic
                path: data/synthetic.json
        """
        return self.cfg.get("dataset", {})

    # -------------------------
    # Experiment parameters
    # -------------------------
    def get_experiment(self):
        return self.cfg.get("experiment", {})

INDIRECT_COLORS = [
    " Black",
    " Blue",
    " Brown",
    " Gold",
    " Golden",
    " Gray",
    " Green",
    " Grey",
    " Orange",
    " Pink",
    " Purple",
    " Red",
    " Silver",
    " White",
    " Yellow",
]

DIRECT_COLORS = [
    "beige",
    "black",
    "blue",
    "brown",
    "coral",
    "cyan",
    "gold",
    "gray",
    "green",
    "lime",
    "magenta",
    "maroon",
    "navy",
    "olive",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "teal",
    "turquoise",
    "violet",
    "white",
    "yellow",
]

COLORS_TO_HEX = {
    " Black": "#000000",
    " Blue": "#0000ff",
    " Brown": "#a52a2a",
    " Gold": "#ffd700",
    " Golden": "#ffd700",
    " Gray": "#808080",
    " Green": "#00ff00",
    " Grey": "#808080",
    " Orange": "#ffa500",
    " Pink": "#ffc0cb",
    " Purple": "#800080",
    " Red": "#ff0000",
    " Silver": "#c0c0c0",
    " White": "#ffffff",
    " Yellow": "#c0c000",
    "beige": "#f5f5dc",
    "black": "#000000",
    "blue": "#0000ff",
    "brown": "#a52a2a",
    "coral": "#ff7f50",
    "cyan": "#00ffff",
    "gold": "#ffd700",
    "gray": "#808080",
    "green": "#00ff00",
    "lime": "#00ff00",
    "magenta": "#ff00ff",
    "maroon": "#800000",
    "navy": "#000080",
    "olive": "#808000",
    "orange": "#ffa500",
    "pink": "#ffc0cb",
    "purple": "#800080",
    "red": "#ff0000",
    "silver": "#c0c0c0",
    "teal": "#008080",
    "turquoise": "#40e0d0",
    "violet": "#800080",
    "white": "#ffffff",
    "yellow": "#c0c000",
}