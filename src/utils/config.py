import yaml
from pathlib import Path

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
