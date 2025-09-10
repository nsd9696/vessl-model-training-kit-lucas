from pathlib import Path

def get_project_root():
    return Path(__file__).parent.parent

def get_assets_dir():
    return get_project_root() / "assets"
