from pathlib import Path

def get_base_dir():
    return Path(__file__).parent.parent

def get_nb_dir():
    nb_dir = get_base_dir() / 'nb'
    assert nb_dir.exists(), "The 'nb' directory does not exist."
    return nb_dir

def get_simulations_dir():
    simulations_dir = get_base_dir() / 'simulations'
    simulations_dir.mkdir(parents=True, exist_ok=True)
    return simulations_dir

def get_media_dir():
    media_dir = get_base_dir() / 'media'
    media_dir.mkdir(parents=True, exist_ok=True)
    return media_dir