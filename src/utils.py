import os

from dotenv import load_dotenv


def get_wandb_token():
    load_dotenv()
    try:
        return os.environ["WANDB_TOKEN"]
    except KeyError:
        raise RuntimeError("Eviron does not have wandb token!")
