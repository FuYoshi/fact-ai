from socialjax.environments import CoinGame

REGISTERED_ENVS = ["coin_game"]


def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in REGISTERED_ENVS:
        raise ValueError(f"{env_id} is not in registered SocialJax environments")
    elif env_id == "coin_game":
        env = CoinGame(**env_kwargs)
    return env
