from .base import Environment


def make_environment(name: str = None, **kwargs) -> Environment:
    if name in ("default", "", None):
        from .env import DefaultEnvironment
        return DefaultEnvironment(**kwargs)

    raise ValueError(f"Unknown env: {name}")
