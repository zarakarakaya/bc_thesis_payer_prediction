import yaml
from types import SimpleNamespace


def load_config(path):
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    return dict_to_namespace(cfg_dict)


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(
            **{k: dict_to_namespace(v) for k, v in d.items()}
        )
    return d
def namespace_to_dict(ns):
    return {
        k: namespace_to_dict(v) if hasattr(v, "__dict__") else v
        for k, v in vars(ns).items()
    }