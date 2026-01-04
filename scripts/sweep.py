import itertools
import subprocess
import copy
from pathlib import Path
from src.utils import load_yaml, save_yaml


def main():
    sweep_cfg = load_yaml("configs/sweep.yaml")
    base_cfg = load_yaml(sweep_cfg["base_config"])

    sweep_params = sweep_cfg["sweep"]
    keys = sweep_params.keys()
    values = sweep_params.values()

    runs = list(itertools.product(*values))

    out_dir = Path("sweep_runs")
    out_dir.mkdir(exist_ok=True)

    for i, combo in enumerate(runs):
        cfg = copy.deepcopy(base_cfg)

        for k, v in zip(keys, combo):
            if k in cfg["training"]:
                cfg["training"][k] = v
            elif k in cfg["model"]:
                cfg["model"][k] = v
            else:
                raise ValueError(f"Unknown sweep key: {k}")

        run_cfg_path = out_dir / f"run_{i}.yaml"
        save_yaml(cfg, run_cfg_path)

        print(f"\n=== Running sweep {i} ===")
        subprocess.run(
            [
                "python",
                "-m",
                "scripts.train",
                "--config",
                str(run_cfg_path),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
