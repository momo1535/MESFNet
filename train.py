"""Portable configuration and logging entrypoint for the original trainer."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "model" / "train.py"


def config_keys():
    tree = ast.parse(ENGINE.read_text(encoding="utf-8-sig"))
    return {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and ast.unparse(node.func) == "os.environ.get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train.json")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES; this trainer uses one GPU")
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--hard-case-manifest", type=Path)
    parser.add_argument("--resume", action="store_true", help="Resume last complete epoch in save-dir")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print config without training")
    return parser.parse_args()


def build_environment(args):
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    allowed = config_keys()
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    if args.resume and args.init_checkpoint:
        raise ValueError("Use either --resume or --init-checkpoint")
    data_root = args.data_root.resolve()
    if not any((data_root / i).is_dir() and (data_root / m).is_dir()
               for i, m in (("image", "mask"), ("images", "labels"))):
        raise FileNotFoundError(f"Expected image/mask or images/labels in {data_root}")
    for path in (args.init_checkpoint, args.hard_case_manifest):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    if args.split_dir:
        for name in ("train_indices.txt", "val_indices.txt"):
            if not (args.split_dir / name).is_file():
                raise FileNotFoundError(args.split_dir / name)
    save_dir = args.save_dir.resolve()
    if not args.resume and save_dir.exists() and any(save_dir.iterdir()):
        raise FileExistsError(f"Use a new save-dir or --resume: {save_dir}")
    resume_state = save_dir / "training_state.pth"
    if args.resume:
        if not resume_state.is_file():
            resume_state = save_dir / "training_state_prev.pth"
        if not resume_state.is_file():
            raise FileNotFoundError("No training_state.pth or training_state_prev.pth in save-dir")
        if not (save_dir / "train_metrics_mesfnet.csv").is_file():
            raise FileNotFoundError("Resume requires the original metrics CSV")
    config.update({
        "WATER_DATA_ROOT": str(data_root), "SAVE_DIR": str(save_dir),
        "SPLIT_SOURCE_DIR": str(args.split_dir.resolve()) if args.split_dir else "",
        "INIT_CKPT": str(args.init_checkpoint.resolve()) if args.init_checkpoint else "",
        "HARD_CASE_MANIFEST": str(args.hard_case_manifest.resolve()) if args.hard_case_manifest else "",
        "RESUME": "1" if args.resume else "0",
        "RESUME_STATE": str(resume_state) if args.resume else "",
    })
    config = {key: str(int(value)) if isinstance(value, bool) else str(value)
              for key, value in config.items()}
    if config.get("AUG_MODE") == "targeted" or config.get("HARD_CASE_SAMPLING") == "1":
        if not config["HARD_CASE_MANIFEST"]:
            raise ValueError("Targeted augmentation/sampling requires --hard-case-manifest")
    if config.get("FULL_TRAIN") == "1" and args.split_dir:
        raise ValueError("FULL_TRAIN=1 cannot be combined with --split-dir")
    env = os.environ.copy()
    # Do not inherit an unrelated experiment's settings from the parent shell.
    for key in allowed:
        env.pop(key, None)
    env.update(config)
    env.update({"CUDA_VISIBLE_DEVICES": args.gpu,
                "PYTHONHASHSEED": config.get("SEED", "42"),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    return config, env


def relay(stream, console, log):
    for line in iter(stream.readline, ""):
        log.write(line)
        log.flush()
        console.write(line)
        console.flush()


def main():
    args = parse_args()
    config, env = build_environment(args)
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0
    save_dir = Path(config["SAVE_DIR"])
    logs = save_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    config_path = save_dir / "resolved_config.json"
    if not args.resume:
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        (save_dir / "entrypoint_sha256.txt").write_text(
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest() + "\n", encoding="ascii")
    with (logs / "train.stdout.log").open("a", encoding="utf-8") as stdout, \
         (logs / "train.stderr.log").open("a", encoding="utf-8") as stderr:
        process = subprocess.Popen([sys.executable, str(ENGINE)], cwd=ROOT, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8", errors="replace")
        workers = [threading.Thread(target=relay, args=(process.stdout, sys.stdout, stdout)),
                   threading.Thread(target=relay, args=(process.stderr, sys.stderr, stderr))]
        for worker in workers:
            worker.start()
        try:
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            code = 130
        finally:
            for worker in workers:
                worker.join()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
