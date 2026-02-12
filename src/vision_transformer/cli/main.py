from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import replace
from typing import Any, Literal, Union, get_args, get_origin

from vision_transformer.config.dataset import DatasetConfig
from vision_transformer.config.experiment import ExperimentConfig
from vision_transformer.config.loss import LossConfig
from vision_transformer.config.lr_scheduler import LRSchedulerConfig
from vision_transformer.config.model import ViTConfig
from vision_transformer.config.optimizer import OptimizerConfig
from vision_transformer.config.training import TrainingConfig
from vision_transformer.config.transform import TransformConfig
from vision_transformer.logger.logger_factory import LoggerFactory
from vision_transformer.training.train import training_loop

LoggerFactory.configure()
_log = LoggerFactory.get_logger()


def _parse_bool(s: str) -> bool:
    v = s.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid bool: {s!r}")


def _split_csv(s: str) -> list[str]:
    # Allow commas with optional spaces: "0.9,0.999" or "0.9, 0.999"
    return [p.strip() for p in s.split(",") if p.strip()]


def _parse_value(value_str: str, annotation: Any) -> Any:
    """
    Parse a CLI string into a python value based on type annotation.
    Handles: str, int, float, bool, Optional[T], Literal[...], tuple/Tuple[T,...]
    """
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional[T] == Union[T, NoneType]
    if origin is Union and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        inner = non_none[0] if non_none else str
        if value_str.strip().lower() in {"none", "null"}:
            return None
        return _parse_value(value_str, inner)

    # Literal
    if origin is Literal:
        # Compare against literal values as strings or parsed types when possible
        for lit in args:
            if str(lit) == value_str:
                return lit
        # Try a more permissive match
        if value_str in {str(a) for a in args}:
            return value_str
        raise ValueError(f"Invalid literal {value_str!r}, expected one of {args}")

    # tuple / Tuple
    if origin in {tuple, Tuple := tuple}:  # Tuple alias
        # Examples in configs:
        # - betas: tuple[float, float]
        # - mean/std: Tuple[float, float, float]
        parts = _split_csv(value_str)
        if len(args) == 2 and args[1] is Ellipsis:
            # tuple[T, ...]
            inner_t = args[0]
            return tuple(_parse_value(p, inner_t) for p in parts)
        if len(args) >= 1:
            if len(parts) != len(args):
                raise ValueError(
                    f"Expected {len(args)} values for tuple, got {len(parts)}: {value_str!r}"
                )
            return tuple(_parse_value(p, t) for p, t in zip(parts, args))
        # untyped tuple -> keep as strings
        return tuple(parts)

    # Plain types
    if annotation is bool:
        return _parse_bool(value_str)
    if annotation is int:
        return int(value_str)
    if annotation is float:
        return float(value_str)
    if annotation is str:
        return value_str

    # Fallback: attempt JSON, then string
    try:
        return json.loads(value_str)
    except Exception:
        return value_str


def _apply_set_overrides(cfg: ExperimentConfig, sets: list[str]) -> ExperimentConfig:
    """
    Apply overrides like:
      --set training.epochs=100
      --set optimizer.betas=0.9,0.98
      --set training.grad_clip_global_norm=None
    """
    # Build a mapping from section name to dataclass object
    sections: dict[str, Any] = {
        "model": cfg.model,
        "transform": cfg.transform,
        "dataset": cfg.dataset,
        "optimizer": cfg.optimizer,
        "lr_scheduler": cfg.lr_scheduler,
        "loss": cfg.loss,
        "training": cfg.training,
    }

    for item in sets:
        if "=" not in item:
            raise ValueError(f"--set must be like section.field=value, got: {item!r}")
        key, raw_val = item.split("=", 1)
        if "." not in key:
            raise ValueError(f"--set key must be section.field, got: {key!r}")

        section_name, field_name = key.split(".", 1)
        if section_name not in sections:
            raise ValueError(
                f"Unknown section {section_name!r}. "
                f"Expected one of {sorted(sections.keys())}"
            )

        section_obj = sections[section_name]
        if not dataclasses.is_dataclass(section_obj):
            raise ValueError(f"Section {section_name!r} is not a dataclass")

        field_map = {f.name: f for f in dataclasses.fields(section_obj)}
        if field_name not in field_map:
            raise ValueError(
                f"Unknown field {section_name}.{field_name}. "
                f"Expected one of {sorted(field_map.keys())}"
            )

        annotation = field_map[field_name].type
        parsed_val = _parse_value(raw_val, annotation)

        # Update the section dataclass, then rebuild ExperimentConfig
        new_section_obj = replace(section_obj, **{field_name: parsed_val})
        sections[section_name] = new_section_obj

    return ExperimentConfig(
        model=sections["model"],
        transform=sections["transform"],
        dataset=sections["dataset"],
        optimizer=sections["optimizer"],
        lr_scheduler=sections["lr_scheduler"],
        loss=sections["loss"],
        training=sections["training"],
    )


def _build_default_experiment() -> ExperimentConfig:
    return ExperimentConfig(
        model=ViTConfig(),
        transform=TransformConfig(),
        dataset=DatasetConfig(),
        optimizer=OptimizerConfig(),
        lr_scheduler=LRSchedulerConfig(),
        loss=LossConfig(),
        training=TrainingConfig(),
    )


def _add_train_args(p: argparse.ArgumentParser) -> None:

    # Generic override mechanism
    p.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override any field: section.field=value (repeatable). "
        "Example: --set training.grad_clip_global_norm=None",
    )

    # Debug: print resolved config and exit
    p.add_argument(
        "--print-config", action="store_true", help="Print resolved config then exit"
    )


def _as_dict(dc: Any) -> Any:
    if dataclasses.is_dataclass(dc):
        return {f.name: _as_dict(getattr(dc, f.name)) for f in dataclasses.fields(dc)}
    if isinstance(dc, (list, tuple)):
        return [_as_dict(x) for x in dc]
    return dc


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    parser = argparse.ArgumentParser(prog="program")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="training: program train arguments")
    _add_train_args(p_train)

    # Placeholder for later
    p_infer = sub.add_parser("infer", help="inference: program infer arguments (TODO)")
    p_infer.add_argument("--todo", action="store_true", help="Not implemented yet")

    args = parser.parse_args(argv)

    if args.command == "infer":
        raise NotImplementedError("infer CLI is not written yet (as requested).")

    if args.command == "train":
        cfg = _build_default_experiment()
        cfg = _apply_set_overrides(cfg, args.set)

        if args.print_config:
            print(json.dumps(_as_dict(cfg), indent=2))
            return 0

        _log.info(">    Experiment Configuration:\n")
        _log.info(cfg)
        _log.info("\n")

        # Training
        training_loop(experiment_config=cfg)

        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
