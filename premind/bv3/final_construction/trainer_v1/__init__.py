"""trainer_v1: final-matrix GRPO trainer wiring (PROVISIONAL, CPU-verified). Imported as a package so that
``trainer_v1.config`` never shadows ``baseline_eval/config.py`` (imported by train_interface_v1/render.py)."""
__all__ = ["config", "rollout", "trl_hook", "mock", "train"]
