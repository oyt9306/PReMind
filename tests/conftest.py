"""CPU-only test session: no bytecode, no GPU.

The stored render fixtures (tests/fixtures/vcm_*.json) were rendered with system prompt v1 (the pre-2026-10-06
prompt); the byte-equality re-render check in vcm_data.build_items needs the same prompt, so the test session pins
PREMIND_PROMPT=v1. Training and evaluation default to v2 (configs/*.yaml).
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PREMIND_PROMPT"] = "v1"
os.environ.pop("CONFIDANT_PROMPT", None)
