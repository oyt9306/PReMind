"""The prompt's output schema (premind.prompts.final_prompts) and the parser's strict format check
(premind.rewards.legacy.baseline_eval.evaluate._STRICT) must follow the same env lookup:
PREMIND_SCHEMA, else CONFIDANT_SCHEMA, else "full". Each check runs in a fresh interpreter (both modules read the
env at import time)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ("full", "lite", "lite_ve", "lite_rs")
PROBE = r"""
import json, re
from premind.prompts import final_prompts as FP
from premind.rewards.legacy.baseline_eval import evaluate as EV
def filled(fmt):
    return re.sub(r">[^<]+</", ">x</", fmt)
tags = lambda fmt: re.findall(r"<(\w+)>", fmt)
own = FP._OUTPUT_FORMATS[FP.SCHEMA]
print(json.dumps({"schema": FP.SCHEMA, "own_ok": EV.parse(filled(own))["format_strict"],
                  "others": {k: EV.parse(filled(v))["format_strict"] for k, v in FP._OUTPUT_FORMATS.items()},
                  "tags": {k: tags(v) for k, v in FP._OUTPUT_FORMATS.items()}}))
"""


def _probe(env_extra: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("PREMIND_SCHEMA", "CONFIDANT_SCHEMA")}
    env.update(env_extra, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _subseq(a: list, b: list) -> bool:
    it = iter(b)
    return all(x in it for x in a)


@pytest.mark.parametrize("name", ["PREMIND_SCHEMA", "CONFIDANT_SCHEMA"])
@pytest.mark.parametrize("schema", SCHEMAS)
def test_prompt_and_parser_schema_agree(name: str, schema: str) -> None:
    r = _probe({name: schema})
    assert r["schema"] == schema
    assert r["own_ok"], f"{name}={schema}: the prompt's own output format fails the strict parser"
    own = r["tags"][schema]
    for other, ok in r["others"].items():   # strict check accepts exactly the formats carrying the schema's tags in order
        assert ok == _subseq(own, r["tags"][other]), (name, schema, other)


def test_default_is_full_and_premind_takes_precedence() -> None:
    assert _probe({})["schema"] == "full"
    r = _probe({"PREMIND_SCHEMA": "lite", "CONFIDANT_SCHEMA": "full"})
    assert r["schema"] == "lite" and r["own_ok"]
