"""Lightweight CLI registration for NatScale calibration commands."""

from __future__ import annotations

import argparse


def run(args: argparse.Namespace) -> int:
    from mfq.calibration.natscale_errors import run as run_natscale

    return run_natscale(args)


def add_parsers(stages: argparse._SubParsersAction) -> None:
    for name, help_text in (
        ("natscale-errors", "measure imatrix-weighted format errors with export codecs"),
        ("natscale-statistics", "collect weight spectra and column energies in one pass"),
    ):
        parser = stages.add_parser(name, help=help_text)
        parser.add_argument("--model", required=True)
        parser.add_argument("--imatrix", required=True)
        parser.add_argument("--cache", required=True)
        parser.add_argument(
            "--backend", choices=("auto", "cpu", "cuda", "metal"), default="auto"
        )
        parser.add_argument("--device", default="")
        if name == "natscale-errors":
            parser.add_argument("--profiles", default="")
            parser.add_argument("--sample-layers", type=int, default=6)
            parser.add_argument("--sample-experts", type=int, default=3)
            parser.add_argument("--row-chunk", type=int, default=0)
            parser.add_argument("--output", default="")
        parser.set_defaults(_impl=run)


__all__ = ["add_parsers", "run"]
