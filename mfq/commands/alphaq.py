"""Weight-only AlphaQ calibration and allocation CLI."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def run(args: argparse.Namespace) -> int:
    from mfq.calibration.alphaq import allocate_alphaq, alphaq_nint_candidates
    from mfq.calibration.alphaq_source import _atomic_json, _progress, collect_alphaq
    from mfq.calibration.artifact import save_scheme
    from mfq.calibration.ew_solver import EwBudget, RateBounds, load_budget, load_candidate_table
    from mfq.commands.solve_ew import _rebase_scheme_artifacts

    output = Path(args.output).resolve()
    statistics = (
        Path(args.statistics).resolve()
        if args.statistics
        else output.with_suffix(".alphaq-statistics.json")
    )
    report = Path(args.report).resolve() if args.report else output.with_suffix(".report.json")
    if len({output, statistics, report}) != 3:
        raise ValueError("AlphaQ output, statistics and report paths must differ")
    inputs = {Path(p).resolve() for p in (args.model, args.candidates, args.budget) if p}
    if {output, statistics, report} & inputs:
        raise ValueError("AlphaQ output paths must differ from input paths")
    if args.profile and args.candidates:
        raise ValueError("--profile and --candidates cannot be combined")
    if args.target_bpw is not None and (not math.isfinite(args.target_bpw) or args.target_bpw <= 0):
        raise ValueError("target BPW must be finite and positive")
    for path in (output, report):
        if path.exists():
            raise FileExistsError(f"output already exists: {path}")
    # Metadata-only inputs are checked before doing expensive spectral work.
    candidates = load_candidate_table(args.candidates) if args.candidates else None
    budget = load_budget(args.budget) if args.budget else None
    if args.solver == "hull":
        joint = False
        if budget is not None:
            lower, upper = budget.total.resolve(budget.model_weight_count, "total")
            joint = (
                lower not in (None, 0)
                or upper is None
                or budget.projections
                or budget.layers
                or budget.shape_constraints
            )
        if joint or (
            candidates is not None and any(c.pool_storage_bits for c in candidates.candidates)
        ):
            raise ValueError("shared pool costs or joint constraints require --solver exact")
    values = collect_alphaq(args.model, statistics, device=args.device)
    if candidates is None:
        candidates = alphaq_nint_candidates(values, args.profile)
    if budget is None:
        budget = EwBudget(
            "AlphaQ",
            candidates.routed_weight_count,
            0,
            RateBounds(max_bpw=args.target_bpw),
            {},
            {},
            (),
            {},
        )
    result = allocate_alphaq(values, candidates, budget, solver=args.solver)
    scheme = _rebase_scheme_artifacts(result.scheme, candidates, output)
    save_scheme(output, scheme)
    _atomic_json(report, {**result.report, "statistics": str(statistics), "scheme": str(output)})
    _progress(
        status="ok",
        scheme=str(output),
        report=str(report),
        statistics=str(statistics),
        routed_bpw=scheme.bpw,
        quality_status="not evaluated; surrogate only",
    )
    return 0


def add_parser(stages: argparse._SubParsersAction) -> None:
    parser = stages.add_parser(
        "alphaq",
        help="allocate MoE precision from weights only (no calibration data)",
        description="AlphaQ FARMS/Hill importance with fixed code-default settings; no corpus, forward or backward pass.",
    )
    parser.add_argument(
        "--model", required=True, help="local HF Safetensors directory or full-precision MFQ file"
    )
    parser.add_argument(
        "--output",
        "--output-scheme",
        dest="output",
        required=True,
        help="MFQ calibration scheme for quantize --scheme",
    )
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument(
        "--target-bpw",
        type=float,
        help="global routed-expert payload BPW ceiling; excludes fixed container headers",
    )
    budget.add_argument("--budget", help="existing mfq.ew-budget.v1 document")
    parser.add_argument(
        "--statistics", help="reusable/resumable statistics JSON (default: beside output)"
    )
    parser.add_argument("--report", help="allocation report JSON (default: beside output)")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu or cuda:N; used only for uncached spectral statistics",
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=("NINT2", "NINT3", "NINT4", "NINT5", "NINT6", "NINT8"),
        help="repeat to restrict the built-in standard NINT candidate set",
    )
    parser.add_argument(
        "--candidates",
        help="mfq.ew-candidates.v1 for custom/NVQ candidates and exact storage costs; measured distortions are not needed",
    )
    parser.add_argument(
        "--solver",
        choices=("hull", "exact"),
        default="hull",
        help="hull: fast global-budget allocation; exact: EW MILP with shared pool costs/joint constraints (can be expensive)",
    )
    parser.set_defaults(_impl=run)


__all__ = ["add_parser", "run"]
