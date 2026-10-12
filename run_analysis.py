from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gjb2_analysis.data import complete_cases, read_genotype_workbook
from gjb2_analysis.external import analyse_external_counts
from gjb2_analysis.models import fit_ebm, fit_svm
from gjb2_analysis.pairs import enumerate_pairs, select_contexts
from gjb2_analysis.plotting import plot_context_case_fractions


CONTEXTS = [
    ("c.235delC", "c.299_300delAT"),
    ("c.176del16bp", "c.235delC"),
    ("c.235delC", "p.V27I"),
    ("c.235delC", "p.E114G"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--external-counts", type=Path, default=Path("data/external/contrasts.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--skip-svm", action="store_true")
    parser.add_argument("--skip-ebm", action="store_true")
    return parser.parse_args()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    discovery_raw = read_genotype_workbook(args.data_dir / "Discovery Set.xlsx")
    validation_raw = read_genotype_workbook(args.data_dir / "Validation Set.xlsx")
    discovery, discovery_missing = complete_cases(discovery_raw)
    validation, validation_missing = complete_cases(validation_raw)

    discovery_missing.to_csv(args.output_dir / "discovery_missing_genotypes.csv", index=False)
    validation_missing.to_csv(args.output_dir / "validation_missing_genotypes.csv", index=False)

    discovery_pairs = enumerate_pairs(discovery)
    validation_pairs = enumerate_pairs(validation)
    discovery_pairs.to_csv(args.output_dir / "gjb2_pairs_discovery.csv", index=False)
    validation_pairs.to_csv(args.output_dir / "gjb2_pairs_validation.csv", index=False)

    discovery_primary = select_contexts(discovery_pairs, CONTEXTS)
    validation_primary = select_contexts(validation_pairs, CONTEXTS)
    discovery_primary.to_csv(args.output_dir / "primary_contexts_discovery.csv", index=False)
    validation_primary.to_csv(args.output_dir / "primary_contexts_validation.csv", index=False)

    plot_context_case_fractions(
        discovery_primary,
        validation_primary,
        discovery.case_fraction,
        validation.case_fraction,
        args.output_dir / "gjb2_context_case_fractions.png",
    )

    external = analyse_external_counts(args.external_counts)
    external.to_csv(args.output_dir / "external_genotype_contrasts.csv", index=False)

    dataset_summary = {
        "discovery": {
            "raw_n": discovery_raw.n,
            "complete_n": discovery.n,
            "cases": int(discovery.y.sum()),
            "controls": int(discovery.n - discovery.y.sum()),
            "case_fraction": discovery.case_fraction,
        },
        "validation": {
            "raw_n": validation_raw.n,
            "complete_n": validation.n,
            "cases": int(validation.y.sum()),
            "controls": int(validation.n - validation.y.sum()),
            "case_fraction": validation.case_fraction,
        },
    }
    write_json(args.output_dir / "dataset_summary.json", dataset_summary)

    if not args.skip_svm:
        svm_metrics, svm_predictions = fit_svm(discovery_raw, validation_raw)
        write_json(args.output_dir / "svm_metrics.json", svm_metrics)
        svm_predictions.to_csv(args.output_dir / "svm_predictions.csv", index=False)

    if not args.skip_ebm:
        ebm_metrics, ebm_importance, ebm_predictions = fit_ebm(discovery, validation)
        write_json(args.output_dir / "ebm_metrics.json", ebm_metrics)
        ebm_importance.to_csv(args.output_dir / "ebm_term_importance.csv", index=False)
        ebm_predictions.to_csv(args.output_dir / "ebm_predictions.csv", index=False)


if __name__ == "__main__":
    main()
