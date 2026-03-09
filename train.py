"""
=============================================================================
  Insurance Fraud Detection System
  Module: Main Training Script
  Description: Trains and evaluates all three fraud detection models,
               generates all evaluation plots, and prints a final summary.
               Run this script FIRST before launching the Streamlit UI.

  Usage:
    python train.py
=============================================================================
"""

import sys, os, time

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scenarios.auto_fraud     import run_auto_pipeline
from scenarios.health_fraud   import run_health_pipeline
from scenarios.property_fraud import run_property_pipeline

DIVIDER = "═" * 60


def print_report_card(pipeline_result: dict):
    """Print a formatted evaluation table for one scenario."""
    scenario = pipeline_result["scenario"].upper()
    results  = pipeline_result["results"]

    print(f"\n{'─'*60}")
    print(f"  📋  {scenario} INSURANCE — MODEL REPORT CARD")
    print(f"{'─'*60}")
    print(f"  {'Model':<24} | {'Accuracy':>8} | {'AUC-ROC':>8} | {'F1-Fraud':>9}")
    print(f"  {'─'*22} | {'─'*8} | {'─'*8} | {'─'*9}")

    for res in results:
        f1_fraud = res["report"].get("Fraud", {}).get("f1-score", 0.0)
        flag = " 🏆" if res == pipeline_result["best"] else "   "
        print(f"  {res['name']:<24} | {res['acc']:>8.4f} | {res['auc']:>8.4f} | {f1_fraud:>9.4f}{flag}")

    print(f"\n  Best Model : {pipeline_result['best']['name']}")
    print(f"  AUC-ROC    : {pipeline_result['best']['auc']:.4f}")


def main():
    t0 = time.time()
    print(f"\n{DIVIDER}")
    print("  🛡️   INSURANCE FRAUD DETECTION SYSTEM")
    print("       Training All Three Scenarios")
    print(DIVIDER)

    # ── Run all pipelines ──────────────────────────────────────────────────
    auto_res     = run_auto_pipeline(verbose=True)
    health_res   = run_health_pipeline(verbose=True)
    property_res = run_property_pipeline(verbose=True)

    # ── Print summary report cards ─────────────────────────────────────────
    print(f"\n\n{DIVIDER}")
    print("  📊  FINAL SUMMARY — ALL SCENARIOS")
    print(DIVIDER)

    print_report_card(auto_res)
    print_report_card(health_res)
    print_report_card(property_res)

    elapsed = time.time() - t0
    print(f"\n{DIVIDER}")
    print(f"  ✅  All models trained and saved in {elapsed:.1f}s")
    print(f"  📁  Models  → saved_models/")
    print(f"  🖼️  Plots   → plots/")
    print(f"  🌐  Launch UI: streamlit run app.py")
    print(DIVIDER + "\n")


if __name__ == "__main__":
    main()
