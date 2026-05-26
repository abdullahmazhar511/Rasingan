#!/usr/bin/env python
"""
Main runner script for the therapeutic agent pipeline.
Example usage of the three-agent system: Patient, Primary Therapist, Senior Therapist Supervisor.
"""

import argparse
import csv
import sys
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm.auto import tqdm

from pipeline import TherapyPipeline
from reddit_posts import get_reddit_post, get_patient_profile, list_scenarios
from shared_vllm import DEFAULT_MODEL as DEFAULT_SHARED_MODEL
from hf_therapist import HFTherapist


# ---------------------------------------------------------------------------
# Scenario specs
#
# A "scenario spec" is the data a single session needs:
#   {
#     "name":            str,        # unique label used as the scenario id
#     "reddit_posts":    list[str],  # patient role-play seed text
#     "patient_profile": dict,       # demographics — used to scaffold the prompt
#     "extra_meta":      dict,       # extra fields stored in session metadata
#                                    # (e.g. {"post_id": "...", "category": "anxiety"})
#   }
#
# Two sources today:
#   (a) the 6 hardcoded scenarios in reddit_posts.py  (legacy / default)
#   (b) a CSV file with columns post_id,title,body,category — used to drive
#       runs against the multiturn_reddit_data/splits/{train,val,test}.csv
# ---------------------------------------------------------------------------


def _spec_from_named_scenario(name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "reddit_posts": get_reddit_post(name),
        "patient_profile": get_patient_profile(name),
        "extra_meta": {"source": "reddit_posts.py", "category": None, "post_id": None},
    }


def _spec_from_csv_row(row: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Build a scenario spec from a row of multiturn_reddit_data splits CSV.

    Uses the same minimal patient_profile synthesis as
    verl/examples/faith/data_preprocess/preprocess_multiturn.py so the
    two pipelines see equivalent inputs.
    """
    title = (row.get("title") or "").strip()
    body = (row.get("body") or "").strip()
    post_id = (row.get("post_id") or f"row{idx}").strip()
    category = (row.get("category") or "").strip()

    # Reddit-post text used by the patient simulator (title + body, like prefix)
    text = f"{title}\n\n{body}" if title else body
    reddit_posts = [text]

    patient_profile = {
        "name": "Patient",
        "age": "adult",
        "occupation": "not specified",
        "family_structure": "not specified",
        "primary_concern": category or "mental health",
    }

    # Slug used as both the in-memory scenario id and the on-disk session tag.
    name = f"{category or 'post'}_{post_id}" if post_id else f"post_{idx}"

    return {
        "name": name,
        "reddit_posts": reddit_posts,
        "patient_profile": patient_profile,
        "extra_meta": {
            "source": "csv",
            "post_id": post_id,
            "category": category or None,
            "title": title or None,
        },
    }


def load_specs_from_csv(
    csv_path: str,
    categories: Optional[List[str]] = None,
    n_per_category: Optional[int] = None,
    n_total: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load scenario specs from a CSV with columns post_id,title,body,category.

    Args:
        csv_path: path to CSV
        categories: if set, only rows whose `category` is in this list are kept
        n_per_category: if set, take the first N rows from each category
                        (after filtering by `categories`)
        n_total: optional hard cap on the total returned (applied last)
    """
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            cat = (row.get("category") or "").strip()
            if categories and cat not in categories:
                continue
            by_cat[cat].append(_spec_from_csv_row(row, idx))

    specs: List[Dict[str, Any]] = []
    cat_order = categories if categories else sorted(by_cat.keys())
    for cat in cat_order:
        bucket = by_cat.get(cat, [])
        if n_per_category is not None:
            bucket = bucket[:n_per_category]
        specs.extend(bucket)

    if n_total is not None:
        specs = specs[:n_total]
    return specs


def run_therapy_session(
    scenario: Optional[str] = None,
    max_turns: int = 8,
    enable_supervisor: bool = True,
    enable_suggestions: bool = True,
    verbose: bool = True,
    output_dir: str = "sessions",
    custom_post: Optional[str] = None,
    custom_profile: Optional[dict] = None,
    shared_model: str = DEFAULT_SHARED_MODEL,
    therapist_model: str = "Qwen/Qwen3-4B-Instruct-2507",
    therapist_llm: Optional[HFTherapist] = None,
    supervisor_endpoint: Optional[str] = None,
    spec: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run a therapy session with the specified configuration.

    Either pass `scenario` (a name registered in reddit_posts.py) OR a pre-built
    `spec` dict ({name, reddit_posts, patient_profile, extra_meta}). When `spec`
    is supplied it takes precedence — that's the CSV-driven path.
    """

    # Resolve into a spec dict regardless of how the caller specified input.
    if spec is None:
        if scenario is None:
            raise ValueError("run_therapy_session needs either `scenario` or `spec`")
        spec = _spec_from_named_scenario(scenario)

    # Allow the old custom_post / custom_profile overrides on the named-scenario path.
    if custom_post:
        spec = {**spec, "reddit_posts": [custom_post]}
    if custom_profile:
        spec = {**spec, "patient_profile": custom_profile}

    scenario_name = spec["name"]
    reddit_posts = spec["reddit_posts"]
    patient_profile = spec["patient_profile"]
    extra_meta = spec.get("extra_meta", {}) or {}

    if verbose:
        print(f"\n{'='*80}")
        print(f"STARTING THERAPY SESSION: {scenario_name}")
        print(f"{'='*80}")
        print(f"Patient: {patient_profile.get('name', 'Patient')}")
        print(f"Age: {patient_profile.get('age', '?')}, Occupation: {patient_profile.get('occupation', '?')}")
        print(f"Primary Concern: {patient_profile.get('primary_concern', '?')}")
        print(f"Max Turns: {max_turns}")
        print(f"Supervisor Feedback: {'Enabled' if enable_supervisor else 'Disabled'}")
        if extra_meta:
            print(f"Extra: {extra_meta}")
        print(f"{'='*80}\n")

    # Create and run pipeline (skips Agent 2 entirely when supervisor disabled).
    # `scenario_meta` is consumed by TherapyPipeline to tag the session JSON.
    pipeline = TherapyPipeline(
        reddit_posts=reddit_posts,
        patient_profile=patient_profile,
        output_dir=output_dir,
        max_turns=max_turns,
        enable_supervisor=enable_supervisor,
        shared_model=shared_model,
        therapist_model=therapist_model,
        therapist_llm=therapist_llm,
        supervisor_endpoint=supervisor_endpoint,
        scenario_meta={"scenario": scenario_name, **extra_meta},
    )

    # Run the session
    results = pipeline.run_session(
        enable_supervisor=enable_supervisor,
        enable_suggestions=enable_suggestions,
        verbose=verbose
    )
    
    # Print statistics
    if verbose:
        stats = pipeline.get_session_stats()
        print(f"\n{'='*80}")
        print("SESSION STATISTICS")
        print(f"{'='*80}")
        print(f"Total Exchanges: {stats['total_transcript_entries']}")
        print(f"Therapist Turns: {stats['therapist_turns']}")
        print(f"Patient Turns: {stats['patient_turns']}")
        print(f"Avg Therapist Response: {stats['avg_therapist_response_length']} words")
        print(f"Avg Patient Response: {stats['avg_patient_response_length']} words")
        print(f"Supervisor Feedbacks: {stats['supervisor_feedbacks']}")
        print(f"{'='*80}\n")
        
        print(f"\nResults saved to:")
        print(f"  - Full transcript: {results['transcript_file']}")
        print(f"  - Summary: {results['summary_file']}\n")


def _run_one_scenario(
    spec: Dict[str, Any],
    max_turns: int,
    enable_supervisor: bool,
    enable_suggestions: bool,
    verbose: bool,
    output_dir: str,
    shared_model: str,
    therapist_model: str,
    therapist_llm: HFTherapist,
    supervisor_endpoint: Optional[str],
) -> None:
    """Run a single scenario. Pulled out so it can be the unit of work for the
    ThreadPoolExecutor in run_many_scenarios()."""
    run_therapy_session(
        spec=spec,
        max_turns=max_turns,
        enable_supervisor=enable_supervisor,
        enable_suggestions=enable_suggestions,
        verbose=verbose,
        output_dir=output_dir,
        shared_model=shared_model,
        therapist_model=therapist_model,
        therapist_llm=therapist_llm,
        supervisor_endpoint=supervisor_endpoint,
    )


def run_many_scenarios(
    scenarios: List[Any],
    therapist_llm: HFTherapist,
    max_turns: int = 8,
    enable_supervisor: bool = True,
    enable_suggestions: bool = True,
    verbose: bool = True,
    output_dir: str = "sessions",
    shared_model: str = DEFAULT_SHARED_MODEL,
    therapist_model: str = "Qwen/Qwen3-4B-Instruct-2507",
    supervisor_endpoint: Optional[str] = None,
    concurrency: int = 1,
) -> None:
    """Run a list of scenarios in a single Python process.

    All scenarios share the cached vLLM clients (see shared_vllm.get_shared_vllm)
    so models load exactly once across the whole batch — the per-scenario cost is
    just one therapy session, no cold-start. Conversation state (transcript,
    supervisor checklist) is fresh per scenario because we build a new
    TherapyPipeline each time.

    When `concurrency > 1`, scenarios are dispatched to a ThreadPoolExecutor.
    Both vLLM backends (local LLM and the remote HTTP server) handle concurrent
    requests with continuous batching, so the per-session calls overlap on the
    GPU rather than serializing. session_id is collision-proof (microseconds +
    random suffix in TherapyPipeline.__init__).
    """
    failed: List[str] = []
    concurrency = max(1, int(concurrency))

    # Normalize: each element may be a bare scenario name (str) or a full spec dict.
    specs: List[Dict[str, Any]] = []
    for item in scenarios:
        if isinstance(item, str):
            specs.append(_spec_from_named_scenario(item))
        else:
            specs.append(item)

    if concurrency == 1:
        bar = tqdm(specs, desc="scenarios", unit="scenario", disable=verbose, leave=True)
        for spec in bar:
            bar.set_postfix_str(spec["name"])
            try:
                _run_one_scenario(
                    spec, max_turns, enable_supervisor, enable_suggestions,
                    verbose, output_dir, shared_model, therapist_model,
                    therapist_llm, supervisor_endpoint,
                )
            except Exception as e:
                failed.append(spec["name"])
                tqdm.write(f"[run_many_scenarios] scenario '{spec['name']}' failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        # Parallel mode — verbose printing would interleave wildly, force quiet.
        if verbose:
            print(f"[run_many_scenarios] forcing quiet mode for concurrency={concurrency}")
            verbose = False
        bar = tqdm(total=len(specs), desc=f"scenarios (×{concurrency})",
                   unit="scenario", leave=True)
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {
                ex.submit(
                    _run_one_scenario,
                    spec, max_turns, enable_supervisor, enable_suggestions,
                    verbose, output_dir, shared_model, therapist_model,
                    therapist_llm, supervisor_endpoint,
                ): spec
                for spec in specs
            }
            for fut in as_completed(futures):
                spec = futures[fut]
                scenario = spec["name"]
                try:
                    fut.result()
                    bar.set_postfix_str(f"done: {scenario}")
                except Exception as e:
                    failed.append(scenario)
                    tqdm.write(f"[run_many_scenarios] scenario '{scenario}' failed: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    bar.update(1)
        bar.close()

    if failed:
        print(f"\n[run_many_scenarios] {len(failed)}/{len(scenarios)} failed: {failed}")
    else:
        print(f"\n[run_many_scenarios] all {len(scenarios)} scenarios completed")


def run_interactive_mode() -> None:
    """Run the pipeline in interactive mode."""
    print("\n" + "="*80)
    print("THERAPEUTIC AGENT PIPELINE - INTERACTIVE MODE")
    print("="*80 + "\n")
    
    print("Available scenarios:")
    scenarios = list_scenarios()
    for i, scenario in enumerate(scenarios, 1):
        print(f"  {i}. {scenario}")
    
    # Get user input
    try:
        choice = int(input("\nSelect scenario (number): "))
        if 1 <= choice <= len(scenarios):
            scenario = scenarios[choice - 1]
        else:
            print("Invalid choice!")
            return
    except ValueError:
        print("Invalid input!")
        return
    
    # Get parameters
    turns = input("\nMax turns (default 8): ")
    max_turns = int(turns) if turns.isdigit() else 8
    
    supervisor = input("Enable supervisor feedback? (y/n, default y): ").lower() != 'n'
    suggestions = input("Show suggestions in real-time? (y/n, default y): ").lower() != 'n'
    
    # Run session
    run_therapy_session(
        scenario=scenario,
        max_turns=max_turns,
        enable_supervisor=supervisor,
        enable_suggestions=suggestions,
        verbose=True,
        output_dir="sessions"
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Therapeutic Agent Pipeline - Three-agent therapy simulation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all scenarios (one process, vLLM loaded once)
  python run.py --max-turns 20

  # Run a subset
  python run.py --scenarios anxiety_workplace depression_isolation --max-turns 12

  # Run with no supervisor
  python run.py --scenarios grief_loss --no-supervisor

  # List available scenarios
  python run.py --list-scenarios
        """
    )
    
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=None,
        choices=list_scenarios(),
        help="Scenarios to run (one or more). vLLM models stay loaded across "
             "scenarios — only conversation state resets between them. "
             "If omitted, runs ALL scenarios."
    )

    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=None,
        help="Run only the first N scenarios from the selected set. Useful for quick "
             "smoke tests (e.g. --n-scenarios 2)."
    )

    parser.add_argument(
        "--scenarios-from",
        type=str,
        default=None,
        help="Path to a CSV with columns post_id,title,body,category (e.g. "
             "multiturn_reddit_data/splits/test.csv). When set, scenarios are "
             "loaded from this CSV instead of the hardcoded names in reddit_posts.py."
    )

    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="When --scenarios-from is set, restrict to these category values "
             "(e.g. --categories anxiety depression). Default: all categories in the CSV."
    )

    parser.add_argument(
        "--n-per-category",
        type=int,
        default=None,
        help="When --scenarios-from is set, take the first N rows from each category "
             "(after filtering). E.g. --n-per-category 20 → 20 per category."
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Run this many scenarios in parallel via a thread pool. All workers share "
             "the same vLLM clients; vLLM batches their concurrent calls on the GPU. "
             "Forces --quiet (verbose prints from parallel sessions would interleave). "
             "Default: 1 (sequential)."
    )
    
    parser.add_argument(
        "--max-turns",
        type=int,
        default=8,
        help="Maximum number of conversation turns (default: 8)"
    )
    
    parser.add_argument(
        "--no-supervisor",
        action="store_true",
        help="Disable supervisor feedback"
    )
    
    parser.add_argument(
        "--no-suggestions",
        action="store_true",
        help="Don't print suggestions during session"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sessions",
        help="Directory to save session transcripts (default: sessions)"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )
    
    parser.add_argument(
        "--shared-model",
        type=str,
        default=DEFAULT_SHARED_MODEL,
        help=f"HF model id for Patient + Supervisor (shared vLLM). Default: {DEFAULT_SHARED_MODEL}"
    )

    parser.add_argument(
        "--therapist-model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HF model id for Agent 1 (primary therapist — the model under evaluation). "
             "Loaded as a separate vLLM client. Default: Qwen/Qwen3-4B-Instruct-2507"
    )

    parser.add_argument(
        "--supervisor-endpoint",
        type=str,
        default=None,
        help="If set, route Patient + Supervisor through a remote vLLM OpenAI-compatible "
             "server at this URL (e.g. http://localhost:8001/v1) instead of loading in-process. "
             "Lets the supervisor model run in a different Python env from the therapist."
    )

    parser.add_argument(
        "--therapist-endpoint",
        type=str,
        default=None,
        help="If set, route Agent 1 (therapist) through a remote vLLM OpenAI-compatible "
             "server at this URL (e.g. http://127.0.0.1:8002/v1) instead of loading the HF "
             "model in-process. Use this for ~5-10x faster therapist generation via "
             "vLLM continuous batching. --therapist-model must match what the server "
             "registered as --served-model-name."
    )

    args = parser.parse_args()
    
    # Handle list scenarios
    if args.list_scenarios:
        print("\nAvailable Scenarios:")
        for scenario in list_scenarios():
            profile = get_patient_profile(scenario)
            print(f"  • {scenario}")
            print(f"    Patient: {profile['name']}")
            print(f"    Concern: {profile['primary_concern']}\n")
        return
    
    # Handle interactive mode
    if args.interactive:
        run_interactive_mode()
        return
    
    # Build the scenarios list. Two paths:
    #   (a) --scenarios-from CSV → load specs from CSV, optionally filtered by
    #       --categories and capped by --n-per-category.
    #   (b) named scenarios from reddit_posts.py — --scenarios A B (default: all).
    if args.scenarios_from:
        if args.scenarios:
            print("[warn] --scenarios is ignored when --scenarios-from is set.",
                  file=sys.stderr)
        scenarios = load_specs_from_csv(
            csv_path=args.scenarios_from,
            categories=args.categories,
            n_per_category=args.n_per_category,
        )
        if not scenarios:
            print(f"[error] no scenarios loaded from {args.scenarios_from} "
                  f"(categories={args.categories}, n_per_category={args.n_per_category})",
                  file=sys.stderr)
            sys.exit(2)
        cats_count = {}
        for s in scenarios:
            c = s["extra_meta"].get("category") or "(none)"
            cats_count[c] = cats_count.get(c, 0) + 1
        print(f"[info] loaded {len(scenarios)} scenarios from {args.scenarios_from}: "
              f"{cats_count}")
    else:
        scenarios = list(args.scenarios) if args.scenarios else list_scenarios()

    if args.n_scenarios is not None:
        if args.n_scenarios <= 0:
            print(f"[error] --n-scenarios must be positive, got {args.n_scenarios}", file=sys.stderr)
            sys.exit(2)
        scenarios = scenarios[:args.n_scenarios]
        names = [s if isinstance(s, str) else s["name"] for s in scenarios]
        print(f"[info] limiting to first {len(scenarios)} scenario(s): {names}")

    # ---- Load the therapist (Agent 1) ONCE for the whole run ---------------
    # Two backends:
    #   (a) --therapist-endpoint set → RemoteVLLM (HTTP client to an external
    #       vLLM OpenAI server). Continuous batching => parallel sessions
    #       actually parallelize. ~5-10x faster than HFTherapist.
    #   (b) otherwise → HFTherapist in-process (transformers.generate). Has
    #       a per-process lock so concurrent sessions serialize on it.
    if args.therapist_endpoint:
        from shared_vllm import RemoteVLLM  # local import to avoid HF-only paths
        endpoint = args.therapist_endpoint.rstrip("/")
        if not endpoint.endswith("/v1"):
            endpoint = endpoint + "/v1"
        print(f"[therapist] using remote vLLM at {endpoint}  model='{args.therapist_model}'")
        therapist_llm = RemoteVLLM(base_url=endpoint, model_name=args.therapist_model)
    else:
        # GPU pinning is the caller's job (CUDA_VISIBLE_DEVICES) — from inside this
        # process the only visible GPU is always cuda:0, so we hardcode it.
        print(f"[therapist] loading HF model in-process '{args.therapist_model}'")
        therapist_llm = HFTherapist(model_path=args.therapist_model, device="cuda:0")
    print("[therapist] ready.")

    run_many_scenarios(
        scenarios=scenarios,
        therapist_llm=therapist_llm,
        max_turns=args.max_turns,
        enable_supervisor=not args.no_supervisor,
        enable_suggestions=not args.no_suggestions,
        verbose=not args.quiet,
        output_dir=args.output_dir,
        shared_model=args.shared_model,
        therapist_model=args.therapist_model,
        supervisor_endpoint=args.supervisor_endpoint,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()
