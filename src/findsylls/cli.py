"""Command line interface for findsylls.

Provides two main subcommands:
  segment  - run syllable segmentation for one or more audio files
  evaluate - batch evaluate segmentation against TextGrids

Example usage:
  findsylls segment input.wav --envelope sbs --method peaks_and_valleys --out out.json
  findsylls evaluate "data/**/*.wav" "data/**/*.TextGrid" --phone-tier 2 --syllable-tier 1 --word-tier 0 \
      --envelope hilbert --out results.csv
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import List

from .pipeline.pipeline import segment_audio, run_evaluation
from .pipeline.results import aggregate_results


def _positive_float(value: str) -> float:
    try:
        f = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Not a float: {value}")
    if f <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0")
    return f


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="findsylls", description="Unsupervised syllable segmentation & evaluation")
    sub = p.add_subparsers(dest="command", required=True)

    # segment subcommand
    seg = sub.add_parser("segment", help="Segment one or more audio files")
    seg.add_argument("audio", nargs="+", help="Input audio file(s) (wav/flac)")
    seg.add_argument("--envelope", default="sbs", help="Envelope method (rms|hilbert|lowpass|sbs|gammatone|theta)")
    seg.add_argument("--method", default="peaks_and_valleys", help="Segmentation method")
    seg.add_argument("--samplerate", type=int, default=16000, help="Target sample rate")
    seg.add_argument("--out", required=True, help="Output JSON file")
    seg.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")

    # evaluate subcommand
    ev = sub.add_parser("evaluate", help="Batch evaluate against TextGrids")
    ev.add_argument("wav_glob", help="Glob for audio files")
    ev.add_argument("textgrid_glob", help="Glob for TextGrid files")
    ev.add_argument("--phone-tier", type=int, default=2, help="Phone tier index (0-based)")
    ev.add_argument("--syllable-tier", type=int, default=None, help="Syllable tier index or omit for None")
    ev.add_argument("--word-tier", type=int, default=None, help="Word tier index or omit for None")
    ev.add_argument("--tolerance", type=_positive_float, default=0.05, help="Boundary tolerance (s)")
    ev.add_argument("--envelope", default="sbs", help="Envelope method")
    ev.add_argument("--method", default="peaks_and_valleys", help="Segmentation method")
    ev.add_argument("--suffix-strip", default=None, help="Suffix to strip from TextGrid basenames before matching")
    ev.add_argument("--out", required=True, help="Output CSV path for flattened results")
    ev.add_argument("--aggregate", help="Optional CSV path for aggregated (F1 etc.) results")

    return p


def cmd_segment(args: argparse.Namespace) -> int:
    out_records = []
    for audio_path in args.audio:
        try:
            syllables, envelope, times = segment_audio(
                audio_path,
                samplerate=args.samplerate,
                envelope_fn=args.envelope,
                segment_fn=args.method,
            )
        except Exception as e:  # pragma: no cover - CLI convenience
            print(f"Error processing {audio_path}: {e}", file=sys.stderr)
            continue
        rec = {
            "audio_file": audio_path,
            "syllables": [(float(s), float(p), float(e)) for (s, p, e) in syllables],
            "envelope": args.envelope,
            "segmentation": args.method,
        }
        out_records.append(rec)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(out_records, f, indent=2)
        else:
            json.dump(out_records, f)
    print(f"Wrote {len(out_records)} records -> {out_path}")
    return 0 if out_records else 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    df = run_evaluation(
        textgrid_paths=args.textgrid_glob,
        wav_paths=args.wav_glob,
        phone_tier=args.phone_tier,
        syllable_tier=args.syllable_tier,
        word_tier=args.word_tier,
        tolerance=args.tolerance,
        envelope_fn=args.envelope,
        segmentation_fn=args.method,
        tg_suffix_to_strip=args.suffix_strip,
    )
    if df.empty:
        print("No results produced (check globs and tier indices)", file=sys.stderr)
        return 1
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Flattened results -> {out_path} ({len(df)} rows)")
    if args.aggregate:
        agg = aggregate_results(df, dataset_name="dataset")
        agg_path = Path(args.aggregate); agg_path.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(agg_path, index=False)
        print(f"Aggregate results -> {agg_path}")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "segment":
        return cmd_segment(args)
    if args.command == "evaluate":
        return cmd_evaluate(args)
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
