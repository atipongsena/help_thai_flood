"""
Utility script to turn manually collected flood-relief posts into a
structured dataset that can be used for model training or triage dashboards.

Usage:
    python scripts/extract_manual_posts.py \
        --input data/manual_posts.txt \
        --output data/manual_posts_structured.csv
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from typing import Dict, List

import pandas as pd

from utils.config import DATA_DIR, FLOOD_HASHTAGS, URGENCY_KEYWORDS
from utils.preprocessing import (
    basic_clean,
    calculate_urgency_score,
    extract_coordinates,
    extract_location_line,
    extract_phone_numbers,
    has_flood_hashtags,
)

from utils.risk_tags import (
    infer_risk_flags,
    decide_priority,
    infer_resource_tags,
    summarize_context_reason,
    serialize_flags,
    serialize_tags,
    extract_people_counts,
    extract_duration_hours,
)


def parse_posts(raw_text: str, min_chars: int) -> List[str]:
    """Split a raw text file into individual posts."""
    posts: List[str] = []
    buffer: List[str] = []

    for line in raw_text.splitlines():
        if line.strip():
            buffer.append(line.rstrip())
        else:
            if buffer:
                joined = "\n".join(buffer).strip()
                if len(joined) >= min_chars:
                    posts.append(joined)
                buffer = []

    if buffer:
        joined = "\n".join(buffer).strip()
        if len(joined) >= min_chars:
            posts.append(joined)

    return posts


def build_record(text: str, idx: int) -> Dict[str, object]:
    """Convert a raw text post into a structured dictionary."""
    phones = extract_phone_numbers(text)
    coords = extract_coordinates(text)
    location_line = extract_location_line(text)
    hashtags = has_flood_hashtags(text, FLOOD_HASHTAGS)
    urgency_score = calculate_urgency_score(text, URGENCY_KEYWORDS)
    risk_flags = infer_risk_flags(text)
    priority, auto_label = decide_priority(urgency_score, risk_flags)
    resource_tags = infer_resource_tags(text)
    context_reason = summarize_context_reason(text, risk_flags)
    people_counts = extract_people_counts(text)
    duration_hours = extract_duration_hours(text)
    total_people = sum(people_counts.values())

    return {
        "post_id": idx + 1,
        "source": "manual",
        "text": text,
        "text_clean": basic_clean(text),
        "phones": json.dumps(phones, ensure_ascii=False),
        "location_line": location_line or "",
        "lat": coords["lat"] if coords else None,
        "lng": coords["lng"] if coords else None,
        "hashtags": json.dumps(hashtags, ensure_ascii=False),
        "urgency_score": round(urgency_score, 3),
        "priority": priority,
        "auto_label": auto_label,
        "risk_flags": json.dumps(risk_flags, ensure_ascii=False),
        "risk_flags_active": serialize_flags(risk_flags),
        "resource_tags": serialize_tags(resource_tags),
        "context_reason": context_reason,
        "num_children": people_counts["children"],
        "num_elderly": people_counts["elderly"],
        "num_adults": people_counts["adults"],
        "num_unknown_people": people_counts["unknown"],
        "num_people_total": total_people,
        "duration_hours": round(duration_hours, 2) if duration_hours else None,
        "extracted_at": datetime.utcnow().isoformat(),
    }


def print_summary(df: pd.DataFrame) -> None:
    """Print quick aggregate stats to help triage teams."""
    print("\n=== Summary ===")
    print(f"Total posts: {len(df)}")
    if "priority" in df.columns:
        print("\nPriority counts:")
        for priority, count in df["priority"].value_counts().items():
            print(f"  {priority}: {count}")

    if "auto_label" in df.columns:
        print("\nAuto label distribution:")
        for label, count in df["auto_label"].value_counts().items():
            print(f"  {label}: {count}")

    # Risk flags
    try:
        flag_dicts = df["risk_flags"].apply(json.loads)
        flag_counter = Counter()
        for record in flag_dicts:
            for flag, value in record.items():
                if value:
                    flag_counter[flag] += 1
        if flag_counter:
            print("\nTop risk flags:")
            for flag, count in flag_counter.most_common(10):
                print(f"  {flag}: {count}")
    except Exception:
        pass

    if "resource_tags" in df.columns:
        try:
            res_counter = Counter()
            for tags in df["resource_tags"]:
                for tag in tags.split("|"):
                    if tag:
                        res_counter[tag] += 1
            if res_counter:
                print("\nResource tags:")
                for tag, count in res_counter.most_common():
                    print(f"  {tag}: {count}")
        except Exception:
            pass

    if "num_people_total" in df.columns:
        avg_people = df["num_people_total"].replace({0: None}).dropna()
        if not avg_people.empty:
            print(f"\nAverage people per post (non-zero): {avg_people.mean():.1f}")
    if "duration_hours" in df.columns:
        durations = df["duration_hours"].replace({0: None}).dropna()
        if not durations.empty:
            print(f"Average reported duration without aid: {durations.mean():.1f} hrs")

    # Hashtags
    try:
        hashtag_counter = Counter()
        for tags_json in df["hashtags"]:
            tags = json.loads(tags_json)
            hashtag_counter.update(tags)
        if hashtag_counter:
            print("\nTop hashtags:")
            for tag, count in hashtag_counter.most_common(10):
                print(f"  #{tag}: {count}")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured fields from manual flood-relief posts."
    )
    parser.add_argument(
        "--input",
        "-i",
        default=os.path.join(DATA_DIR, "manual_posts.txt"),
        help="Path to the raw manual posts text file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(DATA_DIR, "manual_posts_structured.csv"),
        help="Output CSV path for structured data.",
    )
    parser.add_argument(
        "--jsonl",
        help="Optional JSONL output path (one record per line).",
    )
    parser.add_argument(
        "--parquet",
        help="Optional Parquet output path for faster downstream loading.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print aggregate statistics after extraction.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=40,
        help="Minimum character length for a chunk to be treated as a post.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop exact-duplicate posts after parsing.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with open(args.input, encoding="utf-8") as f:
        raw_text = f.read()

    posts = parse_posts(raw_text, min_chars=args.min_chars)
    if args.dedupe:
        deduped = []
        seen = set()
        for post in posts:
            if post not in seen:
                deduped.append(post)
                seen.add(post)
        posts = deduped

    records = [build_record(text, idx) for idx, text in enumerate(posts)]

    if not records:
        print("No posts were parsed. Try lowering --min-chars.")
        return

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} structured posts to {args.output}")

    if args.jsonl:
        os.makedirs(os.path.dirname(args.jsonl), exist_ok=True)
        with open(args.jsonl, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Also wrote JSONL output to {args.jsonl}")

    if args.parquet:
        try:
            os.makedirs(os.path.dirname(args.parquet), exist_ok=True)
            df.to_parquet(args.parquet, index=False)
            print(f"Also wrote Parquet output to {args.parquet}")
        except Exception as exc:
            print(f"Warning: could not write Parquet file: {exc}")

    if args.summary:
        print_summary(df)


if __name__ == "__main__":
    main()

