#!/usr/bin/env python3
"""Builds a thought-completion SFT dataset from an Apple Notes markdown export.

Each training pair is (fragment -> continuation), cut mid-sentence at a natural
connector, formatted with the app's own system prompt and user template so the
finetuned model sees exactly what the runtime sends. Fragments are normalized
the way Parakeet transcripts arrive (lowercase, no punctuation), because that
is the distribution the model completes at runtime.

Uses the local LFM2.5 tokenizer to enforce the browser's output and context
limits. Requires transformers. Output stays out of git: it is personal data.

Usage:
  python3 scripts/finetune/prepare_dataset.py \
      --notes /path/to/apple-notes-export \
      --out data/personal-lfm \
      [--exclude "Dwarkesh*" --exclude "*quotes*"] [--max-per-file 80]

Then eyeball the per-file stats it prints and re-run with --exclude for files
that are mostly pasted content rather than your own writing.
"""

import argparse
import fnmatch
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------- cleaning

MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
BARE_URL = re.compile(r"https?://\S+")
MD_HEADER = re.compile(r"^#{1,6}\s.*$", re.M)
MD_EMPHASIS = re.compile(r"[*_]{1,3}([^*_]+)[*_]{1,3}")
BULLET = re.compile(r"^\s*[-*+]\s+", re.M)

# The tail of a URL-heavy note is other people's prose more often than not.
QUOTE_LINE = re.compile(r"^\s*>", re.M)

ENGLISH_STOPWORDS = frozenset(
    "the a an and or but if of to in on for with is are was were be been i my "
    "me we our you it this that not no so at as by from have has had do does "
    "what when how why there here they them".split()
)

FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'll|i'd|my|me|myself|we|our|us)\b", re.I)

# Mid-sentence cut points that mirror how live utterances trail off.
CONNECTORS = frozenset(
    "because is are was that to and but so when if about like how why "
    "what where which while for with without than then".split()
)


def clean_markdown(text: str) -> str:
    text = MD_IMAGE.sub(" ", text)
    text = MD_LINK.sub(r"\1", text)
    text = BARE_URL.sub(" ", text)
    text = MD_HEADER.sub(" ", text)
    text = QUOTE_LINE.sub(" ", text)
    text = MD_EMPHASIS.sub(r"\1", text)
    text = BULLET.sub("", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return text


def sentences(text: str):
    for raw in re.split(r"(?<=[.!?])\s+|\n+", text):
        s = re.sub(r"\s+", " ", raw).strip()
        if s:
            yield s


def looks_english(words) -> bool:
    if not words:
        return False
    hits = sum(1 for w in words if w.lower().strip(".,!?;:'\"") in ENGLISH_STOPWORDS)
    return hits / len(words) >= 0.18


def transcript_style(words) -> str:
    """Lowercase, punctuation-free, the way cleanTranscript output arrives."""
    out = []
    for w in words:
        w = re.sub(r"[^A-Za-z0-9']+", "", w.lower())
        if w:
            out.append(w)
    return " ".join(out)


def continuation_text(words) -> str:
    text = " ".join(words).strip()
    text = re.sub(r'^[\s"\'`.,:;!?()\[\]{}-]+', "", text)
    text = re.sub(r"[\s,;:]+$", "", text)
    # First word lowercase: it continues the speaker's sentence.
    return text[:1].lower() + text[1:] if text else text


def pair_from_sentence(sentence: str, rng: random.Random, tokenizer):
    words = sentence.split()
    if not (9 <= len(words) <= 40):
        return None
    if not looks_english(words):
        return None
    if not FIRST_PERSON.search(sentence):
        return None
    # Reject leftovers that are mostly symbols or numbers.
    alpha = sum(1 for w in words if re.search(r"[A-Za-z]", w))
    if alpha / len(words) < 0.8:
        return None

    cuts = [
        i
        for i in range(4, len(words) - 4)
        if words[i].lower().strip(".,!?;:'\"") in CONNECTORS
        and 5 <= i + 1 <= 18
        and 4 <= len(words) - i - 1 <= 24
    ]
    if not cuts:
        return None
    rng.shuffle(cuts)
    for index in cuts:
        cut = index + 1  # keep the connector in the fragment
        fragment = transcript_style(words[:cut])
        target = continuation_text(words[cut:])
        if len(fragment.split()) < 5 or len(target.split()) < 4:
            continue
        # Leave one token for EOS within the browser's 32-token cap.
        # Reject long tails rather than teaching truncated sentences.
        if len(tokenizer.encode(target, add_special_tokens=False)) > 31:
            continue
        return fragment, target
    return None


# ------------------------------------------------------- runtime templates


def read_system_prompt(repo_root: Path) -> str:
    """Reads COMPLETION_SYSTEM_PROMPT out of completion-logic.ts so training
    and runtime never drift."""
    source = (repo_root / "app/llm/completion-logic.ts").read_text()
    match = re.search(r"COMPLETION_SYSTEM_PROMPT = `([^`]+)`", source)
    if not match:
        sys.exit("Could not find COMPLETION_SYSTEM_PROMPT in completion-logic.ts")
    return match.group(1).strip()


def to_messages(system_prompt: str, fragment: str, target: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Text before cursor: {fragment}\nText after cursor:",
            },
            {"role": "assistant", "content": target},
        ]
    }


# ----------------------------------------------------------------- driver


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", required=True, type=Path)
    ap.add_argument("--out", default=Path("data/personal-lfm"), type=Path)
    ap.add_argument("--model", default="data/models/lfm2.5-350m-base",
                    help="local LFM2.5 model/tokenizer directory")
    ap.add_argument("--exclude", action="append", default=[], help="filename glob to skip")
    ap.add_argument("--max-per-file", type=int, default=80,
                    help="cap pairs per note so one huge note cannot dominate")
    ap.add_argument("--valid-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if not 0 < args.valid_fraction < 1:
        ap.error("--valid-fraction must be between 0 and 1")
    if args.max_per_file < 1:
        ap.error("--max-per-file must be positive")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    rng = random.Random(args.seed)
    repo_root = Path(__file__).resolve().parents[2]
    system_prompt = read_system_prompt(repo_root)

    per_file = Counter()
    pairs = []
    seen_fragments = set()

    files = sorted(args.notes.rglob("*.md"))
    if not files:
        sys.exit(f"No .md files under {args.notes}")

    for path in files:
        if any(fnmatch.fnmatch(path.name, pat) for pat in args.exclude):
            continue
        text = clean_markdown(path.read_text(errors="ignore"))
        file_pairs = []
        for sentence in sentences(text):
            pair = pair_from_sentence(sentence, rng, tokenizer)
            if pair is None:
                continue
            fragment, target = pair
            messages = to_messages(system_prompt, fragment, target)["messages"]
            if len(tokenizer.apply_chat_template(messages, tokenize=True)) > 512:
                continue
            if fragment in seen_fragments:
                continue
            seen_fragments.add(fragment)
            file_pairs.append(pair)
        rng.shuffle(file_pairs)
        file_pairs = file_pairs[: args.max_per_file]
        per_file[path.name] = len(file_pairs)
        pairs.extend(file_pairs)

    rng.shuffle(pairs)
    if len(pairs) < 2:
        sys.exit("Need at least two usable pairs for nonempty training and validation splits.")
    n_valid = min(len(pairs) - 1, max(8, int(len(pairs) * args.valid_fraction)))
    valid, train = pairs[:n_valid], pairs[n_valid:]

    args.out.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train", train), ("valid", valid)):
        with open(args.out / f"{name}.jsonl", "w") as f:
            for fragment, target in subset:
                f.write(json.dumps(to_messages(system_prompt, fragment, target)) + "\n")

    print(f"train: {len(train)}  valid: {len(valid)}  (from {len(files)} notes)")
    print("\nTop contributing notes (eyeball these; --exclude the pasted-content ones):")
    for name, count in per_file.most_common(15):
        if count:
            print(f"  {count:4d}  {name}")
    print("\nSample pairs:")
    for fragment, target in train[:5]:
        print(f"  [{fragment}] -> [{target}]")


if __name__ == "__main__":
    main()
