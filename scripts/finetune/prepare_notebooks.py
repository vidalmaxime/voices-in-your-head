#!/usr/bin/env python3
"""Prepare the supplied English Weil/Camus EPUB editions for completion SFT.

No network access. Source prose and derived examples are written only to data/.
Selection is edition-specific: fail if the expected titles/body sections differ.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import xml.etree.ElementTree as ET
import zipfile

from prepare_dataset import read_system_prompt, to_messages, transcript_style


def tag(element):
    return element.tag.rsplit("}", 1)[-1]


def prose(element):
    """Drop note markers without dropping adjacent words or punctuation."""
    parts = [element.text or ""]
    for child in element:
        excluded = (tag(child) in {"sup", "aside"}
                    or child.get("role") == "doc-noteref"
                    or "fnref" in child.get("class", "").split())
        if not excluded:
            parts.append(prose(child))
        parts.append(child.tail or "")
    return "".join(parts)


def normalize(text):
    return re.sub(r"\s+", " ", text.replace("’", "'").replace("\xad", "")).strip()


def extract(path):
    with zipfile.ZipFile(path) as archive:
        container = ET.fromstring(archive.read("META-INF/container.xml"))
        opf = next(e.get("full-path") for e in container.iter() if tag(e) == "rootfile")
        root = ET.fromstring(archive.read(opf))
        titles = [e.text or "" for e in root.iter() if tag(e) == "title"]
        if "The Notebooks of Simone Weil" in titles:
            author = "weil"
            body = {"ch1.xhtml", "ch2.xhtml"}
        elif "The Complete Notebooks" in titles:
            author = "camus"
            body = {f"chi-camus-notebooks-{i:04}.xhtml" for i in range(7, 24)}
        else:
            raise ValueError(f"Unrecognized edition: {path.name}: {titles}")
        items = {e.get("id"): e.get("href") for e in root.iter() if tag(e) == "item"}
        spine = [items[e.get("idref")] for e in root.iter() if tag(e) == "itemref"]
        if not body.issubset(set(spine)):
            raise ValueError(f"Body sections missing from {path.name}")
        for href in spine:
            if href not in body:
                continue
            chapter = ET.fromstring(archive.read(str(Path(opf).parent / href)))
            for index, element in enumerate(e for e in chapter.iter() if tag(e) == "p"):
                classes = set(element.get("class", "").split())
                if classes & {"ed", "edf", "edl", "doi", "byLine", "center", "auto"}:
                    continue
                text = normalize(prose(element))
                if not text or text.startswith(('“', '"', '«')):
                    continue
                yield {"author": author, "section": href, "paragraph": index,
                       "group": f"{author}/{href}/{index // 80}", "text": text}


# Target bounds follow the runtime: the notebooks preset generates up to 32
# tokens, so a tail keeps one token free for EOS.
TARGET_WORDS = (8, 24)
TARGET_TOKENS = 31
SENTENCE_WORDS = (10, 50)
CUTS_PER_SENTENCE = 1
# Greek, Hebrew, and mathematical notation in Weil's notebooks are not speech.
NON_LATIN = re.compile(r"[^\x00-\x7F\u00C0-\u024F\u2010-\u2027]")


def candidates(text, tokenizer, target_words=TARGET_WORDS, target_tokens=TARGET_TOKENS,
               sentence_words=SENTENCE_WORDS):
    """Only complete sentence tails: never truncate a target to fit the cap."""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not sentence.endswith((".", "!", "?")):
            continue
        # Quotation marks of either kind flag someone else's words. An ellipsis
        # is not a split point, so a tail could span two thoughts.
        if any(c in sentence for c in '“”"«»‘‛[]…'):
            continue
        # Notebook entries need not name a speaker: impersonal aphorisms and
        # observations are the authors' own voice too. Only addressing a
        # listener is excluded, since the app's speaker talks to themselves.
        if re.search(r"\b(you|your|yours)\b", sentence, re.I):
            continue
        if NON_LATIN.search(sentence) or sentence.count("(") != sentence.count(")"):
            continue
        words = sentence.split()
        if not sentence_words[0] <= len(words) <= sentence_words[1]:
            continue
        # Reading lists, page references, and formulas are mostly numbers.
        if sum(1 for w in words if re.search(r"[A-Za-z]", w)) < 0.85 * len(words):
            continue
        cuts = []
        for cut in range(5, len(words) - target_words[0] + 1):
            tail = " ".join(words[cut:])
            if not target_words[0] <= len(words) - cut <= target_words[1]:
                continue
            if len(tokenizer.encode(tail, add_special_tokens=False)) > target_tokens:
                continue
            if words[cut - 1].endswith(('.', '!', '?', ';', ':')):
                continue
            # Spoken punctuation creates word boundaries ("happiness—we"),
            # not concatenated tokens ("happinesswe").
            fragment = transcript_style(re.sub(r"[—–/:;()]", " ", " ".join(words[:cut])).split())
            if len(fragment.split()) >= 5:
                cuts.append((fragment, tail))
        if cuts:
            yield sentence, cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/notebooks"))
    ap.add_argument("--model", default="data/models/lfm2.5-350m-base")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cuts-per-sentence", type=int, default=CUTS_PER_SENTENCE,
                    help="distinct cursor positions kept per sentence")
    ap.add_argument("--balance-authors", action="store_true",
                    help="downsample the larger author's training rows to match the smaller")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    system = read_system_prompt(Path(__file__).resolve().parents[2])
    rng = random.Random(args.seed)
    rows, seen, seen_fragments, sources = [], set(), set(), []
    for path in sorted(args.source.glob("*.epub")):
        sources.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for paragraph in extract(path):
            for sentence, cuts in candidates(paragraph["text"], tokenizer):
                key = re.sub(r"\W+", "", sentence.lower())
                if key in seen:
                    continue
                seen.add(key)
                # Several cursor positions per sentence multiply the examples
                # without synthetic text; the passage grouping below keeps every
                # cut of a sentence in the same split.
                rng.shuffle(cuts)
                for fragment, target in cuts[:args.cuts_per_sentence]:
                    if fragment in seen_fragments:
                        continue
                    seen_fragments.add(fragment)
                    row = {**paragraph, "fragment": fragment, "target": target}
                    row.pop("text")
                    row["messages"] = to_messages(system, fragment, target)["messages"]
                    if len(tokenizer.apply_chat_template(row["messages"], tokenize=True)) <= 512:
                        rows.append(row)
    if {r["author"] for r in rows} != {"weil", "camus"}:
        raise ValueError("Both authors must contribute usable examples")
    # Contiguous passage blocks stay together. Exact sentence/fragment duplicates
    # were removed globally before assigning any train/validation/test groups.
    splits = {k: [] for k in ("train", "valid", "test")}
    for author in ("weil", "camus"):
        groups = sorted({r["group"] for r in rows if r["author"] == author})
        if len(groups) < 10:
            raise ValueError(f"Too few independent passage groups for {author}")
        rng.shuffle(groups)
        n = max(1, round(len(groups) * .1))
        membership = {g: "valid" if i < n else "test" if i < 2*n else "train"
                      for i, g in enumerate(groups)}
        for row in rows:
            if row["author"] == author:
                splits[membership[row["group"]]].append(row)
    # Every usable sentence trains by default. Optional equal representation
    # drops the larger author's rows rather than duplicating the smaller's.
    before_balance = dict(Counter(r["author"] for r in splits["train"]))
    if args.balance_authors:
        count = min(before_balance.values())
        balanced = []
        for author in ("weil", "camus"):
            subset = [r for r in splits["train"] if r["author"] == author]
            rng.shuffle(subset)
            balanced.extend(subset[:count])
        splits["train"] = balanced
    if min(before_balance.values()) < 100:
        raise ValueError(f"Too few training pairs: {before_balance}")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
        rng.shuffle(subset)
        (args.out / f"{name}.jsonl").write_text("".join(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n" for r in subset))
        (args.out / f"{name}.provenance.jsonl").write_text("".join(json.dumps({k:v for k,v in r.items() if k != "messages"}, ensure_ascii=False) + "\n" for r in subset))
    report = {"seed": args.seed, "sources": sources, "cuts_per_sentence": args.cuts_per_sentence,
              "balance_authors": args.balance_authors,
              "target_words": list(TARGET_WORDS), "target_tokens": TARGET_TOKENS,
              "train_before_balance": before_balance,
              "splits": {k:dict(Counter(r["author"] for r in v)) for k,v in splits.items()},
              "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
              "notes": "English translations; automatic attribution filters are imperfect. Holdouts are contiguous 80-paragraph blocks, not entire books. Targets 8-24 words and <=31 LFM tokens; no chopped targets. One cursor position per sentence unless --cuts-per-sentence is raised."}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
