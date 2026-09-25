#!/usr/bin/env python3
"""Native MLX smoke evaluation; browser latency must be measured separately."""
import argparse
import json
from pathlib import Path
import re
import time

from prepare_dataset import read_system_prompt, to_messages
from prepare_notebooks import extract

PROBES = [
    "I keep putting off the things I actually want to do because",
    "When I finally have a free afternoon I want to",
    "I thought moving somewhere new would",
    "The thing I miss most about being a kid is",
    "Sometimes I wonder whether my friends",
    "If I stopped worrying about what people think I would",
    "Today I felt proud of myself because",
    "I changed my mind when I realized",
    "I don't know whether being useful is the same as",
    "When I pay attention to someone else's suffering I",
    "I want to be honest with myself even when",
    "The ordinary things around me suddenly seem",
    "I can accept uncertainty without",
    "I used to mistake being busy for",
    "I feel most free when I",
    "After the argument I kept thinking about",
]


def words(text):
    return re.findall(r"[a-z']+", text.lower().replace("’", "'"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='data/models/lfm2.5-350m-base')
    parser.add_argument('--adapter')
    parser.add_argument('--test-loss', action='store_true', help='Use only after checkpoint selection')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    model, tokenizer = load(args.model, adapter_path=args.adapter)
    test_losses = {}
    if args.test_loss:
        from mlx_lm.tuner.datasets import ChatDataset, CacheDataset
        from mlx_lm.tuner.trainer import evaluate
        test_rows = [json.loads(s) for s in Path('data/notebooks/test.jsonl').read_text().splitlines()]
        provenance = [json.loads(s) for s in Path('data/notebooks/test.provenance.jsonl').read_text().splitlines()]
        for author in ('weil', 'camus'):
            subset = [r for r, p in zip(test_rows, provenance) if p['author'] == author]
            dataset = CacheDataset(ChatDataset(subset, tokenizer, mask_prompt=True))
            test_losses[author] = float(evaluate(model, dataset, batch_size=1, num_batches=-1, max_seq_length=512))
            print(f'{author} test loss: {test_losses[author]:.4f}', flush=True)
    system = read_system_prompt(Path(__file__).resolve().parents[2])
    source_ngrams = set()
    for epub in Path('data').glob('*.epub'):
        for paragraph in extract(epub):
            tokens = words(paragraph['text'])
            source_ngrams.update(tuple(tokens[i:i+10]) for i in range(len(tokens)-9))
    rows = []
    targets = set()
    for name in ('train', 'valid', 'test'):
        for line in Path(f'data/notebooks/{name}.provenance.jsonl').read_text().splitlines():
            target = tuple(words(json.loads(line)['target']))
            if len(target) >= 6:
                targets.add(target)
    for index, fragment in enumerate(PROBES):
        messages = to_messages(system, fragment, '')['messages'][:-1]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        mx.random.seed(100 + index)
        start = time.perf_counter()
        output = generate(model, tokenizer, prompt=prompt, max_tokens=32,  # the notebooks preset's runtime cap
                          sampler=make_sampler(temp=.4, top_p=.88, top_k=20), verbose=False)
        tokens = words(output)
        row = {'fragment': fragment, 'output': output,
               'native_seconds': time.perf_counter()-start,
               'exact_source_target_match_6plus': tuple(tokens) in targets,
               'source_10gram_match': any(tuple(tokens[i:i+10]) in source_ngrams for i in range(len(tokens)-9))}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'model':args.model, 'adapter':args.adapter, 'test_losses':test_losses, 'results':rows,
                                  'note':'Sampled native MLX outputs, no prefix cache or app sanitizer. 10-word match check is only a memorization screen, not proof of originality.'}, indent=2, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
