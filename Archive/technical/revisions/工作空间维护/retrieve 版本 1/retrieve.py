#!/usr/bin/env python3
"""Read-only, in-memory retrieval over the workspace's active documentation."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
TOKEN = re.compile(r'[a-z0-9_]+|[\u3400-\u9fff]+', re.I)


def tokens(text):
    result = []
    for match in TOKEN.findall(text.lower()):
        if '\u3400' <= match[0] <= '\u9fff':
            result.extend(match[i:i + 2] for i in range(max(1, len(match) - 1)))
        else:
            result.append(match)
    return result


def sources(root):
    paths = [root / 'AGENTS.md', root / 'README.md', root / 'config/perception_tasks.json',
             root / 'lab/README.md', root / 'ws/README.md', root / 'ws/src/aec_vision/README.md']
    for folder in ('docs', 'lab/features'):
        paths.extend((root / folder).rglob('*.md'))
    return sorted(set(p for p in paths if p.is_file() and not p.is_symlink()))


def chunks(path, root):
    relative = path.relative_to(root).as_posix()
    content = path.read_text(encoding='utf-8')
    if relative == 'config/perception_tasks.json':
        data = json.loads(content)
        for task in data['tasks']:
            ident = task['id']
            match = re.search(r'"id"\s*:\s*' + re.escape(json.dumps(ident)), content)
            line = content.count('\n', 0, match.start()) + 1 if match else 1
            yield relative, line, 'task ' + ident, json.dumps(task, ensure_ascii=False, indent=2)
        return
    lines = content.splitlines()
    heading = ''
    block, start = [], 1
    for number, line in enumerate(lines, 1):
        if line.startswith('#') or (not line.strip() and block) or len(block) >= 16:
            if block:
                yield relative, start, heading, '\n'.join(block)
                block = []
            if line.startswith('#'):
                heading = line.lstrip('# ').strip()
                continue
        if line.strip():
            if not block:
                start = number
            block.append(line)
    if block:
        yield relative, start, heading, '\n'.join(block)


def search(root, query, limit):
    corpus, errors = [], []
    for path in sources(root):
        try:
            for item in chunks(path, root):
                terms = Counter(tokens(item[2] + ' ' + item[3]))
                if terms:
                    corpus.append((item, terms))
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
            errors.append('{}: {}'.format(path.relative_to(root), exc))
    words = set(tokens(query))
    if not words or not corpus:
        return [], errors
    postings = {word: [] for word in words}
    for index, (_, terms) in enumerate(corpus):
        for word in words.intersection(terms):
            postings[word].append(index)
    candidates = set(index for indices in postings.values() for index in indices)
    freq = {word: len(indices) for word, indices in postings.items()}
    average = sum(sum(terms.values()) for _, terms in corpus) / len(corpus)
    ranked = []
    for index in candidates:
        item, terms = corpus[index]
        length = sum(terms.values())
        score = sum(math.log(1 + (len(corpus) - freq[w] + .5) / (freq[w] + .5)) *
                    (terms[w] * 2.2 / (terms[w] + 1.2 * (.25 + .75 * length / average)))
                    for w in words if terms[w])
        if score:
            ranked.append((score, item))
    ranked.sort(key=lambda result: (-result[0], result[1][0], result[1][1]))
    return ranked[:limit], errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query', help='Chinese or English search terms')
    parser.add_argument('--limit', type=int, default=5, choices=range(1, 21))
    args = parser.parse_args()
    results, errors = search(ROOT, args.query, args.limit)
    for score, (path, line, heading, body) in results:
        print('\n{}:{} [{}] score={:.2f}\n{}'.format(path, line, heading, score, body[:1600]))
    for error in errors:
        print('Could not index ' + error, file=sys.stderr)
    if not results:
        print('No lexical matches; try different terms or inspect sources directly.', file=sys.stderr)
    return 2 if errors else (0 if results else 1)


if __name__ == '__main__':
    sys.exit(main())
