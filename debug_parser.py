import re
from voice.voice_utils import _word_to_int, parse_voice_command

def debug(t):
    t = t.lower()
    print('\nINPUT:', t)
    m = re.search(r"(?:nodes|node count)\s*(?:to|=|is)?\s*(\d{1,3})", t)
    print('m1', m)
    m2 = re.search(r"(\d{1,3})\s+nodes\b", t)
    print('m2', m2)
    m3 = re.search(r"([a-z\s-]{3,30})\s+nodes\b", t)
    print('m3', m3, 'group:', m3.group(1) if m3 else None)
    tokens = re.findall(r"\b[a-z]+\b", t)
    print('tokens', tokens)
    if 'nodes' in tokens:
        last_idx = len(tokens) - 1 - tokens[::-1].index('nodes')
        print('last_idx', last_idx)
        for start in range(max(0, last_idx - 3), last_idx):
            candidate = ' '.join(tokens[start:last_idx])
            print('candidate', candidate, '->', _word_to_int(candidate))


for s in ['Set nodes to 15 and edge density to 30 percent', 'Run BFS from node 3', 'Create twenty nodes', 'Create twenty five nodes']:
    debug(s)
    print('parse_voice_command ->', parse_voice_command(s))
