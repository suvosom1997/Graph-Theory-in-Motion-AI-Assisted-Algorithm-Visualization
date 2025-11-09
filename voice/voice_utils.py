import re
from typing import Dict, Optional


def _word_to_int(word: str) -> Optional[int]:
    # Minimal mapping for common small numbers (extendable)
    small = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
        'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
        'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
        'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
    }
    word = word.lower().strip()
    if word.isdigit():
        return int(word)
    # handle simple compound words like 'twenty five'
    parts = word.split()
    total = 0
    for p in parts:
        if p in small:
            total += small[p]
        else:
            return None
    return total if total != 0 or parts[0] == 'zero' else None


def parse_voice_command(text: str) -> Dict:
    """Parse a spoken command into visualization parameters.

    Returns a dict with any of the keys: nodes, connectivity, algorithm,
    start_node, end_node, action
    """
    result = {}
    if not text:
        return result

    t = text.lower()

    # Algorithm detection
    alg_map = {
        'dijkstra': "Dijkstra's Shortest Path",
        'shortest': "Dijkstra's Shortest Path",
        'bfs': "Breadth-First Search (BFS)",
        'breadth': "Breadth-First Search (BFS)",
        'dfs': "Depth-First Search (DFS)",
        'depth': "Depth-First Search (DFS)",
        'prim': "Prim's Minimum Spanning Tree",
        'mst': "Prim's Minimum Spanning Tree",
    }
    for k, v in alg_map.items():
        if re.search(rf"\b{k}\b", t):
            result['algorithm'] = v
            break

    # Start/End node detection (do this early to avoid clashing with generic numbers)
    m = re.search(r"(?:start|from) node\s*(?:number\s*)?(\d{1,3})", t)
    if m:
        result['start_node'] = int(m.group(1))
    else:
        m2 = re.search(r"from node\s*([a-z\s-]+)", t)
        if m2:
            wordnum = _word_to_int(m2.group(1))
            if wordnum is not None:
                result['start_node'] = wordnum

    m = re.search(r"(?:end|to) node\s*(?:number\s*)?(\d{1,3})", t)
    if m:
        result['end_node'] = int(m.group(1))

    # Nodes: look for phrases like '20 nodes' or 'twenty nodes' or 'nodes to 20'
    # prefer explicit plural 'nodes' or phrase 'node count' to avoid matching 'start node' or 'from node'
    m = re.search(r"(?:nodes|node count)\s*(?:to|=|is)?\s*(\d{1,3})", t)
    if m:
        result['Nodes'] = int(m.group(1))
        result['nodes'] = int(m.group(1))
    else:
        # number before the word 'nodes' (e.g. '20 nodes')
        m2 = re.search(r"(\d{1,3})\s+nodes\b", t)
        if m2:
            result['nodes'] = int(m2.group(1))
        else:
            # word-numbers before 'nodes' (e.g. 'twenty nodes' or 'twenty five nodes')
            m3 = re.search(r"([a-z\s-]{3,30})\s+nodes\b", t)
            if m3:
                val = _word_to_int(m3.group(1).strip())
                if val is not None:
                    result['nodes'] = val
                else:
                    # fallback: if m3 matched but word->int failed, look back tokens before 'nodes'
                    tokens = re.findall(r"\b[a-z]+\b", t)
                    if 'nodes' in tokens:
                        last_idx = len(tokens) - 1 - tokens[::-1].index('nodes')
                        # look back up to 3 tokens before 'nodes'
                        for start in range(max(0, last_idx - 3), last_idx):
                            candidate = ' '.join(tokens[start:last_idx])
                            val = _word_to_int(candidate)
                            if val is not None:
                                result['nodes'] = val
                                break

    # Edge density / connectivity as percent
    m = re.search(r"(\d{1,3})\s*(?:%|percent)\b", t)
    if m:
        percent = int(m.group(1))
        if 0 <= percent <= 100:
            result['connectivity'] = round(percent / 100.0, 2)
    else:
        # phrases like 'edge density 0.3' or 'connectivity 0.3'
        m2 = re.search(r"(?:edge density|connectivity)\s*(?:to|is)?\s*(0?\.?\d{1,3})", t)
        if m2:
            try:
                val = float(m2.group(1))
                if 0 <= val <= 1:
                    result['connectivity'] = val
            except ValueError:
                pass

    # Start/End node detection
    m = re.search(r"(?:start|from) node\s*(?:number\s*)?(\d{1,3})", t)
    if m:
        result['start_node'] = int(m.group(1))
    else:
        m2 = re.search(r"from node\s*([a-z\s-]+)", t)
        if m2:
            wordnum = _word_to_int(m2.group(1))
            if wordnum is not None:
                result['start_node'] = wordnum

    m = re.search(r"(?:end|to) node\s*(?:number\s*)?(\d{1,3})", t)
    if m:
        result['end_node'] = int(m.group(1))

    # Action detection
    if re.search(r"\b(run|start|execute)\b", t):
        result['action'] = 'run'
    elif re.search(r"\b(reset|randomize|shuffle)\b", t):
        result['action'] = 'reset'
    elif re.search(r"\b(pause|stop)\b", t):
        result['action'] = 'pause'

    return result
