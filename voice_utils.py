import re

def parse_voice_command(text):
    """
    Parse voice command text into visualization parameters.
    Handles phrases like:
    - "use 10 nodes"
    - "set connectivity to 0.5"
    - "change algorithm to Dijkstra"
    - "set start node to 3"
    - "set end node to 7"
    
    Returns a dict with recognized parameters.
    """
    params = {}
    text = text.lower().strip()
    
    # Node count
    node_match = re.search(r'(?:use|set|with)\s+(\d+)\s+nodes?', text)
    if node_match:
        params['nodes'] = node_match.group(1)
    
    # Connectivity
    conn_match = re.search(r'(?:set\s+)?connectivity\s+(?:to\s+)?([0-9]*\.?[0-9]+)', text)
    if conn_match:
        params['connectivity'] = conn_match.group(1)
    
    # Algorithm selection
    algo_patterns = {
        r'(?:use|run|change\s+to|switch\s+to)\s+dijkstra': "Dijkstra's Shortest Path",
        r'(?:use|run|change\s+to|switch\s+to)\s+(?:bfs|breadth[\s-]?first)': "Breadth-First Search (BFS)",
        r'(?:use|run|change\s+to|switch\s+to)\s+(?:dfs|depth[\s-]?first)': "Depth-First Search (DFS)",
        r'(?:use|run|change\s+to|switch\s+to)\s+prim': "Prim's Minimum Spanning Tree"
    }
    
    for pattern, algo_name in algo_patterns.items():
        if re.search(pattern, text):
            params['algorithm'] = algo_name
            break
    
    # Start node
    start_match = re.search(r'(?:set\s+)?start\s+(?:node\s+)?(?:to\s+)?(\d+)', text)
    if start_match:
        params['start_node'] = start_match.group(1)
    
    # End node
    end_match = re.search(r'(?:set\s+)?end\s+(?:node\s+)?(?:to\s+)?(\d+)', text)
    if end_match:
        params['end_node'] = end_match.group(1)
    
    return params