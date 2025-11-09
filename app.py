import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
from collections import deque
import tempfile
import sounddevice as sd
import soundfile as sf
import time
from voice.voice_utils import parse_voice_command

# Set page config for a wider layout
st.set_page_config(layout="wide", page_title="Graph Theory in Motion")

# Initialize session state variables if they don't exist
if 'algorithm_running' not in st.session_state:
    st.session_state.algorithm_running = False
if 'algorithm_steps' not in st.session_state:
    st.session_state.algorithm_steps = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'last_narrated_step' not in st.session_state:
    st.session_state.last_narrated_step = None
if 'voice_params_to_apply' not in st.session_state:
    st.session_state.voice_params_to_apply = None
if 'apply_voice_success' not in st.session_state:
    st.session_state.apply_voice_success = None
if 'apply_voice_errors' not in st.session_state:
    st.session_state.apply_voice_errors = None

# Apply any pending voice parameters BEFORE widget creation
if st.session_state.voice_params_to_apply:
    params = st.session_state.voice_params_to_apply
    # Reset so we don't re-apply on next rerun
    st.session_state.voice_params_to_apply = None
    
    # Try applying parsed parameters
    applied = []
    errors = []

    # Nodes (accept various capitalizations)
    node_val = None
    if 'nodes' in params:
        node_val = params.get('nodes')
    elif 'Nodes' in params:
        node_val = params.get('Nodes')
    elif 'node_count' in params:
        node_val = params.get('node_count')

    if node_val is not None:
        try:
            st.session_state.num_nodes = int(node_val)
            applied.append(f"num_nodes={int(node_val)}")
        except Exception as e:
            errors.append(f"nodes: {e}")

    # Connectivity
    if 'connectivity' in params:
        try:
            st.session_state.connectivity = float(params['connectivity'])
            applied.append(f"connectivity={float(params['connectivity'])}")
        except Exception as e:
            errors.append(f"connectivity: {e}")
    elif 'edge_density' in params:
        try:
            st.session_state.connectivity = float(params['edge_density'])
            applied.append(f"connectivity={float(params['edge_density'])}")
        except Exception as e:
            errors.append(f"edge_density: {e}")

    # Algorithm
    if 'algorithm' in params:
        algo_val = params['algorithm']
        algo_options = ["Dijkstra's Shortest Path", "Breadth-First Search (BFS)", 
                       "Depth-First Search (DFS)", "Prim's Minimum Spanning Tree"]
        chosen = None
        if algo_val in algo_options:
            chosen = algo_val
        else:
            # try fuzzy matching by lowercasing
            for opt in algo_options:
                if algo_val.lower() in opt.lower() or opt.lower() in algo_val.lower():
                    chosen = opt
                    break
        if chosen:
            st.session_state.algorithm = chosen
            applied.append(f"algorithm={chosen}")

    # Start / End nodes
    if 'start_node' in params:
        try:
            st.session_state.start = int(params['start_node'])
            applied.append(f"start={int(params['start_node'])}")
        except Exception as e:
            errors.append(f"start_node: {e}")

    if 'end_node' in params:
        try:
            st.session_state.end = int(params['end_node'])
            applied.append(f"end={int(params['end_node'])}")
        except Exception as e:
            errors.append(f"end_node: {e}")

    st.session_state.apply_voice_success = applied if applied else None
    st.session_state.apply_voice_errors = errors if errors else None

# Load OpenAI API key from environment and initialize client
load_dotenv()
client = None
try:
    client = OpenAI()  # will use OPENAI_API_KEY from environment
except Exception as e:
    st.warning("OpenAI client not initialized (needed for voice and AI explanations)")
    print(f"OpenAI init error: {e}")

def get_ai_narration(algorithm_name, step_description, graph_info):
    """Get AI narration for algorithm steps"""
    if not client:
        return "AI narration unavailable - OpenAI API key not configured"
    
    try:
        prompt = f"""
        You are explaining the {algorithm_name} algorithm step by step. 
        Current step: {step_description}
        Graph context: {graph_info}
        
        Provide a clear, concise explanation (2-3 sentences) of what's happening in this step 
        and why it's important for the algorithm. Use simple language suitable for learning.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI narration error: {str(e)}"

def create_graph(graph_type, num_nodes, connectivity):
    """Create different types of graphs"""
    try:
        if graph_type == "Random":
            G = nx.erdos_renyi_graph(num_nodes, connectivity, seed=42)
        elif graph_type == "Small World":
            k = max(2, int(num_nodes * connectivity))
            if k >= num_nodes:
                k = num_nodes - 1
            G = nx.watts_strogatz_graph(num_nodes, k, 0.3, seed=42)
        elif graph_type == "Scale-Free":
            m = max(1, int(num_nodes * connectivity / 2))
            G = nx.barabasi_albert_graph(num_nodes, m, seed=42)
        elif graph_type == "Complete":
            G = nx.complete_graph(num_nodes)
        elif graph_type == "Grid":
            side = int(np.sqrt(num_nodes))
            G = nx.grid_2d_graph(side, side)
            G = nx.convert_node_labels_to_integers(G)
        else:
            G = nx.erdos_renyi_graph(num_nodes, connectivity, seed=42)
        
        # Add random weights to edges
        np.random.seed(42)
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.randint(1, 20)
        
        return G
    except Exception as e:
        st.error(f"Error creating graph: {e}")
        return nx.erdos_renyi_graph(5, 0.3, seed=42)

def get_node_positions(G, layout_type):
    """Get node positions based on layout type"""
    try:
        if layout_type == "Spring":
            return nx.spring_layout(G, seed=42, k=2)
        elif layout_type == "Circular":
            return nx.circular_layout(G)
        elif layout_type == "Random":
            return nx.random_layout(G, seed=42)
        elif layout_type == "Kamada-Kawai":
            if len(G.nodes()) > 2:
                return nx.kamada_kawai_layout(G)
            else:
                return nx.spring_layout(G, seed=42)
        else:
            return nx.spring_layout(G, seed=42)
    except Exception as e:
        st.error(f"Error creating layout: {e}")
        return nx.spring_layout(G, seed=42)

def get_color_gradient(value, max_value):
    """Generate color based on distance value"""
    if value == float('infinity'):
        return 'rgb(100, 100, 100)'
    
    normalized = value / max_value if max_value > 0 else 0
    r = int(255 * normalized)
    g = int(255 * (1 - normalized))
    b = 50
    
    return f'rgb({r}, {g}, {b})'

def dijkstra_algorithm(G, start_node, end_node):
    """Dijkstra's algorithm with detailed edge-by-edge steps"""
    distances = {node: float('infinity') for node in G.nodes()}
    distances[start_node] = 0
    previous = {node: None for node in G.nodes()}
    unvisited = set(G.nodes())
    visited = set()
    
    steps = []
    steps.append({
        'step_num': 0,
        'description': f" Starting at node {start_node}",
        'narration': f"We begin our journey at node {start_node} with distance 0.",
        'distances': distances.copy(),
        'current_node': start_node,
        'visited': set(),
        'exploring_edge': None,
        'path': [],
        'finished': False
    })
    
    step_num = 1
    while unvisited and step_num < 200:
        current_node = min(unvisited, key=lambda node: distances[node])
        
        if distances[current_node] == float('infinity'):
            break
        
        steps.append({
            'step_num': step_num,
            'description': f"📍 Visiting node {current_node}",
            'narration': f"Now examining node {current_node} which has distance {distances[current_node]} from the start.",
            'distances': distances.copy(),
            'current_node': current_node,
            'visited': visited.copy(),
            'exploring_edge': None,
            'updated_neighbors': [],
            'path': [],
            'finished': False
        })
        step_num += 1
        
        unvisited.remove(current_node)
        visited.add(current_node)
        
        updated_neighbors = []
        for neighbor in G.neighbors(current_node):
            if neighbor in unvisited:
                weight = G[current_node][neighbor]['weight']
                old_distance = distances[neighbor]
                new_distance = distances[current_node] + weight
                
                edge = (current_node, neighbor)
                steps.append({
                    'step_num': step_num,
                    'description': f" Checking edge {current_node} → {neighbor}",
                    'narration': f"Exploring edge from {current_node} to {neighbor} with weight {weight}. Current distance to {neighbor} is {old_distance if old_distance != float('infinity') else '∞'}. New distance would be {distances[current_node]} + {weight} = {new_distance}.",
                    'distances': distances.copy(),
                    'current_node': current_node,
                    'visited': visited.copy(),
                    'exploring_edge': edge,
                    'updated_neighbors': updated_neighbors.copy(),
                    'path': [],
                    'finished': False
                })
                step_num += 1
                
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current_node
                    updated_neighbors.append(neighbor)
                    
                    steps.append({
                        'step_num': step_num,
                        'description': f" Updated node {neighbor}",
                        'narration': f"Found a shorter path! Updated distance to node {neighbor} from {old_distance if old_distance != float('infinity') else '∞'} to {new_distance}.",
                        'distances': distances.copy(),
                        'current_node': current_node,
                        'visited': visited.copy(),
                        'exploring_edge': edge,
                        'updated_neighbors': [neighbor],
                        'path': [],
                        'finished': False
                    })
                    step_num += 1
                else:
                    steps.append({
                        'step_num': step_num,
                        'description': f" No improvement for {neighbor}",
                        'narration': f"The new distance {new_distance} is not better than the current distance {old_distance if old_distance != float('infinity') else '∞'}, so we keep the old value.",
                        'distances': distances.copy(),
                        'current_node': current_node,
                        'visited': visited.copy(),
                        'exploring_edge': edge,
                        'updated_neighbors': [],
                        'path': [],
                        'finished': False
                    })
                    step_num += 1
        
        if current_node == end_node:
            path = []
            current = end_node
            while current is not None:
                path.append(current)
                current = previous[current]
            path.reverse()
            
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            
            steps.append({
                'step_num': step_num,
                'description': f"🎉 Destination reached!",
                'narration': f"We've reached the destination node {end_node}! The shortest path is: {' → '.join(map(str, path))} with total distance {distances[end_node]}.",
                'distances': distances.copy(),
                'current_node': current_node,
                'visited': visited.copy(),
                'exploring_edge': None,
                'path': path,
                'path_edges': path_edges,
                'finished': True
            })
            break
    
    return steps

def bfs_algorithm(G, start_node, end_node):
    """Breadth-First Search with detailed edge-by-edge steps"""
    visited = set()
    queue = deque([start_node])
    parent = {start_node: None}
    
    steps = []
    steps.append({
        'step_num': 0,
        'description': f" Starting BFS at node {start_node}",
        'narration': f"Beginning Breadth-First Search at node {start_node}. We'll explore level by level.",
        'current_node': start_node,
        'visited': set(),
        'queue': [start_node],
        'exploring_edge': None,
        'path': [],
        'finished': False
    })
    
    step_num = 1
    while queue and step_num < 200:
        current_node = queue.popleft()
        
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        steps.append({
            'step_num': step_num,
            'description': f" Visiting node {current_node}",
            'narration': f"Dequeued and visiting node {current_node}. Now exploring its neighbors.",
            'current_node': current_node,
            'visited': visited.copy(),
            'queue': list(queue),
            'exploring_edge': None,
            'path': [],
            'finished': False
        })
        step_num += 1
        
        if current_node == end_node:
            path = []
            current = end_node
            while current is not None:
                path.append(current)
                current = parent.get(current)
            path.reverse()
            
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            
            steps.append({
                'step_num': step_num,
                'description': f" Found destination!",
                'narration': f"Reached node {end_node}! Path: {' → '.join(map(str, path))}",
                'current_node': current_node,
                'visited': visited.copy(),
                'queue': list(queue),
                'exploring_edge': None,
                'path': path,
                'path_edges': path_edges,
                'finished': True
            })
            break
        
        for neighbor in G.neighbors(current_node):
            edge = (current_node, neighbor)
            
            steps.append({
                'step_num': step_num,
                'description': f"Checking edge {current_node} → {neighbor}",
                'narration': f"Examining neighbor {neighbor} of node {current_node}.",
                'current_node': current_node,
                'visited': visited.copy(),
                'queue': list(queue),
                'exploring_edge': edge,
                'path': [],
                'finished': False
            })
            step_num += 1
            
            if neighbor not in visited and neighbor not in queue:
                queue.append(neighbor)
                parent[neighbor] = current_node
                
                steps.append({
                    'step_num': step_num,
                    'description': f" Added {neighbor} to queue",
                    'narration': f"Node {neighbor} is unvisited, adding it to the queue for later exploration.",
                    'current_node': current_node,
                    'visited': visited.copy(),
                    'queue': list(queue),
                    'exploring_edge': edge,
                    'path': [],
                    'finished': False
                })
                step_num += 1
            else:
                steps.append({
                    'step_num': step_num,
                    'description': f" Skip {neighbor} (already seen)",
                    'narration': f"Node {neighbor} has already been visited or is in the queue.",
                    'current_node': current_node,
                    'visited': visited.copy(),
                    'queue': list(queue),
                    'exploring_edge': edge,
                    'path': [],
                    'finished': False
                })
                step_num += 1
    
    return steps

def dfs_algorithm(G, start_node, end_node):
    """Depth-First Search with detailed edge-by-edge steps"""
    visited = set()
    stack = [start_node]
    parent = {start_node: None}
    
    steps = []
    steps.append({
        'step_num': 0,
        'description': f" Starting DFS at node {start_node}",
        'narration': f"Beginning Depth-First Search at node {start_node}. We'll explore as deep as possible before backtracking.",
        'current_node': start_node,
        'visited': set(),
        'stack': [start_node],
        'exploring_edge': None,
        'path': [],
        'finished': False
    })
    
    step_num = 1
    while stack and step_num < 200:
        current_node = stack.pop()
        
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        steps.append({
            'step_num': step_num,
            'description': f" Visiting node {current_node}",
            'narration': f"Popped and visiting node {current_node} from the stack.",
            'current_node': current_node,
            'visited': visited.copy(),
            'stack': list(stack),
            'exploring_edge': None,
            'path': [],
            'finished': False
        })
        step_num += 1
        
        if current_node == end_node:
            path = []
            current = end_node
            while current is not None:
                path.append(current)
                current = parent.get(current)
            path.reverse()
            
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            
            steps.append({
                'step_num': step_num,
                'description': f"🎉 Found destination!",
                'narration': f"Reached node {end_node}! Path: {' → '.join(map(str, path))}",
                'current_node': current_node,
                'visited': visited.copy(),
                'stack': list(stack),
                'exploring_edge': None,
                'path': path,
                'path_edges': path_edges,
                'finished': True
            })
            break
        
        for neighbor in G.neighbors(current_node):
            edge = (current_node, neighbor)
            
            steps.append({
                'step_num': step_num,
                'description': f" Checking edge {current_node} → {neighbor}",
                'narration': f"Examining neighbor {neighbor} of node {current_node}.",
                'current_node': current_node,
                'visited': visited.copy(),
                'stack': list(stack),
                'exploring_edge': edge,
                'path': [],
                'finished': False
            })
            step_num += 1
            
            if neighbor not in visited and neighbor not in stack:
                stack.append(neighbor)
                parent[neighbor] = current_node
                
                steps.append({
                    'step_num': step_num,
                    'description': f" Pushed {neighbor} to stack",
                    'narration': f"Node {neighbor} is unvisited, pushing it to the stack for deep exploration.",
                    'current_node': current_node,
                    'visited': visited.copy(),
                    'stack': list(stack),
                    'exploring_edge': edge,
                    'path': [],
                    'finished': False
                })
                step_num += 1
            else:
                steps.append({
                    'step_num': step_num,
                    'description': f"Skip {neighbor} (already seen)",
                    'narration': f"Node {neighbor} has already been visited or is in the stack.",
                    'current_node': current_node,
                    'visited': visited.copy(),
                    'stack': list(stack),
                    'exploring_edge': edge,
                    'path': [],
                    'finished': False
                })
                step_num += 1
    
    return steps

def prims_algorithm(G, start_node):
    """Prim's Minimum Spanning Tree with detailed steps"""
    if len(G.nodes()) == 0:
        return []
    
    mst_edges = []
    visited = set([start_node])
    
    steps = []
    steps.append({
        'step_num': 0,
        'description': f"🎯 Starting Prim's MST at node {start_node}",
        'narration': f"Beginning Prim's algorithm at node {start_node}. Building a Minimum Spanning Tree.",
        'current_node': start_node,
        'visited': visited.copy(),
        'mst_edges': [],
        'exploring_edge': None,
        'finished': False
    })
    
    step_num = 1
    while len(visited) < len(G.nodes()) and step_num < 200:
        min_edge = None
        min_weight = float('infinity')
        
        for node in visited:
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    weight = G[node][neighbor]['weight']
                    edge = (node, neighbor)
                    
                    steps.append({
                        'step_num': step_num,
                        'description': f"🔍 Considering edge {node} → {neighbor}",
                        'narration': f"Checking edge from {node} to {neighbor} with weight {weight}.",
                        'current_node': node,
                        'visited': visited.copy(),
                        'mst_edges': mst_edges.copy(),
                        'exploring_edge': edge,
                        'finished': False
                    })
                    step_num += 1
                    
                    if weight < min_weight:
                        min_weight = weight
                        min_edge = edge
        
        if min_edge is None:
            break
        
        mst_edges.append(min_edge)
        visited.add(min_edge[1])
        
        steps.append({
            'step_num': step_num,
            'description': f"Added edge {min_edge[0]} → {min_edge[1]} to MST",
            'narration': f"Selected edge {min_edge[0]} → {min_edge[1]} with weight {min_weight}. Total MST weight so far: {sum(G[e[0]][e[1]]['weight'] for e in mst_edges)}.",
            'current_node': min_edge[1],
            'visited': visited.copy(),
            'mst_edges': mst_edges.copy(),
            'path_edges': mst_edges.copy(),
            'exploring_edge': min_edge,
            'finished': False
        })
        step_num += 1
    
    total_weight = sum(G[e[0]][e[1]]['weight'] for e in mst_edges)
    steps.append({
        'step_num': step_num,
        'description': f"🎉 MST Complete!",
        'narration': f"Minimum Spanning Tree completed with {len(mst_edges)} edges and total weight {total_weight}.",
        'current_node': None,
        'visited': visited.copy(),
        'mst_edges': mst_edges.copy(),
        'path_edges': mst_edges.copy(),
        'exploring_edge': None,
        'finished': True
    })
    
    return steps

def visualize_graph_animated(G, pos, step=None, animation_style="modern"):
    """Create animated visualization with multiple style options"""
    try:
        current_node = step.get('current_node') if step else None
        visited = step.get('visited', set()) if step else set()
        updated_neighbors = step.get('updated_neighbors', []) if step else []
        path_edges = step.get('path_edges', []) if step else []
        exploring_edge = step.get('exploring_edge') if step else None
        distances = step.get('distances', {}) if step else {}
        
        finite_distances = [d for d in distances.values() if d != float('infinity')]
        max_distance = max(finite_distances) if finite_distances else 1
        
        traces = []
        
        # Background edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        traces.append(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color='rgba(150, 150, 150, 0.3)'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Path edges
        if path_edges:
            path_x = []
            path_y = []
            for edge in path_edges:
                if edge in G.edges() or (edge[1], edge[0]) in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    path_x.extend([x0, x1, None])
                    path_y.extend([y0, y1, None])
            
            if path_x:
                traces.append(go.Scatter(
                    x=path_x, y=path_y,
                    line=dict(width=6, color='rgba(255, 215, 0, 0.8)'),
                    hoverinfo='none',
                    mode='lines',
                    name='Shortest Path',
                    showlegend=False
                ))
        
        # General exploration edges
        if current_node is not None and not exploring_edge:
            explore_x = []
            explore_y = []
            for neighbor in G.neighbors(current_node):
                x0, y0 = pos[current_node]
                x1, y1 = pos[neighbor]
                explore_x.extend([x0, x1, None])
                explore_y.extend([y0, y1, None])
            
            if explore_x:
                traces.append(go.Scatter(
                    x=explore_x, y=explore_y,
                    line=dict(width=3, color='rgba(255, 165, 0, 0.5)'),
                    hoverinfo='none',
                    mode='lines',
                    showlegend=False
                ))
        
        # Specific edge being explored
        if exploring_edge:
            edge = exploring_edge
            if edge in G.edges() or (edge[1], edge[0]) in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                
                traces.append(go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    line=dict(width=8, color='rgba(0, 255, 255, 1)'),
                    hoverinfo='none',
                    mode='lines',
                    showlegend=False
                ))
                
                mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                weight = G[edge[0]][edge[1]].get('weight', 1)
                
                traces.append(go.Scatter(
                    x=[mid_x],
                    y=[mid_y],
                    mode='markers+text',
                    text=[f'⚡{weight}'],
                    textposition='top center',
                    textfont=dict(size=16, color='cyan'),
                    marker=dict(size=15, color='cyan', symbol='circle'),
                    hoverinfo='none',
                    showlegend=False
                ))
        
        # Edge weight labels
        edge_labels_x = []
        edge_labels_y = []
        edge_labels_text = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_labels_x.append((x0 + x1) / 2)
            edge_labels_y.append((y0 + y1) / 2)
            weight = G[edge[0]][edge[1]].get('weight', 1)
            edge_labels_text.append(str(weight))
        
        if len(G.edges()) < 30:
            traces.append(go.Scatter(
                x=edge_labels_x, y=edge_labels_y,
                mode='text',
                text=edge_labels_text,
                textfont=dict(size=10, color='white'),
                textposition='middle center',
                hoverinfo='none',
                showlegend=False
            ))
        
        # Nodes
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        node_line_widths = []
        node_line_colors = []
        hover_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            if node == current_node:
                node_colors.append('rgba(255, 50, 50, 1)')
                node_sizes.append(50)
                node_line_widths.append(6)
                node_line_colors.append('rgba(255, 255, 0, 1)')
                node_text.append(f'<b>{node}</b>')
                hover_info = f'<b>CURRENT NODE: {node}</b><br>Distance: {distances.get(node, "∞")}'
            elif node in updated_neighbors:
                node_colors.append('rgba(255, 200, 0, 0.9)')
                node_sizes.append(35)
                node_line_widths.append(4)
                node_line_colors.append('rgba(255, 140, 0, 1)')
                node_text.append(f'<b>{node}</b>')
                hover_info = f'<b>UPDATED: {node}</b><br>New Distance: {distances.get(node, "∞")}'
            elif node in visited:
                dist = distances.get(node, float('infinity'))
                color = get_color_gradient(dist, max_distance)
                node_colors.append(color)
                node_sizes.append(28)
                node_line_widths.append(3)
                node_line_colors.append('rgba(0, 200, 0, 0.8)')
                node_text.append(f'<b>{node}</b>')
                hover_info = f'<b>Visited: {node}</b><br>Distance: {dist}'
            else:
                node_colors.append('rgba(100, 150, 255, 0.6)')
                node_sizes.append(25)
                node_line_widths.append(2)
                node_line_colors.append('rgba(100, 100, 100, 0.5)')
                node_text.append(f'<b>{node}</b>')
                dist = distances.get(node, "∞")
                hover_info = f'Node: {node}<br>Distance: {dist}<br>Status: Unvisited'
            
            hover_text.append(hover_info)
        
        traces.append(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="middle center",
            textfont=dict(
                size=14, 
                color='white', 
                family='Arial Black',
            ),
            hoverinfo='text',
            hovertext=hover_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=node_line_widths, color=node_line_colors),
                opacity=1
            ),
            showlegend=False
        ))
        
        fig = go.Figure(data=traces)
        
        title_text = "Graph Algorithm Visualization"
        if step:
            title_text = f"Step {step.get('step_num', 0)} - {step.get('description', '')}"
        
        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(size=20, color='white')
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=60),
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                showline=False
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                showline=False
            ),
            plot_bgcolor='rgba(20, 20, 40, 1)',
            paper_bgcolor='rgba(20, 20, 40, 1)',
            height=700,
            font=dict(color='white')
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return go.Figure()

def main():
    st.title("Graph Theory AI Visualizer")
    st.markdown("### Experience algorithm animations in stunning visual detail!")
    
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'algorithm_steps' not in st.session_state:
        st.session_state.algorithm_steps = []
    if 'algorithm_running' not in st.session_state:
        st.session_state.algorithm_running = False
    
    with st.sidebar:
        st.header("Graph Configuration")
        
        graph_type = st.selectbox(
            "Graph Type", 
            ["Random", "Small World", "Scale-Free", "Complete", "Grid"]
        )
        
        # Use session_state keys so voice commands can update these values programmatically
        num_nodes = st.slider("Number of Nodes", 5, 20, st.session_state.get('num_nodes', 10), key="num_nodes")
        connectivity = st.slider("Connectivity", 0.1, 1.0, st.session_state.get('connectivity', 0.3), 0.1, key="connectivity")
        
        layout_type = st.selectbox(
            "Layout", 
            ["Spring", "Circular", "Kamada-Kawai", "Random"]
        )
        
        st.divider()
        
        st.header(" Animation Style")
        animation_style = st.selectbox(
            "Visual Style",
            ["modern", "classic", "minimal"]
        )
        st.divider()
        
        # Add TTS toggle
        st.header("🔊 Narration")
        if 'tts_enabled' not in st.session_state:
            st.session_state.tts_enabled = False
        st.session_state.tts_enabled = st.toggle("Enable Voice Narration", value=st.session_state.tts_enabled)
        
        st.divider()
        st.header("Voice Command")
        rec_seconds = st.slider("Record duration (seconds)", 1, 8, 4)
        if st.button("🎤 Record Voice Command", width="stretch"):
            # Local recording path
            tmp_path = None
            try:
                with st.spinner("Recording..."):
                    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    tmp_path = tmp.name
                    tmp.close()
                    fs = 44100
                    sd.default.samplerate = fs
                    sd.default.channels = 1
                    recording = sd.rec(int(rec_seconds * fs), samplerate=fs, channels=1, dtype='float32')
                    sd.wait()
                    sf.write(tmp_path, recording, fs)
                    time.sleep(0.2)
                st.success(f"Saved recording to {tmp_path}")
            except Exception as e:
                st.error(f"Recording failed: {e}")
                tmp_path = None

            if tmp_path:
                # Transcribe with OpenAI Whisper via existing client
                if not client:
                    st.error("OpenAI client not configured - set OPENAI_API_KEY in your environment")
                else:
                    with st.spinner("Transcribing audio with Whisper..."):
                        try:
                            with open(tmp_path, 'rb') as af:
                                # Use server-side OpenAI client to transcribe (force English)
                                resp = client.audio.transcriptions.create(file=af, model="whisper-1", language="en")
                                # SDK can return different structures; try common keys
                                transcript = None
                                if hasattr(resp, 'text'):
                                    transcript = resp.text
                                elif isinstance(resp, dict) and 'text' in resp:
                                    transcript = resp['text']
                                else:
                                    # fallback: try str()
                                    transcript = str(resp)
                        except Exception as e:
                            st.error(f"Transcription error: {e}")
                            transcript = None

                    if transcript:
                        st.info(f"Transcript: {transcript}")
                        params = parse_voice_command(transcript)
                        st.write("Detected parameters:", params)
                        
                        # Store params to apply on next run (before widgets are created)
                        if params:
                            st.session_state.voice_params_to_apply = params
                            st.rerun()

                        # Show status from previous apply attempt if any
                        if st.session_state.apply_voice_success:
                            st.success("Applied: " + ", ".join(st.session_state.apply_voice_success))
                            st.session_state.apply_voice_success = None
                        
                        if st.session_state.apply_voice_errors:
                            st.error("Errors while applying params: " + 
                                   ", ".join(st.session_state.apply_voice_errors))
                            st.session_state.apply_voice_errors = None
    
    G = create_graph(graph_type, num_nodes, connectivity)
    pos = get_node_positions(G, layout_type)
    
    st.sidebar.header(" Algorithm")
    algo_options = ["Dijkstra's Shortest Path", "Breadth-First Search (BFS)", "Depth-First Search (DFS)", "Prim's Minimum Spanning Tree"]
    default_alg = st.session_state.get('algorithm', algo_options[0])
    try:
        default_index = algo_options.index(default_alg) if default_alg in algo_options else 0
    except Exception:
        default_index = 0
    algorithm = st.sidebar.selectbox(
        "Choose Algorithm",
        algo_options,
        index=default_index,
        key="algorithm"
    )
    
    if algorithm == "Prim's Minimum Spanning Tree":
        start_node = st.selectbox("🟢 Start Node", sorted(G.nodes()), key="start")
        end_node = None
    else:
        col1, col2 = st.columns(2)
        with col1:
            start_node = st.selectbox("🟢 Start Node", sorted(G.nodes()), key="start")
        with col2:
            end_node = st.selectbox("🔴 End Node", sorted(G.nodes()), key="end")
    
    st.sidebar.header(" Controls")
    
    if st.sidebar.button("▶Start Algorithm", width="stretch"):
        if algorithm == "Prim's Minimum Spanning Tree":
            st.session_state.algorithm_steps = prims_algorithm(G, start_node)
            st.session_state.current_step = 0
            st.session_state.algorithm_running = True
        elif start_node != end_node:
            if algorithm == "Dijkstra's Shortest Path":
                st.session_state.algorithm_steps = dijkstra_algorithm(G, start_node, end_node)
            elif algorithm == "Breadth-First Search (BFS)":
                st.session_state.algorithm_steps = bfs_algorithm(G, start_node, end_node)
            elif algorithm == "Depth-First Search (DFS)":
                st.session_state.algorithm_steps = dfs_algorithm(G, start_node, end_node)
            st.session_state.current_step = 0
            st.session_state.algorithm_running = True
        else:
            st.sidebar.error("Start and end nodes must be different!")
    
    if st.session_state.algorithm_running and st.session_state.algorithm_steps:
        st.sidebar.divider()
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("⏮Previous", width="stretch") and st.session_state.current_step > 0:
                st.session_state.current_step -= 1
        
        with col2:
            if st.button("⏭Next", width="stretch") and st.session_state.current_step < len(st.session_state.algorithm_steps) - 1:
                st.session_state.current_step += 1
        
        if st.sidebar.button("Reset", width="stretch"):
            st.session_state.current_step = 0
            st.session_state.algorithm_running = False
            st.session_state.algorithm_steps = []
            st.rerun()
        
        # Only show progress if algorithm_steps is not empty
        if st.session_state.algorithm_steps:
            progress = (st.session_state.current_step + 1) / len(st.session_state.algorithm_steps)
            st.sidebar.progress(progress)
            st.sidebar.caption(f"Step {st.session_state.current_step + 1} of {len(st.session_state.algorithm_steps)}")
    
    col1, col2 = st.columns([2.5, 1])
    
    with col1:
        current_step_data = None
        if st.session_state.algorithm_running and st.session_state.algorithm_steps:
            if st.session_state.current_step < len(st.session_state.algorithm_steps):
                current_step_data = st.session_state.algorithm_steps[st.session_state.current_step]
        
        fig = visualize_graph_animated(G, pos, current_step_data, animation_style)
        st.plotly_chart(fig, width="stretch", key="main_graph")
    
    with col2:
        st.subheader("Algorithm Status")
        
        if st.session_state.algorithm_running and st.session_state.algorithm_steps:
            if st.session_state.current_step < len(st.session_state.algorithm_steps):
                step = st.session_state.algorithm_steps[st.session_state.current_step]
                
                st.markdown(f"### {step.get('description', '')}")
                
                if step.get('narration'):
                    narration_text = step['narration']
                    st.info(narration_text)
                    
                    # Generate and play TTS if enabled
                    if st.session_state.tts_enabled and client and st.session_state.current_step != st.session_state.last_narrated_step:
                        try:
                            # Generate speech
                            audio_response = client.audio.speech.create(
                                model="tts-1",
                                voice="alloy",
                                input=narration_text
                            )
                            
                            # Save to temp file
                            temp_path = "temp_narration.mp3"
                            audio_response.stream_to_file(temp_path)
                            
                            # Play audio using sounddevice and soundfile
                            data, samplerate = sf.read(temp_path)
                            sd.play(data, samplerate)
                            
                            # Update last narrated step
                            st.session_state.last_narrated_step = st.session_state.current_step
                        except Exception as e:
                            st.error(f"TTS Error: {str(e)}")
                
                st.divider()
                
                current_node = step.get('current_node')
                if current_node is not None:
                    st.markdown(f"#### Current Node: **{current_node}**")
                    if algorithm == "Dijkstra's Shortest Path":
                        current_distance = step.get('distances', {}).get(current_node, '∞')
                        if current_distance != float('infinity'):
                            st.metric("Distance from Start", current_distance)
                        else:
                            st.metric("Distance from Start", "∞")
                
                if step.get('exploring_edge'):
                    edge = step['exploring_edge']
                    weight = G[edge[0]][edge[1]].get('weight', 1) if G.has_edge(edge[0], edge[1]) else 'N/A'
                    st.markdown(f"####  Examining Edge")
                    st.success(f"**{edge[0]} → {edge[1]}** (weight: {weight})")
                
                if algorithm == "Breadth-First Search (BFS)" and step.get('queue') is not None:
                    st.markdown(f"####  Queue: {step['queue']}")
                elif algorithm == "Depth-First Search (DFS)" and step.get('stack') is not None:
                    st.markdown(f"####  Stack: {step['stack']}")
                
                st.divider()
                
                if step.get('updated_neighbors'):
                    st.markdown("####  Updated This Step:")
                    for neighbor in step['updated_neighbors']:
                        if algorithm == "Dijkstra's Shortest Path":
                            new_dist = step['distances'][neighbor]
                            st.success(f"Node {neighbor} → Distance: {new_dist}")
                        else:
                            st.success(f"Node {neighbor}")
                
                if algorithm == "Dijkstra's Shortest Path" and step.get('distances'):
                    st.markdown("####  All Distances:")
                    distances_data = []
                    for node in sorted(G.nodes()):
                        dist = step['distances'][node]
                        status = ""
                        if node == step.get('current_node'):
                            status = "🎯"
                        elif node in step.get('visited', set()):
                            status = "✅"
                        else:
                            status = "⏳"
                        
                        dist_str = str(dist) if dist != float('infinity') else '∞'
                        
                        distances_data.append({
                            "": status,
                            "Node": str(node), 
                            "Dist": dist_str
                        })
                    
                    st.dataframe(distances_data, hide_index=True, width="stretch")
                
                if algorithm in ["Breadth-First Search (BFS)", "Depth-First Search (DFS)"]:
                    st.markdown("#### Visited Nodes:")
                    visited_list = sorted(step.get('visited', set()))
                    st.write(visited_list if visited_list else "None yet")
                
                if step.get('finished'):
                    st.balloons()
                    if algorithm == "Prim's Minimum Spanning Tree":
                        total_weight = sum(G[e[0]][e[1]]['weight'] for e in step.get('mst_edges', []))
                        st.success(f"###  MST Complete!")
                        st.metric("Total MST Weight", total_weight)
                        st.metric("Edges in MST", len(step.get('mst_edges', [])))
                    else:
                        st.success(f"### Path Found!")
                        if algorithm == "Dijkstra's Shortest Path" and end_node:
                            st.metric("Total Distance", step['distances'][end_node])
                        path_str = ' → '.join(map(str, step['path']))
                        st.info(f"**Path:** {path_str}")
                
                if client:
                    st.divider()
                    st.markdown("####  AI Explanation")
                    with st.spinner("Generating explanation..."):
                        graph_info = f"{len(G.nodes())} nodes, {len(G.edges())} edges"
                        narration = get_ai_narration(algorithm, step['description'], graph_info)
                        st.info(narration)
        else:
            st.info(" Select nodes and click **Start Algorithm** to begin")
            
            st.divider()
            st.markdown("####  Graph Stats")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Nodes", len(G.nodes()))
                st.metric("Density", f"{nx.density(G):.2f}")
            with col_b:
                st.metric("Edges", len(G.edges()))
                if nx.is_connected(G):
                    try:
                        st.metric("Diameter", nx.diameter(G))
                    except:
                        pass

if __name__ == "__main__":
    main()