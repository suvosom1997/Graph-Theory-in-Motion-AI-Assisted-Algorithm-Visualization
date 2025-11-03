# Graph Theory in Motion: AI-Assisted Algorithm Visualization

An interactive web application that visualizes graph algorithms with step-by-step execution and AI-powered explanations. Watch algorithms like Dijkstra's, BFS, DFS, and Prim's MST come to life through detailed animations and intelligent narration.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

### Algorithms
- Dijkstra's Shortest Path Algorithm
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Prim's Minimum Spanning Tree

### Visualizations
- Interactive graph displays with Plotly
- Real-time edge exploration highlighting
- Color-coded node states (current, visited, unvisited)
- Animated path discovery
- Multiple graph types (Random, Small World, Scale-Free, Complete, Grid)
- Customizable layouts (Spring, Circular, Kamada-Kawai, Random)

### AI-Powered Learning
- Step-by-step algorithm explanations using OpenAI GPT
- Natural language descriptions of each operation
- Educational insights into algorithm behavior

### Interactive Controls
- Step forward/backward through algorithm execution
- Visual progress tracking
- Detailed algorithm status panel
- Real-time distance/queue/stack monitoring

## Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (optional, for AI explanations)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/graph-theory-visualizer.git
   cd graph-theory-visualizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   
   > **Note:** The AI narration feature is optional. The app will work without an API key, but won't provide AI-generated explanations.

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   
   The app will automatically open at `http://localhost:8501`

## Usage

### Basic Workflow

1. **Configure Your Graph**
   - Select graph type (Random, Small World, Scale-Free, Complete, Grid)
   - Adjust number of nodes (5-20)
   - Set connectivity level (0.1-1.0)
   - Choose layout style

2. **Choose an Algorithm**
   - Select from Dijkstra's, BFS, DFS, or Prim's MST
   - Pick start node (and end node if applicable)

3. **Visualize and Learn**
   - Click "Start Algorithm"
   - Use Previous/Next buttons to step through
   - Watch the graph animate in real-time
   - Read AI explanations for each step

4. **Analyze Results**
   - View final paths or spanning trees
   - Check distance metrics
   - Examine visited nodes and execution order

### Understanding the Visualization

- **Red Node** - Current node being processed
- **Green Border** - Already visited nodes
- **Blue Node** - Unvisited nodes
- **Yellow Edges** - Final path/MST edges
- **Cyan Edge** - Edge currently being examined
- **Edge Labels** - Display edge weights

## Real-Life Applications

### Dijkstra's Shortest Path
- **GPS Navigation**: Finding the shortest route between two locations
- **Network Routing**: Determining optimal paths for data packets in computer networks
- **Logistics**: Optimizing delivery routes for transportation companies
- **Robotics**: Path planning for autonomous vehicles and robots

### Breadth-First Search (BFS)
- **Social Networks**: Finding shortest connection between two people (degrees of separation)
- **Web Crawling**: Systematically exploring and indexing web pages
- **Peer-to-Peer Networks**: Finding nearby nodes for file sharing
- **Chess Engines**: Finding shortest sequence of moves to checkmate

### Depth-First Search (DFS)
- **Maze Solving**: Finding paths through complex labyrinths
- **Topological Sorting**: Determining task execution order with dependencies
- **Cycle Detection**: Identifying circular dependencies in systems
- **Compiler Design**: Analyzing syntax trees and code structure

### Prim's Minimum Spanning Tree
- **Network Design**: Minimizing cable length in telecommunications networks
- **Circuit Design**: Connecting components with minimum wire length
- **Cluster Analysis**: Grouping similar data points in machine learning
- **Power Grid**: Optimizing electrical distribution networks

## Technologies Used

- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[NetworkX](https://networkx.org/)** - Graph data structures and algorithms
- **[Plotly](https://plotly.com/)** - Interactive visualizations
- **[OpenAI API](https://openai.com/)** - AI-powered explanations
- **[Python-dotenv](https://github.com/theskumar/python-dotenv)** - Environment variable management

## Project Structure

```
graph-theory-visualizer/
│
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not in repo)
├── LICENSE               # MIT License
└── README.md             # Documentation
```

## Educational Value

This tool is designed for:
- Computer Science students learning graph algorithms
- Educators teaching data structures and algorithms
- Self-learners exploring algorithmic thinking
- Technical interview preparation
- Anyone curious about how algorithms work

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built using Streamlit framework
- Graph algorithms implementation based on NetworkX
- Visualizations powered by Plotly
- AI explanations via OpenAI API

---

Made with Python | Star this repo if you find it helpful
