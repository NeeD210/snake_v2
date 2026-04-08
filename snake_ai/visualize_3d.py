import argparse
import time
import json
import numpy as np
import torch
import os

from game import SnakeGame
from model import SnakeNet
from mcts import MCTS
from visualize import _load_model_state_compat
from schedules import get_mcts_simulations

# Visual Constants
COLOR_TRUNK = "#A020F0"  # Purple (Solid Spine)
COLOR_LINK_TRUNK = "#A020F0" # Purple for spine connections
COLOR_LINK_SIM = "rgba(255,255,255,0.4)" # Semi-transparent white for branches

def extract_tree_data(mcts_node, game_step, search_depth, global_nodes, global_links, visited_links):
    """
    Recursively traces the MCTS tree, using id(node) to maintain object identity across turns.
    This naturally handles nodes transitioning from simulation to trunk.
    """
    stack = [(mcts_node, search_depth)]
    while stack:
        node, d = stack.pop()
        nid = str(id(node))
        
        status = "sim"
        if getattr(node, "is_trunk", False):
            status = "trunk"
        elif node.reward == 1.0:
            status = "win"
        elif node.reward == -1.0:
            status = "death"
            
        current_y = (game_step + d) * 50
        
        if nid not in global_nodes:
            board_flat = node.state.flatten().tolist()
            global_nodes[nid] = {
                "id": nid,
                "Q": float(node.Q),
                "N": int(node.N),
                "game_step": game_step,
                "search_depth": d,
                "status": status,
                "board": board_flat,
                "fy": current_y
            }
        else:
            # Update dynamic values
            global_nodes[nid]["status"] = status if status != "sim" else global_nodes[nid]["status"]
            global_nodes[nid]["Q"] = float(node.Q)
            global_nodes[nid]["N"] = int(node.N)
            # Ensure the height is corrected if it's now part of a more recent turn's trunk
            if status == "trunk":
                global_nodes[nid]["fy"] = current_y

        # PIN the spine to the center
        if global_nodes[nid]["status"] == "trunk":
            global_nodes[nid]["fx"] = 0
            global_nodes[nid]["fz"] = 0

        for action, child in node.children.items():
            cid = str(id(child))
            link_id = f"{nid}->{cid}"
            
            if link_id not in visited_links:
                visited_links.add(link_id)
                global_links.append({
                    "source": nid,
                    "target": cid
                })
            
            stack.append((child, d + 1))


def generate_html(graph_data, output_file):
    """Generates the interactive 3D HTML viewer with unified tree, purple spine, and gradients."""
    json_data = json.dumps(graph_data)
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Snake AI - 3D Unified MCTS Tree</title>
  <style> 
    body {{ margin: 0; padding: 0; background-color: #050508; overflow: hidden; }} 
    #info-panel {{ 
      position: absolute; top: 10px; left: 10px; color: white;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      pointer-events: none; background: rgba(0, 0, 0, 0.7);
      padding: 15px; border-radius: 12px; border: 1px solid #444;
      box-shadow: 0 4px 15px rgba(0,0,0,0.5); z-index: 10;
    }}
    .legend-item {{ display: flex; align-items: center; margin-bottom: 5px; font-size: 14px; }}
    .dot {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 10px; }}
    
    .mini-board {{
      display: grid; margin-top: 10px; gap: 1px; padding: 2px;
      background: #222; border: 1px solid #555; width: 120px; height: 120px;
    }}
    .cell {{ width: 100%; height: 100%; }}
    .cell-0 {{ background: #111; }} 
    .cell-1 {{ background: #0c0; }} 
    .cell-2 {{ background: #0f0; border: 1px solid white; }} 
    .cell-3 {{ background: #f00; border-radius: 50%; }} 
  </style>
  <script src="https://unpkg.com/3d-force-graph"></script>
</head>
<body>
  <div id="info-panel">
    <h2 style="margin-top: 0;">MCTS Heat-Map Tree</h2>
    <div class="legend-item"><div class="dot" style="background: {COLOR_TRUNK};"></div><b>Purple:</b> Central Spine (Purple Link)</div>
    <div class="legend-item"><div class="dot" style="background: #22dd22;"></div><b>Green Gradient:</b> High-Value Simulation (Q=1)</div>
    <div class="legend-item"><div class="dot" style="background: #dd2222;"></div><b>Red Gradient:</b> Low-Value / Death (Q=-1)</div>
    <p style="margin-top: 15px; border-top: 1px solid #666; padding-top: 10px; font-size: 12px; color: #aaa;">
        <b>Deduplicated Spine:</b> Simulation nodes transition to Trunk.<br/>
        <b>Trunk Links:</b> Purple connecting purple nodes.
    </p>
  </div>
  
  <div id="3d-graph"></div>
  
  <script>
    const gData = {json_data};
    const bSize = gData.board_size;

    // Gradient helper: Interpolates between Red (-1) and Green (1)
    const getQColor = (q) => {{
        if (q > 0) {{
            // 0 (Light Grey/Teal) to 1 (Bright Green)
            const g = Math.floor(100 + 155 * q);
            const b = Math.floor(200 - 100 * q);
            return `rgb(0, ${{g}}, ${{b}})`;
        }} else {{
            // 0 (Light Grey/Teal) to -1 (Bright Red)
            const r = Math.floor(100 + 155 * Math.abs(q));
            const b = Math.floor(200 - 150 * Math.abs(q));
            return `rgb(${{r}}, 50, ${{b}})`;
        }}
    }};

    const elem = document.getElementById('3d-graph');
    const Graph = ForceGraph3D()(elem)
      .graphData(gData)
      .nodeLabel(node => {{
          let boardHtml = `<div class="mini-board" style="grid-template-columns: repeat(${{bSize}}, 1fr); grid-template-rows: repeat(${{bSize}}, 1fr);">`;
          node.board.forEach(cellType => {{ boardHtml += `<div class="cell cell-${{cellType}}"></div>`; }});
          boardHtml += '</div>';

          return `
            <div style="background: rgba(0,0,0,0.9); padding: 12px; border-radius: 8px; font-family: sans-serif; color: white; border: 1px solid #555;">
              <div style="margin-bottom: 8px;">
                <strong>Status:</strong> ${{node.status.toUpperCase()}}<br/>
                <strong>Turn / Depth:</strong> ${{node.game_step}} / ${{node.search_depth}}<br/>
                <strong>Q:</strong> ${{node.Q.toFixed(3)}} | <strong>N:</strong> ${{node.N}}
              </div>
              ${{boardHtml}}
            </div>
          `;
      }})
      .nodeColor(node => {{
          if (node.status === 'trunk') return '{COLOR_TRUNK}';
          return getQColor(node.Q);
      }})
      .nodeResolution(20)
      .linkDirectionalParticles(0)
      .linkWidth(link => {{
          return (link.source.status === 'trunk' && link.target.status === 'trunk') ? 2 : 1;
      }})
      .linkColor(link => {{
          if (link.source.status === 'trunk' && link.target.status === 'trunk') return '{COLOR_LINK_TRUNK}';
          return '{COLOR_LINK_SIM}';
      }})
      .onNodeClick(node => {{
          const distance = 80;
          const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);
          Graph.cameraPosition({{ x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }}, node, 1000);
      }});

    Graph.d3Force('link').distance(40);
    Graph.d3Force('charge').strength(-400);

    Graph.cameraPosition({{ x: 800, y: 100, z: 800 }});
  </script>
</body>
</html>
"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"\nUnified visualization generated: {os.path.abspath(output_file)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="snake_ai/experiments/train_v4/snake_net.pth", help="Path to model file")
    parser.add_argument("--board_size", type=int, default=10, help="Board size")
    parser.add_argument("--sims", type=int, default=20, help="Number of MCTS simulations per turn")
    parser.add_argument("--gen", type=int, default=0, help="Generation index for the MCTS sim schedule")
    parser.add_argument("--no-sim-schedule", action="store_true", help="Disable per-move sim scheduling")
    parser.add_argument("--dev", action="store_true", help="Use dev-mode sim schedule (lower sims)")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum number of game steps to record")
    parser.add_argument("--output", type=str, default="mcts_3d_viewer.html", help="HTML Output Path")

    args = parser.parse_args()

    model = SnakeNet(board_size=args.board_size)
    try:
        _load_model_state_compat(model, args.model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.eval()
    mcts = MCTS(model, n_simulations=args.sims)
    game = SnakeGame(board_size=args.board_size)
    
    global_nodes = {}
    global_links = []
    visited_links = set()
    
    guaranteed_win_path = None
    step = 0
    done = False
    
    print("\nMapping Unified MCTS Tree...")

    while not done and step < args.max_steps:
        if not args.no_sim_schedule:
            mcts.n_simulations = get_mcts_simulations(args.gen, len(game.snake), game.board_size, dev_mode=args.dev)

        if mcts.root:
            mcts.root.is_trunk = True
        
        if guaranteed_win_path:
            rel_action = guaranteed_win_path.pop(0)
        else:
            p_probs, entropy, win_path, _sims_done, _timing = mcts.search(game)
            if win_path is not None:
                guaranteed_win_path = win_path
                rel_action = guaranteed_win_path.pop(0)
            else:
                rel_action = np.argmax(p_probs)
            
            # Extract tree with UNIFIED IDs (id(node))
            extract_tree_data(mcts.root, step, 0, global_nodes, global_links, visited_links)
            
        abs_action = (game.direction + (rel_action - 1)) % 4
        _, _, done = game.step(abs_action)
        
        # Advance MCTS
        mcts.update_root(rel_action)
        step += 1
        print(f"Step {step:03d} | Score: {game.score:2d} | Nodes: {len(global_nodes)}", end='\r')

    if mcts.root:
        mcts.root.is_trunk = True
        extract_tree_data(mcts.root, step, 0, global_nodes, global_links, visited_links)

    print(f"\nGame finished at step {step}. Result: {game.death_reason}")

    graph_data = {
        "nodes": list(global_nodes.values()),
        "links": global_links,
        "board_size": args.board_size
    }

    generate_html(graph_data, args.output)

if __name__ == "__main__":
    main()
