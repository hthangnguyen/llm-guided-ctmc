import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath("."))
from src.scene_graph import SceneGraph
from src.object_graph import ObjectGraph
from src.personas import PERSONAS

def test_object_graph_visualization():
    """
    Load scene_A, build ObjectGraph, render 2D plot.
    Save to experiments/results/object_graph_{scene_id}.png
    """
    SCENE_A_ID = "4acaebcc-6c10-2a2a-858b-29c7e4fb410d"
    
    path = f"data/{SCENE_A_ID}/semseg.v2.json"
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return

    sg = SceneGraph(path)
    og = ObjectGraph(sg)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw edges
    for (id_a, id_b) in og.edges:
        idx_a, idx_b = og.get_idx(id_a), og.get_idx(id_b)
        node_a, node_b = og.get_node(idx_a), og.get_node(idx_b)
        xa, ya = node_a["position"][0], node_a["position"][1]
        xb, yb = node_b["position"][0], node_b["position"][1]
        ax.plot([xa, xb], [ya, yb], "gray", linewidth=0.5, alpha=0.4, zorder=1)

    # Color nodes by persona affinity
    worker_labels = set(PERSONAS["worker"]["preferred_labels"])
    relaxer_labels = set(PERSONAS["relaxer"]["preferred_labels"])
    
    for node in og.nodes:
        x, y = node["position"][0], node["position"][1]
        if node["label"] in worker_labels and node["label"] in relaxer_labels:
            color = "purple" # Both
        elif node["label"] in worker_labels:
            color = "steelblue"
        elif node["label"] in relaxer_labels:
            color = "#E85D24"
        else:
            color = "gray"
        
        ax.scatter(x, y, c=color, s=120, zorder=3)
        ax.annotate(node["label"][:8], (x, y), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")

    legend = [
        mpatches.Patch(color="steelblue", label="Worker-preferred"),
        mpatches.Patch(color="#E85D24", label="Relaxer-preferred"),
        mpatches.Patch(color="purple", label="Both"),
        mpatches.Patch(color="gray", label="Neutral"),
    ]
    ax.legend(handles=legend, fontsize=8)
    ax.set_title(f"ObjectGraph: {sg.scene_id} | {og.N} nodes | connected={og.is_connected()}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"object_graph_{sg.scene_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Visualization saved to: {out_path}")
    print(og.summary())
    
    assert og.is_connected()
    assert og.N > 0

if __name__ == "__main__":
    test_object_graph_visualization()
