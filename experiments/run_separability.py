import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from scipy.spatial.distance import jensenshannon
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath("."))
from src.scene_graph import SceneGraph
from src.object_graph import ObjectGraph
from src.personas import PERSONAS
from src.trajectory import generate_dataset, Trajectory

def check_separability(worker_trajs: list[Trajectory],
                       relaxer_trajs: list[Trajectory],
                       object_graph: ObjectGraph) -> dict:
    # 1. Count visits per node
    worker_counts = Counter([nid for t in worker_trajs for nid in set(t.node_ids)])
    relaxer_counts = Counter([nid for t in relaxer_trajs for nid in set(t.node_ids)])
    
    # 2. Normalize to probabilities
    node_ids = object_graph.node_ids
    w_vec = np.array([worker_counts.get(nid, 0) for nid in node_ids], dtype=float)
    r_vec = np.array([relaxer_counts.get(nid, 0) for nid in node_ids], dtype=float)
    
    # Normalize
    if w_vec.sum() > 0: w_vec /= w_vec.sum()
    if r_vec.sum() > 0: r_vec /= r_vec.sum()
    
    # 3. JS Divergence
    # JS divergence in [0, 1] for base 2
    js_dist = jensenshannon(w_vec, r_vec, base=2)
    js_div = js_dist**2 # JS divergence is the square of JS distance
    
    # 4. Overlap fraction
    both = set(worker_counts.keys()) & set(relaxer_counts.keys())
    overlap = len(both) / object_graph.N
    
    # 5. Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for ax, counts, title, color in [
        (ax1, worker_counts, "Worker Visits", "steelblue"),
        (ax2, relaxer_counts, "Relaxer Visits", "#E85D24")
    ]:
        for node in object_graph.nodes:
            x, y = node["position"][0], node["position"][1]
            count = counts.get(node["id"], 0)
            size = 50 + (count / max(counts.values()) * 500) if counts else 50
            ax.scatter(x, y, s=size, c=color, alpha=0.6)
            ax.annotate(node["label"][:6], (x, y), fontsize=6)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    plt.suptitle(f"Separability: JS Div = {js_div:.3f}, Overlap = {overlap:.1%}")
    
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"separability_{object_graph._scene_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    verdict = js_div > 0.3 and overlap < 0.5
    
    print(f"\n--- Separability Report: {object_graph._scene_id} ---")
    print(f"JS divergence: {js_div:.4f} (target: > 0.3)")
    print(f"Node overlap: {overlap:.1%} (target: < 50%)")
    print(f"Verdict: {'SEPARABLE ✓' if verdict else 'NOT SEPARABLE ✗'}")
    print(f"Plot saved to: {out_path}")
    
    return {"js_divergence": js_div, "overlap": overlap, "separable": verdict}

def main():
    SCENE_IDS = ["4acaebcc-6c10-2a2a-858b-29c7e4fb410d", "754e884c-ea24-2175-8b34-cead19d4198d"]
    
    for scene_id in SCENE_IDS:
        print(f"\nProcessing Scene: {scene_id}")
        sg = SceneGraph(f"data/{scene_id}/semseg.v2.json")
        og = ObjectGraph(sg)
        
        print("Generating trajectories for separability check...")
        dataset = generate_dataset(og, PERSONAS, n_per_persona=20)
        
        check_separability(dataset["worker"], dataset["relaxer"], og)

if __name__ == "__main__":
    main()
