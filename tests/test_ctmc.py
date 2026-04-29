import numpy as np
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath("."))
from src.scene_graph import SceneGraph
from src.object_graph import ObjectGraph
from src.ctmc import build_q_matrix, solve_ctmc

def test_ctmc_propagation():
    SCENE_A_ID = "4acaebcc-6c10-2a2a-858b-29c7e4fb410d"
    path = f"data/{SCENE_A_ID}/semseg.v2.json"
    
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return

    sg = SceneGraph(path)
    og = ObjectGraph(sg)
    
    # 1. Build Q matrix (walking only, no interactions for simplicity)
    current_node_id = og.node_ids[0]
    walk_speed = 1.0
    Q = build_q_matrix(og, current_node_id, is_interacting=False, walk_speed=walk_speed)
    
    # 2. Initial state (one-hot)
    p0 = np.zeros(og.N)
    p0[0] = 1.0
    
    print(f"\n--- Testing CTMC Propagation from Node {current_node_id} ---")
    
    # 3. Solve for different times
    for t in [1.0, 10.0, 60.0]:
        pt = solve_ctmc(Q, t, p0)
        entropy = -np.sum(pt * np.log(pt + 1e-12))
        top_idx = np.argmax(pt)
        top_prob = pt[top_idx]
        top_label = og.get_node(top_idx)["label"]
        
        print(f"t={t:4.1f}s | Sum: {pt.sum():.4f} | Entropy: {entropy:.4f} | Top: {top_label} ({top_prob:.2%})")
        
        assert np.isclose(pt.sum(), 1.0)
        assert np.all(pt >= 0)

if __name__ == "__main__":
    test_ctmc_propagation()
