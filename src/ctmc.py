import numpy as np
from scipy.linalg import expm
from typing import List, Dict, Any
from src.object_graph import ObjectGraph

def build_q_matrix(object_graph: ObjectGraph,
                   current_node_id: str,
                   is_interacting: bool,
                   walk_speed: float,
                   isp_predictions: List[Dict[str, Any]] = None,
                   T_i_override: float = None) -> np.ndarray:
    """
    Builds the CTMC transition matrix Q (N x N).
    """
    N = object_graph.N
    Q = np.zeros((N, N))
    
    current_idx = object_graph.get_idx(current_node_id)
    
    # CASE 1: Walking transitions (between any adjacent nodes)
    for i in range(N):
        node_i_id = object_graph.get_node(i)["id"]
        neighbors = object_graph.neighbors(node_i_id)
        
        # If we are NOT interacting at node i, we can walk from it
        # Actually, the plan says: "x_i is NOT the current interaction node"
        # This implies we can walk from any node EXCEPT the one we are currently interacting at.
        if not (is_interacting and i == current_idx):
            for neighbor_id in neighbors:
                j = object_graph.get_idx(neighbor_id)
                dist = object_graph.distance(node_i_id, neighbor_id)
                if dist > 0:
                    Q[i, j] = walk_speed / dist

    # CASE 2: Interaction transitions (from the current interaction node only)
    if is_interacting and isp_predictions:
        # Use T_i_override if provided (e.g., ground truth persona duration), 
        # otherwise fallback to mean of predicted durations.
        if T_i_override is not None:
            T_i = T_i_override
        else:
            durations = [p.get('duration_seconds', 30.0) for p in isp_predictions]
            T_i = float(np.mean(durations)) if durations else 30.0
        
        for p in isp_predictions:
            target_id = p.get('object_id')
            if target_id in object_graph.node_ids:
                j = object_graph.get_idx(target_id)
                likelihood = p.get('likelihood', 0.0)
                Q[current_idx, j] = likelihood / T_i

    # CASE 3: Diagonal
    for i in range(N):
        Q[i, i] = -np.sum(Q[i, :])
        
    return Q

def solve_ctmc(Q: np.ndarray, t: float, p0: np.ndarray) -> np.ndarray:
    """
    Solves P(x(t)) = expm(Q * t) * p0
    p0: initial distribution (one-hot vector)
    """
    # Note: Q is singular by design (rows sum to 0), so cond(Q) is always high.
    # We rely on expm numerical stability.
    
    M = expm(Q * t)
    pt = M.T @ p0
    
    # 3. Numerical guards
    pt = np.clip(pt, 0, None)
    if pt.sum() > 0:
        pt /= pt.sum()
    else:
        pt = p0 # Fallback
        
    return pt
