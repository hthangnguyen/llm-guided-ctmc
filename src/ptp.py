import numpy as np
from typing import List, Dict, Any
from src.object_graph import ObjectGraph
from src.isp import ISPModule
from src.ctmc import build_q_matrix, solve_ctmc

class PTPModule:
    def __init__(self, isp: ISPModule):
        self.isp = isp

    def predict_distribution(self, 
                             object_graph: ObjectGraph,
                             scene_graph: Any,
                             current_node_id: str,
                             is_interacting: bool,
                             walk_speed: float,
                             horizon_seconds: float,
                             past_node_ids: List[str] = None) -> np.ndarray:
        """
        Full prediction pipeline:
        1. Get ISP predictions (LLM)
        2. Build Q matrix
        3. Solve CTMC for horizon_seconds
        """
        # 1. ISP Predictions
        isp_preds = self.isp.predict_next_interactions(
            scene_graph, current_node_id, past_node_ids
        )
        
        # 2. Build Q
        Q = build_q_matrix(
            object_graph, current_node_id, is_interacting, walk_speed, isp_preds
        )
        
        # 3. Initial distribution (one-hot at current node)
        p0 = np.zeros(object_graph.N)
        p0[object_graph.get_idx(current_node_id)] = 1.0
        
        # 4. Solve
        pt = solve_ctmc(Q, horizon_seconds, p0)
        
        return pt
