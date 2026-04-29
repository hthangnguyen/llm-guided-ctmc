import numpy as np
from typing import List, Dict, Any
from src.object_graph import ObjectGraph
from src.trajectory import Trajectory
from src.ptp import PTPModule

class Evaluator:
    def __init__(self, ptp: PTPModule, object_graph: ObjectGraph, scene_graph: Any):
        self.ptp = ptp
        self.og = object_graph
        self.sg = scene_graph

    def compute_nll(self, trajectory: Trajectory, walk_speed: float, t_eval: List[float] = None) -> Dict[str, float]:
        """
        Computes NLL for a single trajectory at multiple timesteps.
        Returns {"llm_nll": float, "uniform_nll": float}
        """
        if t_eval is None:
            t_eval = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
            
        current_node_id = trajectory.node_ids[0]
        is_interacting = True 
        
        # Predicted distributions for all timesteps in one go
        # Note: We need to update PTPModule to handle multiple t if we want full efficiency,
        # but for now we'll just ensure we don't rebuild Q multiple times if we can.
        # Actually, let's keep it simple but fix the logic.
        
        # 1. Get ISP Predictions (LLM) - once per trajectory
        isp_preds = self.ptp.isp.predict_next_interactions(
            self.sg, current_node_id, past_node_ids=[]
        )
        
        # 2. Build Q - once per trajectory
        # We use the ground-truth persona duration mean as T_i (the stay time at current node)
        # to ensure the CTMC accurately reflects the persona's physical behavior.
        from src.personas import PERSONAS
        persona_duration = PERSONAS.get(trajectory.persona, {}).get("interaction_duration_mean_seconds", 30.0)
        
        from src.ctmc import build_q_matrix, solve_ctmc
        Q = build_q_matrix(
            self.og, current_node_id, is_interacting, walk_speed, isp_preds,
            T_i_override=persona_duration
        )
        
        # 3. Initial distribution
        p0 = np.zeros(self.og.N)
        p0[self.og.get_idx(current_node_id)] = 1.0
        
        llm_nlls = []
        uniform_nlls = []
        uniform_p = 1.0 / self.og.N
        
        for t in t_eval:
            if t > trajectory.total_duration:
                continue
                
            # 4. Solve for each t using the same Q
            pt = solve_ctmc(Q, t, p0)
            
            true_node_id = trajectory.node_at(t)
            true_idx = self.og.get_idx(true_node_id)
            
            p_true = np.clip(pt[true_idx], 1e-10, 1.0)
            llm_nlls.append(-np.log(p_true))
            uniform_nlls.append(-np.log(uniform_p))
            
        return {
            "llm_nll": float(np.mean(llm_nlls)) if llm_nlls else 0.0,
            "uniform_nll": float(np.mean(uniform_nlls)) if uniform_nlls else 0.0
        }

    def evaluate_dataset(self, trajectories: List[Trajectory], walk_speed: float) -> Dict[str, Any]:
        """
        Evaluates a list of trajectories and returns averaged NLLs.
        """
        llm_total = []
        unif_total = []
        
        for i, traj in enumerate(trajectories):
            print(f"Evaluating trajectory {i+1}/{len(trajectories)}...")
            res = self.compute_nll(traj, walk_speed)
            llm_total.append(res["llm_nll"])
            unif_total.append(res["uniform_nll"])
            
        return {
            "avg_llm_nll": np.mean(llm_total),
            "avg_uniform_nll": np.mean(unif_total),
            "win_rate": np.mean(np.array(llm_total) < np.array(unif_total))
        }
