from dataclasses import dataclass, field
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any
from src.object_graph import ObjectGraph

@dataclass
class Interaction:
    object_id: str        # node id
    object_label: str     # e.g. "chair"
    start_time: float     # seconds from trajectory start
    duration: float       # seconds
    position: np.ndarray  # shape (2,): [x, y]

@dataclass
class Trajectory:
    scene_id: str
    persona: str                    # "worker" or "relaxer"
    seed: int                       # for reproducibility
    interactions: list              # List[Interaction]
    timestamps: np.ndarray          # shape (T,): time in seconds
    positions: np.ndarray           # shape (T, 2): [x, y]
    node_ids: list                  # List[str] shape (T,): nearest node id
    dt: float = 1.0                 # timestep in seconds

    def position_at(self, t: float) -> np.ndarray:
        idx = np.argmin(np.abs(self.timestamps - t))
        return self.positions[idx]

    def node_at(self, t: float) -> str:
        idx = np.argmin(np.abs(self.timestamps - t))
        return self.node_ids[idx]

    @property
    def total_duration(self) -> float:
        return float(self.timestamps[-1]) if len(self.timestamps) > 0 else 0.0

def generate_persona_trajectory(object_graph: ObjectGraph,
                                persona_name: str,
                                personas: dict,
                                seed: int = 0,
                                dt: float = 1.0) -> Trajectory:
    rng = np.random.default_rng(seed)
    persona = personas[persona_name]
    preferred = set(persona["preferred_labels"])
    avoided = set(persona["avoided_labels"])
    n_interactions = persona["n_interactions"]
    walk_speed = persona["walk_speed_m_per_s"]
    duration_mean = persona["interaction_duration_mean_seconds"]

    # Step 1: Weighted object pool
    weights = []
    for node in object_graph.nodes:
        if node["label"] in preferred: weight = 10.0
        elif node["label"] in avoided: weight = 0.05
        else: weight = 1.0
        weights.append(weight)
    
    weights = np.array(weights)
    if not np.any(weights > 1.0):
        print(f"WARNING: No preferred objects for persona '{persona_name}' in this scene.")
        # Proceed with uniform if no preferred objects found (not ideal)
        
    weights /= weights.sum()

    # Step 2: Sample interaction sequence
    interaction_nodes = rng.choice(object_graph.nodes, size=n_interactions, 
                                   replace=False, p=weights)

    # Step 3 & 4: Build steps
    current_time = 0.0
    all_timestamps = []
    all_positions = []
    all_node_ids = []
    interactions_list = []

    # Start at the first node
    start_node = interaction_nodes[0]
    
    # Interaction at the first node
    duration = np.clip(rng.exponential(scale=duration_mean), 10.0, duration_mean * 3)
    n_steps = max(2, int(duration / dt))
    pos = np.array(start_node["position"][:2])
    for k in range(n_steps):
        all_timestamps.append(current_time + k * dt)
        all_positions.append(pos)
        all_node_ids.append(start_node["id"])
    
    interactions_list.append(Interaction(
        object_id=start_node["id"],
        object_label=start_node["label"],
        start_time=current_time,
        duration=duration,
        position=pos
    ))
    current_time += duration

    # Move to subsequent nodes
    for i in range(1, len(interaction_nodes)):
        prev_node = interaction_nodes[i-1]
        curr_node = interaction_nodes[i]
        
        # Walking
        dist = object_graph.distance(prev_node["id"], curr_node["id"])
        travel_time = dist / walk_speed
        n_steps = max(2, int(travel_time / dt))
        start_pos = np.array(prev_node["position"][:2])
        end_pos = np.array(curr_node["position"][:2])
        
        for k in range(n_steps):
            frac = k / n_steps
            p = (1 - frac) * start_pos + frac * end_pos
            node_at_step = prev_node["id"] if frac < 0.5 else curr_node["id"]
            all_timestamps.append(current_time + k * dt)
            all_positions.append(p)
            all_node_ids.append(node_at_step)
        current_time += travel_time
        
        # Interaction
        duration = np.clip(rng.exponential(scale=duration_mean), 10.0, duration_mean * 3)
        n_steps = max(2, int(duration / dt))
        pos = np.array(curr_node["position"][:2])
        for k in range(n_steps):
            all_timestamps.append(current_time + k * dt)
            all_positions.append(pos)
            all_node_ids.append(curr_node["id"])
            
        interactions_list.append(Interaction(
            object_id=curr_node["id"],
            object_label=curr_node["label"],
            start_time=current_time,
            duration=duration,
            position=pos
        ))
        current_time += duration

    return Trajectory(
        scene_id=object_graph._scene_id,
        persona=persona_name,
        seed=seed,
        interactions=interactions_list,
        timestamps=np.array(all_timestamps),
        positions=np.array(all_positions),
        node_ids=all_node_ids,
        dt=dt
    )

def generate_dataset(object_graph: ObjectGraph,
                    personas: dict,
                    n_per_persona: int = 20,
                    output_dir: str = "data/trajectories") -> dict:
    results = {"worker": [], "relaxer": []}
    base_path = Path(output_dir) / object_graph._scene_id
    
    for persona_name in ["worker", "relaxer"]:
        persona_path = base_path / persona_name
        persona_path.mkdir(parents=True, exist_ok=True)
        
        for seed in range(n_per_persona):
            traj = generate_persona_trajectory(object_graph, persona_name, personas, seed=seed)
            results[persona_name].append(traj)
            
            with open(persona_path / f"traj_{seed:04d}.pkl", "wb") as f:
                pickle.dump(traj, f)
                
    return results

def load_dataset(scene_id: str, data_dir: str = "data/trajectories") -> dict:
    """
    Loads trajectories from disk.
    Returns: {"worker": List[Trajectory], "relaxer": List[Trajectory]}
    """
    results = {"worker": [], "relaxer": []}
    base_path = Path(data_dir) / scene_id
    
    for persona_name in ["worker", "relaxer"]:
        persona_path = base_path / persona_name
        if not persona_path.exists():
            continue
            
        # Sort files to ensure deterministic order
        files = sorted(persona_path.glob("traj_*.pkl"))
        for f in files:
            with open(f, "rb") as pf:
                results[persona_name].append(pickle.load(pf))
                
    return results
