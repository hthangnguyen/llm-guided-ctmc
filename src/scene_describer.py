import numpy as np
from collections import Counter
from typing import List, Optional
from src.scene_graph import SceneGraph

def describe_scene(scene_graph: SceneGraph, 
                   past_interactions: List[str] = None) -> str:
    """
    Generates a natural language description of the scene for LLM prompt.
    CRITICAL: No persona hints (e.g., 'worker', 'office', 'relaxer').
    """
    nodes = scene_graph.nodes
    
    # Comma-separated counts
    labels = [n["label"] for n in nodes]
    counts = Counter(labels)
    counts_str = ", ".join([f"{c}x {l}" for l, c in counts.most_common()])
    
    # Filter to top 12 objects by distance from center if needed
    if len(nodes) > 12:
        positions = np.array([n["position"] for n in nodes])
        center = np.mean(positions, axis=0)
        # We want objects that are "important" - often those further from center 
        # (on walls/edges) or we could just take all. 
        # The plan says "highest z-score from room center"
        dists = np.linalg.norm(positions - center, axis=1)
        # Sort by distance descending and take top 12
        top_indices = np.argsort(dists)[-12:]
        display_nodes = [nodes[i] for i in top_indices]
    else:
        display_nodes = nodes
        
    # Sort alphabetically by label for deterministic output
    display_nodes = sorted(display_nodes, key=lambda x: x["label"])
    
    loc_lines = []
    for node in display_nodes:
        x, y = node["position"][0], node["position"][1]
        loc_lines.append(f"- {node['label']} at ({x:.1f}, {y:.1f})")
    loc_str = "\n".join(loc_lines)
    
    # Recent activity
    if past_interactions:
        activity_str = f"The person recently visited: {', '.join(past_interactions)}"
    else:
        activity_str = "No prior interactions recorded."
        
    description = f"""---
INDOOR SCENE DESCRIPTION:
This space contains the following objects:
{counts_str}

Object locations (approximate 2D positions in meters):
{loc_str}

RECENT ACTIVITY:
{activity_str}
---"""

    # Validation: length < 1500 chars
    if len(description) > 1500:
        # Fallback: even fewer objects
        display_nodes = sorted(display_nodes[:8], key=lambda x: x["label"])
        loc_lines = [f"- {n['label']} at ({n['position'][0]:.1f}, {n['position'][1]:.1f})" for n in display_nodes]
        loc_str = "\n".join(loc_lines)
        # Re-build
        description = f"""---
INDOOR SCENE DESCRIPTION:
This space contains the following objects:
{counts_str}

Object locations (approximate 2D positions in meters):
{loc_str}

RECENT ACTIVITY:
{activity_str}
---"""

    return description
