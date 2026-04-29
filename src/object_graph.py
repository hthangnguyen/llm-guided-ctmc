import numpy as np
from typing import List, Dict, Tuple, Optional
from src.scene_graph import SceneGraph

class ObjectGraph:
    def __init__(self, scene_graph: SceneGraph, walk_threshold: float = 3.0):
        """
        Simplified graph where nodes = object centroids from SceneGraph.
        
        Args:
            scene_graph: loaded SceneGraph
            walk_threshold: max centroid-to-centroid distance for a walkable edge (meters)
        """
        self._scene_id = scene_graph.scene_id
        self._nodes = scene_graph.nodes
        self._node_list = self._nodes
        self._id_to_idx = {node['id']: i for i, node in enumerate(self._nodes)}
        self.N = len(self._nodes)
        
        self._walk_threshold = walk_threshold
        self._edges = []
        self._build_edges(walk_threshold)
        
        # Connectivity Guarantee
        current_threshold = walk_threshold
        while not self.is_connected() and current_threshold <= 8.0:
            current_threshold += 0.5
            print(f"[ObjectGraph] Not connected. Increasing threshold to {current_threshold:.1f}m")
            self._build_edges(current_threshold)
        
        self._final_threshold = current_threshold
        if not self.is_connected():
            print(f"WARNING: graph not connected for {self._scene_id} even at 8.0m")

    def _build_edges(self, threshold: float):
        self._edges = []
        for i in range(self.N):
            pos_i = np.array(self._nodes[i]['position'])
            for j in range(i + 1, self.N):
                pos_j = np.array(self._nodes[j]['position'])
                dist = np.linalg.norm(pos_i - pos_j)
                if dist <= threshold:
                    self._edges.append((self._nodes[i]['id'], self._nodes[j]['id']))

    @property
    def nodes(self) -> List[dict]:
        return self._nodes

    @property
    def edges(self) -> List[Tuple[str, str]]:
        return self._edges

    @property
    def node_ids(self) -> List[str]:
        return [node['id'] for node in self._nodes]

    def get_idx(self, node_id: str) -> int:
        return self._id_to_idx[str(node_id)]

    def get_node(self, idx: int) -> dict:
        return self._nodes[idx]

    def distance(self, id_a: str, id_b: str) -> float:
        node_a = self._nodes[self.get_idx(id_a)]
        node_b = self._nodes[self.get_idx(id_b)]
        return float(np.linalg.norm(np.array(node_a['position']) - np.array(node_b['position'])))

    def neighbors(self, node_id: str) -> List[str]:
        nbs = []
        for a, b in self._edges:
            if a == node_id: nbs.append(b)
            elif b == node_id: nbs.append(a)
        return nbs

    def is_connected(self) -> bool:
        if self.N == 0: return True
        visited = {self._nodes[0]['id']}
        stack = [self._nodes[0]['id']]
        while stack:
            curr = stack.pop()
            for nb in self.neighbors(curr):
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        return len(visited) == self.N

    def summary(self) -> str:
        return (f"ObjectGraph: {self._scene_id}\n"
                f"  Nodes: {self.N}\n"
                f"  Edges: {len(self._edges)}\n"
                f"  Threshold: {self._final_threshold:.1f}m\n"
                f"  Connected: {self.is_connected()}")
