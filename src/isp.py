import json
import hashlib
import pickle
import requests
from pathlib import Path
from typing import List, Dict, Any, Tuple
from src.scene_graph import SceneGraph
from src.scene_describer import describe_scene

class ISPModule:
    def __init__(self, model: str = "gemma3:latest", cache_dir: str = "data/cache/isp", url: str = "http://localhost:11434"):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.url = f"{url}/api/chat"

    def _get_cache_key(self, scene_id: str, current_node_id: str, past_ids: List[str]) -> str:
        past_str = ",".join(past_ids)
        content = f"{scene_id}_{current_node_id}_{past_str}"
        return hashlib.sha256(content.encode()).hexdigest()

    def predict_next_interactions(self, 
                                 scene_graph: SceneGraph, 
                                 current_node_id: str, 
                                 past_node_ids: List[str] = None) -> List[Dict[str, Any]]:
        """
        Predicts next object interactions using LLM via HTTP.
        Returns a list of dicts: {"id": str, "label": str, "likelihood": float, "duration": float}
        """
        if past_node_ids is None:
            past_node_ids = []
            
        cache_key = self._get_cache_key(scene_graph.scene_id, current_node_id, past_node_ids)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        
        # Build prompt
        past_labels = [scene_graph.get_node(nid)["label"] for nid in past_node_ids]
        scene_desc = describe_scene(scene_graph, past_labels)
        current_node = scene_graph.get_node(current_node_id)
        
        system_prompt = (
            "You are a human behavior predictor in indoor scenes. "
            "Based on a scene description and recent activity, predict the next 3 objects "
            "the person is likely to visit and how long they will stay at each (in seconds). "
            "Output MUST be valid JSON: a list of objects with keys: 'object_id', 'object_label', 'likelihood', 'duration_seconds'. "
            "Likelihoods must sum to 1.0. Durations should be between 10 and 120 seconds. "
            "Only use object IDs provided in the scene description."
        )
        
        user_prompt = f"""
{scene_desc}

The person is currently at: {current_node['label']} (ID: {current_node['id']}).
Predict the next 3 likely interactions.
Return ONLY the JSON list.
"""
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "format": "json",
            "stream": False
        }
        
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result['message']['content']
            predictions = json.loads(content)
            
            # Basic validation/normalization
            if not isinstance(predictions, list):
                predictions = [predictions]
                
            # Normalize likelihoods
            total_l = sum(p.get('likelihood', 0) for p in predictions)
            if total_l > 0:
                for p in predictions:
                    p['likelihood'] /= total_l
            else:
                for p in predictions:
                    p['likelihood'] = 1.0 / len(predictions)
            
            # Cache the result
            with open(cache_file, "wb") as f:
                pickle.dump(predictions, f)
                
            return predictions
            
        except Exception as e:
            print(f"Error in LLM prediction: {e}")
            # Fallback
            neighbors = scene_graph.get_neighbors(current_node_id)
            fallback = []
            for nid in neighbors[:3]:
                node = scene_graph.get_node(nid)
                fallback.append({
                    "object_id": nid,
                    "object_label": node["label"],
                    "likelihood": 1.0 / min(len(neighbors), 3),
                    "duration_seconds": 30.0
                })
            return fallback
