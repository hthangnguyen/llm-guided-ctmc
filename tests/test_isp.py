import sys
import os
import json

# Ensure src is in path
sys.path.append(os.path.abspath("."))
from src.scene_graph import SceneGraph
from src.isp import ISPModule

def test_isp_prediction():
    SCENE_A_ID = "4acaebcc-6c10-2a2a-858b-29c7e4fb410d"
    path = f"data/{SCENE_A_ID}/semseg.v2.json"
    
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return

    sg = SceneGraph(path)
    isp = ISPModule(model="gemma3:latest")

    # Pick an object (e.g., a sofa or a table)
    current_node = sg.nodes[0]
    print(f"\n--- Testing ISP Prediction from: {current_node['label']} (ID: {current_node['id']}) ---")
    
    predictions = isp.predict_next_interactions(sg, current_node['id'])
    
    print("\nPredictions:")
    print(json.dumps(predictions, indent=2))
    
    if predictions:
        print("\nSuccess: Received valid predictions from LLM.")
        # Basic validation
        for p in predictions:
            assert "id" in p
            assert "likelihood" in p
            assert "duration" in p
    else:
        print("\nFailure: No predictions received.")

if __name__ == "__main__":
    test_isp_prediction()
