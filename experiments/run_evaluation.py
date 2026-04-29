import sys
import os
import numpy as np
import pickle
from pathlib import Path

# Ensure src is in path
sys.path.append(os.path.abspath("."))
from src.scene_graph import SceneGraph
from src.object_graph import ObjectGraph
from src.personas import PERSONAS
from src.trajectory import generate_dataset, Trajectory
from src.isp import ISPModule
from src.ptp import PTPModule
from src.evaluator import Evaluator

def evaluate_scene(scene_id: str, label: str):
    print(f"\n" + "="*50)
    print(f"EVALUATING {label}: {scene_id}")
    print("="*50)
    
    # 1. Setup
    sg = SceneGraph(f"data/{scene_id}/semseg.v2.json")
    og = ObjectGraph(sg)
    
    isp = ISPModule(model="gemma3:latest")
    ptp = PTPModule(isp)
    evaluator = Evaluator(ptp, og, sg)
    
    # 2. Generate/Load Dataset
    print(f"Generating/Loading trajectories for {scene_id}...")
    from src.trajectory import load_dataset
    dataset = load_dataset(scene_id)
    
    if not dataset["worker"] or not dataset["relaxer"]:
        print("No saved trajectories found. Generating new dataset...")
        dataset = generate_dataset(og, PERSONAS, n_per_persona=20)
    
    # Filter to test set (seeds 10-19) for consistency
    test_trajs = {
        "worker": [t for t in dataset["worker"] if 10 <= t.seed < 20],
        "relaxer": [t for t in dataset["relaxer"] if 10 <= t.seed < 20]
    }
    
    # 3. Evaluate
    results = {}
    for persona in ["worker", "relaxer"]:
        print(f"\n--- Persona: {persona.upper()} ---")
        walk_speed = PERSONAS[persona]["walk_speed_m_per_s"]
        res = evaluator.evaluate_dataset(test_trajs[persona], walk_speed)
        results[persona] = res
        
        print(f"Results for {persona}:")
        print(f"  Avg LLM NLL:     {res['avg_llm_nll']:.4f}")
        print(f"  Avg Uniform NLL: {res['avg_uniform_nll']:.4f}")
        print(f"  Win Rate:        {res['win_rate']:.1%}")

    return results

def main():
    SCENE_A_ID = "4acaebcc-6c10-2a2a-858b-29c7e4fb410d"
    SCENE_B_ID = "754e884c-ea24-2175-8b34-cead19d4198d"
    
    res_a = evaluate_scene(SCENE_A_ID, "Scene A (Dev)")
    res_b = evaluate_scene(SCENE_B_ID, "Scene B (Eval)")
    
    # Final Summary Table
    print("\n" + "="*60)
    print(f"{'Scene':<15} | {'Persona':<10} | {'LLM NLL':<10} | {'Unif NLL':<10} | {'Win Rate':<10}")
    print("-" * 60)
    
    for scene_label, results in [("Scene A", res_a), ("Scene B", res_b)]:
        for persona, res in results.items():
            print(f"{scene_label:<15} | {persona:<10} | {res['avg_llm_nll']:<10.4f} | {res['avg_uniform_nll']:<10.4f} | {res['win_rate']:<10.1%}")

if __name__ == "__main__":
    main()
