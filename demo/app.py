import os
import sys
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request, jsonify

# Ensure src is in path
sys.path.append(os.path.abspath("."))
from src.scene_graph import SceneGraph
from src.object_graph import ObjectGraph
from src.isp import ISPModule
from src.ptp import PTPModule
from src.personas import PERSONAS

app = Flask(__name__)

# Cache for loaded scenes
_cache = {}

def get_scene(scene_id):
    if scene_id not in _cache:
        sg = SceneGraph(f"data/{scene_id}/semseg.v2.json")
        og = ObjectGraph(sg)
        isp = ISPModule()
        ptp = PTPModule(isp)
        _cache[scene_id] = (sg, og, ptp)
    return _cache[scene_id]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    scene_id = data.get('scene_id')
    past_labels = data.get('past_labels', [])
    t = float(data.get('t', 30.0))
    persona_name = data.get('persona', 'relaxer')
    
    sg, og, ptp = get_scene(scene_id)
    walk_speed = PERSONAS[persona_name]["walk_speed_m_per_s"]
    
    # Map past labels to IDs (just take the first match for simplicity)
    node_ids = []
    for label in past_labels:
        label = label.strip().lower()
        for node in og.nodes:
            if node['label'] == label:
                node_ids.append(node['id'])
                break
    
    # We use the first node (or the last past node) as the starting point
    current_node_id = node_ids[-1] if node_ids else og.node_ids[0]
    
    # Get distribution
    pt = ptp.predict_distribution(
        og, sg, current_node_id, is_interacting=True, 
        walk_speed=walk_speed, horizon_seconds=t, past_node_ids=node_ids
    )
    
    # Render Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter objects
    node_positions = np.array([n["position"] for n in og.nodes])
    x, y = node_positions[:, 0], node_positions[:, 1]
    
    # Heatmap-like scatter
    scatter = ax.scatter(x, y, c=pt, s=200, cmap='YlOrRd', alpha=0.8, edgecolors='k')
    plt.colorbar(scatter, label='Probability')
    
    # Annotate top 5 nodes
    top_indices = np.argsort(pt)[-5:]
    for idx in top_indices:
        node = og.get_node(idx)
        ax.annotate(f"{node['object_label'] if 'object_label' in node else node['label']} ({pt[idx]:.1%})", (x[idx], y[idx]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, weight='bold')
    
    ax.set_title(f"Predicted Distribution at t={t}s")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return jsonify({
        "image": img_base64,
        "entropy": float(-np.sum(pt * np.log(pt + 1e-12)))
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
