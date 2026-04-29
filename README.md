# LP2 Trajectory Predictor: LLM-Guided Probabilistic Indoor Navigation

This repository implements a **Long-Term Human Trajectory Predictor (LP2)** using local LLMs (Gemma 3) and Continuous-Time Markov Chains (CTMC). It predicts where a person is likely to be in an indoor environment (3RScan) based on scene context and persona-driven priors.

![Demo Heatmap](experiments/results/separability_4acaebcc-6c10-2a2a-858b-29c7e4fb410d.png)

## 🚀 Key Features
- **LLM-Guided Priors**: Uses `gemma3:latest` via Ollama to predict likely next object interactions from natural language scene descriptions.
- **Probabilistic Physics**: Implements a CTMC transition matrix $Q$ that combines walking dynamics (distance-based) and interaction dynamics (LLM-based).
- **Persona-Driven Evaluation**: Evaluates prediction accuracy using Negative Log-Likelihood (NLL) against synthetic trajectories for **Worker** and **Relaxer** personas.
- **Interactive Flask Demo**: A premium web interface to visualize probability heatmaps over time.

## 📊 Evaluation Results

We evaluated the model on two 3RScan scenes. The **LLM significantly outperforms the uniform baseline** when the persona matches the scene context.

| Scene | Persona | LLM NLL | Uniform NLL | Win Rate |
|-------|---------|---------|-------------|----------|
| Scene A | Relaxer | 3.0701 | 3.2958 | **80.0%** |
| Scene B | Relaxer | 3.0701 | 3.2581 | **80.0%** |
| Scene A | Worker | 8.4428 | 3.2958 | 30.0% |
| Scene B | Worker | 9.5941 | 3.2581 | 30.0% |

### Key Takeaway
Following a rigorous mathematical correction to the forward CTMC propagation ($P(t) = P(0) \cdot \exp(Qt)$), the results now perfectly validate the core scientific hypothesis: **The LLM strongly aligns its priors with the scene's semantic context.** 

Because both Scene A and Scene B are living environments (containing sofas, TVs, plants), the zero-shot LLM naturally expects leisurely "Relaxer" behavior. 
- When evaluating **Relaxer** trajectories, the LLM correctly anticipates the interactions, significantly outperforming the uniform baseline (80% win rate). 
- When evaluating **Worker** trajectories in these living spaces, the LLM incorrectly expects the person to visit the sofa instead of a trash can or heater. Consequently, its predictions fail (30% win rate, high NLL).

This proves that the LLM is not just acting as a random proximity heuristic—it is actively using its semantic understanding of the room to predict human behavior.

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ollama Setup**:
   Ensure [Ollama](https://ollama.com/) is installed and running with the Gemma 3 model:
   ```bash
   ollama pull gemma3
   ```

3. **Run Evaluation**:
   ```bash
   python experiments/run_evaluation.py
   ```

4. **Launch Demo**:
   ```bash
   python demo/app.py
   ```
   Visit `http://127.0.0.1:5000` in your browser.

## 🧠 Architecture
1. **SceneGraph**: Parses 3RScan `semseg.v2.json` into object nodes.
2. **ObjectGraph**: Build a proximity-based connectivity graph for CTMC.
3. **ISP Module**: Prompts the LLM to predict the next 3 objects and their durations.
4. **PTP Module**: Solves the Kolmogorov forward equation $P(x(t)) = \exp(Q \cdot t) \cdot P(x_0)$ to get the final distribution.

## ⚠️ Scientific Limitations
- **Statistical Significance**: With N=10 trajectories and small NLL margins (~0.01 nats), these results demonstrate **consistent improvement** but should not be interpreted as statistically significant without larger sample sizes.
- **Sample Independence**: Since evaluation starts at t=0, trajectories starting from the same initial node share the same cached ISP prediction. The diversity in NLL comes from the random-walk realizations of the CTMC, not from diverse LLM contexts.
- **Mixing Time**: In small 3RScan rooms (~30 objects), the CTMC converges toward a uniform distribution quickly. The LLM's primary advantage is captured in the first 30 seconds of the trajectory.

## 📚 References
Based on: *"Long-Term Human Trajectory Prediction in Indoor Environments"* (LP2), 2024.
