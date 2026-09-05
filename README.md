# NeuroFence — LLM Weight Poisoning & Backdoor Scanner

**Domain:** AI Security (SecOps) / Model Forensics
**Internship:** Infotact Solutions — Advanced Cybersecurity Engineering

## Problem Statement

Enterprises frequently download open-source Large Language Models (LLMs) from the
internet to run locally. Attackers are increasingly using **model poisoning** —
injecting sleeper-agent backdoors directly into neural network weights. A poisoned
model behaves normally until a specific trigger input (e.g. a rare word or phrase)
is fed to it, at which point it produces malicious or leaked output.

## The Idea

NeuroFence is an **offline model forensic tool**. Before an enterprise deploys a
downloaded model, NeuroFence:

1. Loads the model into an isolated sandbox (weights only, no arbitrary code execution).
2. Fires adversarial/synthetic prompts into it.
3. Monitors internal activation layers via PyTorch hooks.
4. Flags "dormant neurons" — clusters that only spike on highly specific, unnatural
   inputs — as evidence of a mathematically backdoored model.

## Architecture & Team Ownership

| Module | Branch | Tech | Purpose |
|---|---|---|---|
| Model Sandbox | `member1-model-sandbox` | PyTorch + HuggingFace | Safely load `.safetensors` models, no pickle/arbitrary code |
| Adversarial Fuzzer | `member2-adversarial-fuzzer` | Python | Generates benign/edge-case/trigger-candidate prompts |
| Activation Tracker | `member3-activation-tracker` | PyTorch Hooks | Captures per-layer activation statistics |
| Forensic Desktop App | `member4-forensic-desktop-app` | PyQt | Local, offline UI to inspect models & neuron activity |

Each module is developed on its own branch and merged into `main` at the end
of each week's milestone. `main.py` and `src/test_harness.py` are integration
code that depends on all four modules, so they land directly on `main` once
every branch is merged.

## Week 1 Deliverables (this milestone)

- [x] Project scaffold, environment, dependency list
- [x] **Member 1** — Secure model sandbox loader (`src/model_sandbox.py`)
- [x] **Member 2** — Adversarial fuzzer prompt generator (`src/adversarial_fuzzer.py`)
- [x] **Member 3** — PyTorch hook-based activation tracker (`src/activation_tracker.py`)
- [x] **Member 4** — PyQt desktop UI skeleton with model upload + metadata view (`ui/main_window.py`)
- [x] Integration — test harness (`src/test_harness.py`) + UI↔backend wiring (`main.py`)

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# CLI test harness (no GUI) — validates sandbox + hooks against a model
python -m src.test_harness --model sshleifer/tiny-gpt2

# Desktop app
python main.py
```

## Security notes

- Models are loaded with `use_safetensors=True` and `trust_remote_code=False` to
  avoid executing arbitrary code embedded in a model repo.
- No network calls are made once a model is cached locally — analysis is fully
  offline, suitable for air-gapped environments.

## Roadmap (later weeks)

- Week 2: Adversarial fuzzer + neuron activation heatmap in the UI
- Week 3: Backdoor injection (test model) + anomaly detection logic
- Week 4: Automated PDF security reporting, performance polish
