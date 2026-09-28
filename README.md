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

## Week 1 Deliverables

- [x] Project scaffold, environment, dependency list
- [x] **Member 1** — Secure model sandbox loader (`src/model_sandbox.py`)
- [x] **Member 2** — Adversarial fuzzer prompt generator (`src/adversarial_fuzzer.py`)
- [x] **Member 3** — PyTorch hook-based activation tracker (`src/activation_tracker.py`)
- [x] **Member 4** — PyQt desktop UI skeleton with model upload + metadata view (`ui/main_window.py`)
- [x] Integration — test harness (`src/test_harness.py`) + UI↔backend wiring (`main.py`)

## Week 2 Deliverables

- [x] **Member 1** — Weight integrity hashing (`compute_weights_sha256`), safetensors-only
      local-directory validation, batch tokenization for efficient large-scale fuzzing
- [x] **Member 2** — `src/fuzz_runner.py`: feeds prompts through the model, builds a
      benign-only baseline profile, runs a full mixed-category sweep
- [x] **Member 3** — Per-neuron activation capture (`track_neurons=True`) and
      `src/baseline_profile.py`: `NeuronBaselineProfile` using Welford's online
      algorithm, verified against NumPy to 1e-8 precision
- [x] **Member 4** — `ui/heatmap_widget.py`: neuron activation heatmap (custom
      QPainter grid, blue→yellow→red), wired into the desktop app with a
      "Load Fuzz Sweep Log..." button
- [x] Integration — `src/run_week2_pipeline.py` ties all four modules together
      end-to-end, producing `outputs/baseline_profile.json` and
      `outputs/fuzz_sweep_log.json`

## Reproduce the Week 3/4 result

```bash
# 1. Baseline from a clean model
python -m src.run_week2_pipeline --model distilgpt2

# 2. Plant a test backdoor (trigger phrase is one the fuzzer tests for)
python -c "from src.backdoor_injector import create_test_backdoored_model; create_test_backdoored_model('distilgpt2', 'outputs/poisoned_test_model', trigger_phrase='DEPLOY_OVERRIDE')"

# 3. Scan the poisoned model against the baseline
python -m src.run_week3_pipeline --baseline outputs/baseline_profile.json --model outputs/poisoned_test_model

# 4. Write the Markdown report, then open the desktop app
python -m src.generate_security_report --report outputs/anomaly_report.json
python main.py   # Load Fuzz Sweep Log... and Load Anomaly Report...
```

Sample outputs from a real run are in `outputs/`. The scanner finds triggers drawn
from the fuzzer's candidate list; discovering a completely unknown trigger phrase
is future work.

**Troubleshooting:** if importing `transformers` hangs, set
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` — some environments try an update
check on import that stalls without internet access.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# CLI test harness (no GUI) — validates sandbox + hooks against a model
python -m src.test_harness --model distilgpt2

# Desktop app
python main.py
```

## Security notes

- Models are loaded with `use_safetensors=True` and `trust_remote_code=False` to
  avoid executing arbitrary code embedded in a model repo.
- No network calls are made once a model is cached locally — analysis is fully
  offline, suitable for air-gapped environments.

## Week 3 & 4 deliverables

- [x] **Week 3** — `src/backdoor_injector.py` plants a hidden trigger-activated neuron in a
  test model; `src/run_week3_pipeline.py` runs a fuzz sweep (benign / edge-case /
  trigger-candidate prompts) and flags anomalous neurons via z-score comparison
  against the Week 2 baseline. Validated end-to-end against `distilgpt2` (a real
  trained model, not a random-weight test fixture) — 289/290 flagged neurons
  correctly separated as selective triggers (dormant on benign input,
  z ≈ -0.01; spiking sharply on the trigger phrase, z ≈ -14.93).

- [x] **Week 4** — `src/generate_security_report.py` writes a Markdown security report report
  from the anomaly findings (Markdown, not PDF as originally scoped — kept to
  the standard library only, deliberately, to avoid new dependencies under
  deadline pressure). The desktop app's "Backdoor Anomaly Report" panel
  renders findings directly, sorted with selective-trigger neurons first.
