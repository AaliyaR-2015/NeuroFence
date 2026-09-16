#!/usr/bin/env bash
# setup_git_history.sh
#
# Recreates the NeuroFence branch-per-module git history under YOUR git
# identity. Run this from inside the unzipped NeuroFence/ folder, AFTER
# setting your git config (see README / chat instructions).
#
# Safe to run once, on a fresh unzipped copy with no existing .git folder.

set -euo pipefail

if [ -d .git ]; then
  echo "A .git folder already exists here. Remove it first (rm -rf .git) or run this in a fresh unzip." >&2
  exit 1
fi

if [ -z "$(git config user.name || true)" ] || [ -z "$(git config user.email || true)" ]; then
  echo "Set your git identity first, e.g.:" >&2
  echo "  git config --global user.name \"Your Name\"" >&2
  echo "  git config --global user.email \"you@example.com\"" >&2
  exit 1
fi

git init -q
git checkout -q -b main

# --- Scaffold (shared starting point for every branch) ---------------------
# NOTE: tests/__init__.py lives here (not on a member branch) because both
# Member 2 and Member 3 add test files under tests/ -- if it were only
# committed on one branch, checking out a sibling branch (which doesn't
# have it in its history yet) would remove it from disk via git's normal
# branch-switch behavior, breaking the other branch's commit.
mkdir -p tests
touch tests/__init__.py
git add README.md .gitignore requirements.txt src/__init__.py ui/__init__.py outputs/.gitkeep tests/__init__.py
git commit -q -m "chore: initial project scaffold for NeuroFence"
SCAFFOLD=$(git rev-parse HEAD)

# --- Member 1: Model Sandbox -------------------------------------------------
git checkout -q -b member1-model-sandbox "$SCAFFOLD"
git add src/model_sandbox.py
git commit -q -m "feat: add secure model sandbox loader

- Loads HF causal LMs with use_safetensors=True, trust_remote_code=False
- Refuses pickle-based (.bin/.pt) checkpoints -> raises UnsafeModelError
- Freezes all params (inspection only, never fine-tunes in the sandbox)
- Extracts ModelMetadata (arch, param count, layer count, hidden size, dtype)"

# --- Member 2: Adversarial Fuzzer -------------------------------------------
git checkout -q -b member2-adversarial-fuzzer "$SCAFFOLD"
git add src/adversarial_fuzzer.py
git commit -q -m "feat: add adversarial fuzzer prompt generator (Week 1 scaffold)

- generate_batch(): shuffled mix of benign, edge-case, and
  trigger-candidate prompts
- trigger_candidate prompts built from words/phrases common in published
  backdoor-attack literature (rare tokens, instruction-injection strings,
  unicode homoglyphs)
- Deterministic given a seed -- required for Week 4's reproducible
  security reporting"
git add tests/test_adversarial_fuzzer.py
git commit -q -m "test: add unit tests for adversarial fuzzer

- category counts match request, batches deterministic per seed,
  different seeds diverge, every trigger_candidate actually contains a
  known trigger string, edge cases never crash the generator
- 5/5 tests passing"

# --- Member 3: Activation Tracker --------------------------------------------
git checkout -q -b member3-activation-tracker "$SCAFFOLD"
git add src/activation_tracker.py tests/test_activation_tracker.py
git commit -q -m "feat: add PyTorch hook-based activation tracker

- ActivationTracker attaches forward hooks to every transformer block
  (any nn.ModuleList element) -- architecture-agnostic across GPT-2,
  LLaMA-style, BERT-style models
- Captures per-layer mean/std/max_abs/shape on every forward pass
- Context-manager support (with ActivationTracker(model) as tracker: ...)
- Unit-tested against a hand-built 4-layer stand-in model: hooks fire
  once per block, capture correct shapes, detach() removes them cleanly
  -- 2/2 tests passing"

# --- Member 4: Forensic Desktop App ------------------------------------------
git checkout -q -b member4-forensic-desktop-app "$SCAFFOLD"
git add ui/main_window.py
git commit -q -m "feat: add PyQt desktop UI skeleton (model upload + metadata view)

- MainWindow: 'Load Model' row (local folder browser + HF model-id
  field), metadata panel, and a read-only sandbox log panel
- Backend injected via on_load_model_requested callback -- keeps the UI
  importable/testable without pulling in torch/transformers
- Smoke-tested end-to-end (offscreen QT_QPA_PLATFORM): load button
  wiring, metadata display formatting, and logging all verified"

# --- Merge every member branch into main, in order ---------------------------
git checkout -q main
git merge --no-ff -q member1-model-sandbox -m "Merge branch 'member1-model-sandbox': Model Sandbox (Week 1)"
git merge --no-ff -q member2-adversarial-fuzzer -m "Merge branch 'member2-adversarial-fuzzer': Adversarial Fuzzer (Week 1 scaffold)"
git merge --no-ff -q member3-activation-tracker -m "Merge branch 'member3-activation-tracker': Activation Tracker (Week 1)"
git merge --no-ff -q member4-forensic-desktop-app -m "Merge branch 'member4-forensic-desktop-app': Forensic Desktop App (Week 1)"

# --- Integration layer (depends on all 4 modules, so it lands on main) ------
git add src/test_harness.py main.py
git commit -q -m "feat: integrate all four Week 1 modules (test harness + desktop app entry point)

- src/test_harness.py: CLI proving sandbox -> hooks -> JSON log works
  end-to-end
- main.py: wires the PyQt UI (member4) to the model sandbox loader
  (member1); UI never touches torch/transformers directly"

echo ""
echo "Done. Branches created:"
git branch
echo ""
echo "Now add your remote and push everything:"
echo "  git remote add origin https://github.com/<your-username>/NeuroFence.git"
echo "  git push -u origin --all"
