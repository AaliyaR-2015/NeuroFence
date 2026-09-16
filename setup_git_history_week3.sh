#!/usr/bin/env bash
#
# setup_git_history_week3.sh
#
# Layers Week 3 commits onto an EXISTING NeuroFence repo (same pattern as
# setup_git_history_week2.sh) -- backdoor injector (test fixture),
# z-score anomaly detector, integration pipeline, and the new UI widget.
# All commits land under YOUR git identity, same as Weeks 1-2.
#
# Usage:
#   1. Put week3_source/ at the root of your existing NeuroFence repo
#      (the one setup_git_history.sh + setup_git_history_week2.sh already built).
#   2. ./setup_git_history_week3.sh

set -euo pipefail

if [ ! -d .git ]; then
  echo "No .git found here. Run this from inside your NeuroFence repo." >&2
  exit 1
fi

if [ ! -d week3_source ]; then
  echo "week3_source/ not found. Put the Week 3 update package at this repo's root first." >&2
  exit 1
fi

for b in main member1-model-sandbox member2-adversarial-fuzzer member3-activation-tracker member4-forensic-desktop-app; do
  if ! git rev-parse --verify -q "$b" >/dev/null; then
    echo "Branch '$b' not found. Run setup_git_history.sh and setup_git_history_week2.sh first." >&2
    exit 1
  fi
done

if [ -n "$(git status --porcelain | grep -v '^??')" ]; then
  echo "You have uncommitted changes to tracked files. Commit or stash them first." >&2
  exit 1
fi

# --- Member 1: Model Sandbox — supplies the poisoned test fixture -----------
git checkout -q member1-model-sandbox
mkdir -p src
cp week3_source/src/backdoor_injector.py src/backdoor_injector.py
git add src/backdoor_injector.py
git commit -q -m "feat(week3): add backdoor_injector.py -- synthetic poisoned test fixture

- inject_backdoor(): hand-crafted weight perturbation (no training) that
  makes one target neuron fire only on a chosen trigger token, wired
  forward to shift output toward a canary token -- same 'sleeper agent'
  shape as the README's problem statement
- create_test_backdoored_model(): loads a base model fresh, poisons a
  copy, saves it safetensors-only (model_sandbox-compatible), and writes
  ground_truth.json so the detector's precision/recall can be scored
- Entirely local/offline; used only to give our own scanner a
  known-positive to validate against"

mkdir -p tests
cp week3_source/tests/test_backdoor_injector.py tests/test_backdoor_injector.py
git add tests/test_backdoor_injector.py
git commit -q -m "test: add backdoor_injector tests against a hand-built stand-in model

- same 'stand-in model' pattern as test_activation_tracker.py -- no HF
  download needed, fully offline
- clean-input logits barely move; trigger-input logits shift hard toward
  the canary token; empty trigger encoding raises cleanly
- (written + syntax-checked; run locally where torch is already installed)"

# --- Member 2: Adversarial Fuzzer — no Week 3 changes needed ----------------
# generate_batch() is reused as-is by the Week 3 pipeline; nothing to add here.

# --- Member 3: Activation Tracker / Baseline — the actual detector ----------
git checkout -q member3-activation-tracker
cp week3_source/src/anomaly_detector.py src/anomaly_detector.py
git add src/anomaly_detector.py
git commit -q -m "feat(week3): add anomaly_detector.py -- z-score scan against the baseline

- detect_anomalous_neurons_for_layer(): flags neurons whose category mean
  activation deviates z_threshold+ std from the Week 2 benign-only
  baseline
- 'selective trigger' flag: dormant (near-baseline) on benign, but
  anomalous specifically on trigger_candidate prompts -- distinguishes a
  planted backdoor neuron from a merely noisy one
- detect_anomalous_neurons_all_layers() + save_anomaly_report() for the
  full-model sweep"

cp week3_source/tests/test_anomaly_detector.py tests/test_anomaly_detector.py
git add tests/test_anomaly_detector.py
git commit -q -m "test: verify anomaly_detector with synthetic baselines (pure numpy)

- no findings when everything matches baseline
- correctly flags a hand-planted selective-trigger neuron
- distinguishes 'noisy on everything' from 'selective trigger'
- validates required categories, sorts selective findings first
- 5/5 tests passing"

# --- Member 4: Forensic Desktop App — new report widget ---------------------
git checkout -q member4-forensic-desktop-app
mkdir -p ui
cp week3_source/ui/anomaly_report_widget.py ui/anomaly_report_widget.py
cp week3_source/ui/integration_notes.md ui/integration_notes.md
git add ui/anomaly_report_widget.py ui/integration_notes.md
git commit -q -m "feat(week3): add AnomalyReportWidget to render anomaly_report.json

- Sorted table (selective-trigger findings first), red/orange highlighting
  so a likely backdoor is visible at a glance instead of raw JSON
- integration_notes.md: manual wiring diff for main_window.py (kept
  separate from the existing heatmap wiring rather than guessing at and
  overwriting it)"

# --- Merge every member branch into main, in order --------------------------
git checkout -q main
git merge --no-ff -q member1-model-sandbox -m "Merge branch 'member1-model-sandbox': backdoor test fixture (Week 3)"
git merge --no-ff -q member3-activation-tracker -m "Merge branch 'member3-activation-tracker': z-score anomaly detector (Week 3)"
git merge --no-ff -q member4-forensic-desktop-app -m "Merge branch 'member4-forensic-desktop-app': anomaly report widget (Week 3)"

# --- Integration layer on main -----------------------------------------------
cp week3_source/src/run_week3_pipeline.py src/run_week3_pipeline.py
git add src/run_week3_pipeline.py
git commit -q -m "feat: integrate Week 3 -- backdoor fixture + fuzzer + tracker + detector

- run_week3_pipeline.py: loads a model, runs benign/edge_case/trigger
  prompts with per-neuron tracking, scores every neuron against the
  Week 2 baseline, writes outputs/anomaly_report.json
- NOTE: load_model() has an ADAPT ME marker -- copy your real
  model_sandbox loader call from run_week2_pipeline.py into it"

rm -rf week3_source

echo ""
echo "Week 3 history added. Full graph:"
git log --oneline --graph --all

echo ""
echo "Still to do by hand (see comments left in the code):"
echo "  1. src/run_week3_pipeline.py: point load_model() at your real model_sandbox function"
echo "  2. ui/main_window.py: apply the ~10-line diff in ui/integration_notes.md"
echo ""
echo "Push it:"
echo "  git push origin --all"
