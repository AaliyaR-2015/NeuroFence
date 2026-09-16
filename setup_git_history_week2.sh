#!/usr/bin/env bash
# setup_git_history_week2.sh
#
# Layers Week 2 commits onto an EXISTING NeuroFence repo that already has
# the Week 1 branch structure (i.e. you already unzipped Week 1 and ran
# setup_git_history.sh). Copies files from week2_source/ into place AFTER
# checking out each branch -- not before -- so git checkout never has to
# fight with locally modified files.
#
# Usage:
#   1. Unzip the Week 2 update package so that week2_source/ and this
#      script sit at the root of your existing NeuroFence repo.
#   2. ./setup_git_history_week2.sh

set -euo pipefail

if [ ! -d .git ]; then
  echo "No .git found here. Run this from inside your NeuroFence repo (the one setup_git_history.sh already built)." >&2
  exit 1
fi

if [ ! -d week2_source ]; then
  echo "week2_source/ not found. Unzip the Week 2 update package into this repo root first." >&2
  exit 1
fi

for b in main member1-model-sandbox member2-adversarial-fuzzer member3-activation-tracker member4-forensic-desktop-app; do
  if ! git rev-parse --verify -q "$b" >/dev/null; then
    echo "Branch '$b' not found. Run setup_git_history.sh (Week 1) first." >&2
    exit 1
  fi
done

if [ -n "$(git status --porcelain | grep -v '^??')" ]; then
  echo "You have uncommitted changes to tracked files. Commit or stash them first, then re-run this script." >&2
  exit 1
fi

# --- Member 1: Model Sandbox Week 2 -----------------------------------------
git checkout -q member1-model-sandbox
cp week2_source/src/model_sandbox.py src/model_sandbox.py
git add src/model_sandbox.py
git commit -q -m "feat(week2): add weight integrity hashing, safetensors-only local dir check, batch tokenization

- compute_weights_sha256(): deterministic hash over all loaded parameter
  tensors -- works regardless of source file layout, and is exactly what
  Week 4's automated security report needs to prove model identity
- assert_local_path_is_safetensors_only(): explicit rejection of local
  directories containing .bin/.pt/.ckpt files, even alongside safetensors
  -- hardens against Week 3's locally-created 'backdoored test model'
- tokenize_batch(): padded/truncated batch tokenization so the fuzzer can
  feed thousands of prompts through the model efficiently instead of one
  Python-level forward call per prompt
- ModelMetadata now carries weights_sha256"
cp week2_source/tests/test_model_sandbox.py tests/test_model_sandbox.py
git add tests/test_model_sandbox.py
git commit -q -m "test: add unit tests for Week 2 model_sandbox additions

- hash is deterministic for identical weights, changes on any single
  weight perturbation
- local-dir safety check rejects .bin/.pt, accepts safetensors-only,
  no-ops for HF Hub ids
- tokenize_batch sets pad_token when missing and forwards args correctly
- 6/6 tests passing (run with HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 if
  importing transformers hangs in your environment)"

# --- Member 2: Adversarial Fuzzer Week 2 ------------------------------------
git checkout -q member2-adversarial-fuzzer
cp week2_source/src/fuzz_runner.py src/fuzz_runner.py
git add src/fuzz_runner.py
git commit -q -m "feat(week2): add fuzz runner integrating fuzzer + sandbox + tracker + baseline

- build_baseline_profile(): feeds n_benign prompts through the model in
  batches, accumulates a NeuronBaselineProfile from their per-neuron
  activations (baseline = benign-only, by design -- Week 3 compares
  everything else against this)
- run_full_fuzz_sweep(): runs benign + edge_case + trigger_candidate
  prompts, returns one tagged log entry per prompt with per-layer scalar
  stats -- this is what the Week 2 UI heatmap visualizes
- save_sweep_log(): persists sweep results to JSON"
mkdir -p tests
touch tests/__init__.py
cp week2_source/tests/test_fuzz_runner.py tests/test_fuzz_runner.py
git add tests/__init__.py tests/test_fuzz_runner.py
git commit -q -m "test: add fuzz_runner integration tests (mocked model, no internet needed)

- baseline profile covers every hooked layer with correct sample count
- full sweep tags every prompt with its category, keeps scalar-only stats
- cross-check: build_baseline_profile() output matches manual
  tracker+profile accumulation exactly -- proves the integration is
  correct, not just 'runs without crashing'
- 3/3 tests passing"

# --- Member 3: Activation Tracker Week 2 ------------------------------------
git checkout -q member3-activation-tracker
cp week2_source/src/activation_tracker.py src/activation_tracker.py
git add src/activation_tracker.py
git commit -q -m "feat(week2): add per-neuron activation capture to ActivationTracker

- track_neurons=True computes a per-neuron mean vector per layer (mean
  over batch+sequence dims, keeping the hidden dimension) instead of only
  a single scalar mean/std/max per layer
- Stored on LayerActivationStats.neuron_means (None when disabled -- no
  extra memory cost for callers that don't need it)
- This is what makes 'active vs dormant neurons' visualization (Week 2 UI)
  and per-neuron anomaly scoring (Week 3) possible, rather than only
  layer-level granularity"
cp week2_source/tests/test_activation_tracker.py tests/test_activation_tracker.py
git add tests/test_activation_tracker.py
git commit -q -m "test: cover per-neuron capture (enabled/disabled) -- 4/4 passing"
cp week2_source/src/baseline_profile.py src/baseline_profile.py
git add src/baseline_profile.py
git commit -q -m "feat(week2): add NeuronBaselineProfile using Welford's online algorithm

- Running per-neuron, per-layer mean/variance updated one sample at a
  time -- O(hidden_size) memory per layer regardless of how many
  thousands of prompts are fed in, unlike collect-then-numpy.mean/std
- finalize() -> {layer: {mean, std, n}}; save()/load() for JSON
  persistence, needed by Week 3 (anomaly z-scores) and Week 4 (reporting)
- This is the 'what does normal look like' baseline that later detection
  logic compares new activations against"
cp week2_source/tests/test_baseline_profile.py tests/test_baseline_profile.py
git add tests/test_baseline_profile.py
git commit -q -m "test: verify NeuronBaselineProfile against direct numpy computation

- test_matches_numpy_within_tolerance: Welford's result matches
  samples.mean()/samples.std() to 1e-8 -- proves the memory-efficient
  algorithm didn't sacrifice correctness
- Independent per-layer tracking, single-sample edge case (std=0),
  shape-mismatch guard, save/load roundtrip
- 5/5 tests passing"

# --- Member 4: Forensic Desktop App Week 2 ----------------------------------
git checkout -q member4-forensic-desktop-app
mkdir -p tests
touch tests/__init__.py
cp week2_source/ui/heatmap_widget.py ui/heatmap_widget.py
cp week2_source/tests/test_heatmap_widget.py tests/test_heatmap_widget.py
git add ui/heatmap_widget.py tests/__init__.py tests/test_heatmap_widget.py
git commit -q -m "feat(week2): add neuron activation heatmap widget

- aggregate_sweep_log(): reduces a fuzz_runner sweep-log JSON to a
  layers x categories matrix of mean activation -- depends only on the
  JSON's shape, not on importing Member 2's fuzz_runner code directly
- HeatmapWidget: custom QPainter-painted grid, blue->yellow->red heat
  colormap (blue = dormant, red = highly active), no charting-library
  dependency
- 4/4 tests passing: category averaging correct, layer order matches
  first appearance, widget renders offscreen with and without data"
cp week2_source/ui/main_window.py ui/main_window.py
git add ui/main_window.py
git commit -q -m "feat(week2): wire heatmap widget into MainWindow

- New 'Neuron Activation Heatmap' group with a 'Load Fuzz Sweep Log...'
  button (QFileDialog defaulting to outputs/) and the embedded
  HeatmapWidget
- load_sweep_log(): parses the chosen file, populates the heatmap, logs
  the result; parse errors are caught and surfaced via QMessageBox rather
  than crashing the app"

# --- Merge every member branch into main, in order --------------------------
git checkout -q main
git merge --no-ff -q member1-model-sandbox -m "Merge branch 'member1-model-sandbox': Model Sandbox Week 2 (integrity hash, batch tokenization)"
git merge --no-ff -q member2-adversarial-fuzzer -m "Merge branch 'member2-adversarial-fuzzer': Adversarial Fuzzer Week 2 (fuzz runner integration)"
git merge --no-ff -q member3-activation-tracker -m "Merge branch 'member3-activation-tracker': Activation Tracker Week 2 (per-neuron capture, baseline profile)"
git merge --no-ff -q member4-forensic-desktop-app -m "Merge branch 'member4-forensic-desktop-app': Forensic Desktop App Week 2 (neuron heatmap)"

# --- Integration layer on main -----------------------------------------------
cp week2_source/README.md README.md
git add README.md
git commit -q -m "docs: update README with Week 2 status and run instructions"
cp week2_source/src/run_week2_pipeline.py src/run_week2_pipeline.py
git add src/run_week2_pipeline.py
git commit -q -m "feat: integrate all four Week 2 modules into a runnable pipeline

- run_week2_pipeline.py: loads model -> attaches per-neuron hooks ->
  builds baseline profile (benign-only) -> runs full mixed sweep -> saves
  outputs/baseline_profile.json + outputs/fuzz_sweep_log.json"

rm -rf week2_source

echo ""
echo "Week 2 history added. Full graph:"
git log --oneline --graph --all
echo ""
echo "Push it:"
echo "  git push origin --all"
