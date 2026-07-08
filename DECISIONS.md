# Decisions

Running log of project decisions and the evidence behind them. Newest first.

## 2026-07-07 — v2 adapter: epoch 3 (checkpoint-117) promoted to production

**Decision:** After the BOM-echo data fix and retrain, checkpoint-117
(epoch 3) is the production adapter, copied into the deployed adapter
directory (`models/adapter`). Reverses the v1 epoch-4 choice — on clean
data, epoch 4 overfits (eval loss 1.998 vs 1.958) and epoch 3 also wins
behaviorally (0/25 assistant-style openings, median 36 words vs actual 46).

## 2026-07-07 — Training data rebuilt after BOM-echo discovery (v2 dataset)

**Decision:** Rebuilt pairs, re-split (`--force`, breaking eval continuity
deliberately), and retrained after discovering 29% contamination.

**Evidence:** 112/383 v1 replies in the author's corpus contained U+FEFF
followed by a median of 296 words; 78 verified as verbatim unquoted echoes
of the incoming email (a mail-client artifact with no quote markers). The
v1 model learned to reproduce the echo, and the corpus stats were fiction:
real reply median is 39 words, not 75; p90 is 135, not 490. v2 dataset: 363
pairs, 338/25 split, 0 contamination. Consequence: v1 eval numbers are not
comparable to v2.

**Lesson:** validate cleaning against generation behavior, not just scans of
known patterns — the model surfaced a contamination class the leak checks
missed.

## 2026-07-07 — Final adapter: epoch 4 (not checkpoint-120 / epoch 3)

**Decision:** the epoch-4 final adapter (in `models/adapter`) is the
production adapter. Earlier checkpoints stay on disk for reference but are
not deployed.

**Evidence:** Eval loss was flat between epochs 3 and 4 (1.0661 vs 1.0656),
which made epoch 3 look attractive on paper (earlier stop, less overfitting
risk). But behavioral comparison on the 27-pair eval set showed epoch 3 had
not converged on length: median generated reply was 259 words vs 67 for
epoch 4, against an actual median of 90. Median absolute length gap vs the
real replies: 245 words (epoch 3) vs 69 (epoch 4). Style-transfer quality was
equivalent (1/27 assistant-style responses for both, vs 25/27 for the base
model). Full outputs: `data/comparison-epoch3.md` vs `data/comparison.md`
(gitignored - contain personal email content).

**Lesson:** eval loss alone was not a sufficient checkpoint-selection signal;
the last epoch did real work on length calibration that loss didn't surface.
Compare checkpoints behaviorally before choosing.
