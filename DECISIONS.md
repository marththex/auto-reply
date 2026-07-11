# Decisions

Running log of project decisions and the evidence behind them. Newest first.

## 2026-07-11 — job_search grounding gated behind a relevance condition; employer fact retained

**Decision:** The `job_search` section key in facts.yaml now carries an
explicit condition ("mention ONLY if this email is from a recruiter or hiring
contact about a specific role"); the unconditional "never decline or downplay
interest" drafting rule moved inside that gated recruiter line; a "reply only
to what this email actually says" rule and the example file's `[placeholder]`
rule were added. The employer fact stays in the identity block.
facts.example.yaml models the gated shape.

**Evidence:** A run of drafts pitched job availability regardless of what the
incoming email said (including a coffee newsletter and an insurance survey).
The script's wording traced to the ungated job_search block — its status
phrase appeared verbatim in drafts and 0/363 times in the training corpus —
not to memorized training data. A/B on the production llama.cpp server
(3 facts variants × 5 synthetic probes × 2 samples each): gating cut
job-content leakage on survey-style probes to 0/2 (from 1/2) and produced no
hard declines of a genuine recruiter probe, while the ungated baseline
declined 1/2 *despite* an unconditional instruction never to. Removing the
employer fact eliminated employer leakage but caused worse confabulation:
asked "where do you work these days", the model substituted a relative's
employer from the relationships block instead of using a placeholder. Full
outputs: `data/comparison-facts-gating.md` (gitignored).

**Lesson:** at 2B scale an instruction gate is a damper, not a switch —
marketing-style probes triggered the job script in every variant, and the
pre-generation layers (sender filter, minimum-content skip — entry below) are
what actually keep that mail away from the model. Grounding facts can leak
into *any* draft: every always-on line in facts.yaml should either be safe in
an arbitrary reply or carry an explicit condition. The durable fix for the
topical prior is the planned retrain with facts in training prompts and the
dominant reply genre downweighted (>50% of v2 reply tokens).

## 2026-07-11 — Filter vocabulary extended after two live misses; near-empty bodies now skipped

**Decision:** Extended the automated-sender filter and added a minimum-content
layer. The marketing-subdomain vocabulary gains common ESP tokens
(`hello|hi|updates?|promos?|offers?|marketing|engage|connect`), both
do-not-reply layers (hard phrase + soft signal) now also match "respond",
the tracker-link regex learns `clicks.` link domains (the ESP's actual link
host; the old regex knew `link(s).` and `click.` but not `clicks.`), and
`should_skip` gains a final layer skipping bodies under 10 words
("insufficient content"). Allowlisted senders remain exempt from all layers.
Verified against the real leaked newsletter's raw source (which carried
**no** List-Unsubscribe/Precedence/Auto-Submitted headers at all — the
header layer never had a chance): it now scores 3 of the 2 needed signals.

**Evidence:** Two automated messages reached generation on the scheduled
bridge: a newsletter sent from a `hello.<brand>` ESP subdomain (token not in
the vocabulary; scored below the content threshold) and a satisfaction survey
whose boilerplate reads "please do not respond to this email" — the verb
"respond" defeated both do-not-reply layers, which only knew "reply". The
minimum-content layer addresses what happens after such a miss: image-only
marketing HTML reduces to footer boilerplate under `html_to_text` (which keeps
text nodes only), and a near-empty body gives the model nothing to condition
on — a live 10-character body produced the same off-topic scripted reply in
4/4 independent generations. Regression tests pin all three behaviors.

**Lesson:** enumerated vocabularies fail one synonym or subdomain token at a
time (second leak of this class — see the content-score layer's origin).
Every miss becomes a pinned test case, and the minimum-content layer plus
gated grounding (entry above) keep a future miss cheap instead of embarrassing.

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
