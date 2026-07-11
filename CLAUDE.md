# auto-reply — project conventions and context

Fine-tunes Gemma 4 E2B (Unsloth LoRA) to draft Gmail replies in the
mailbox owner's writing voice. Drafts are saved via the Gmail API for human
review. See README.md for setup, DECISIONS.md for the decision log.

## Hard rules

- **Draft-only, never auto-send.** No code path may call Gmail send. The
  model confabulates personal facts (confirmed failure mode); a human reviews
  every draft. `gmail.compose` scope only - never request send scopes.
- **Personal data never reaches git**: mbox exports, `data/*.jsonl`,
  comparison outputs, `facts.yaml`, `allowlist.yaml`, `credentials/`, model
  weights (adapters are trained on private email - treat them like the mail
  itself). The .gitignore encodes this; keep it that way when adding files.
- Ask before touching a real inbox, real secrets, or a live deployment.

## Key decisions and why

- **SQLite over anything heavier**: single user, single writer, one file;
  the feedback loop needs durability, not concurrency.
- **Unsloth over vanilla HF/PEFT**: 2x speed and fits 16 GB VRAM with
  4-bit + LoRA. Works on native Windows (see gotchas below).
- **Manual training trigger**: a ~380-pair dataset trains in ~15 min -
  automation adds risk (accidental retrain on bad data), no cadence to
  justify it.
- **Facts grounding at inference only**: the adapter was trained without a
  facts block; `to_messages` (training) must keep that exact shape while
  `to_prompt_messages` (inference) injects facts. Retraining with facts baked
  in is a possible future step.
- **Prompt bytes are pinned**: the persona name is derived from facts.yaml
  identity.name, and tests/test_prompt_pinning.py pins the rendered bytes -
  a deployed adapter is sensitive to any prompt drift. Don't edit the pinned
  strings to make a refactor pass.
- **Deterministic eval split, overwrite-guarded**: `make-split` refuses to
  re-randomize without `--force`, so eval stays comparable across runs.
  Re-splitting is only justified when the underlying data changes (as in the
  BOM-echo fix - see DECISIONS.md).
- **Checkpoint choice is behavioral, not loss-based**: eval loss missed a 4x
  length miscalibration between epochs. Always run `compare-replies` on
  candidate checkpoints.
- **Model artifacts move over a file share, never git/registry**: the repo
  deploys code only; the inference host mounts the adapter read-only.

## Gotchas that cost real debugging time

- `import unsloth` must come BEFORE datasets/trl in any training entry point
  - unpatched trl prep breaks `train_on_responses_only` marker matching
  (everything masks to -100).
- Gemma 4 E2B is multimodal: loaders return a `Gemma4Processor`, not a
  tokenizer. Render chat templates with `tokenize=False` and tokenize the
  string separately; `apply_chat_template(tokenize=True)` crashes on plain
  string content.
- Gemma 4 turn markers are `<|turn>user\n` / `<|turn>model\n` (NOT Gemma 3's
  `<start_of_turn>`), and generation needs `use_cache=True`.
- torch on Windows must come from the cu128 index (pinned in pyproject);
  unpinned resolution silently downgrades unsloth below Gemma 4 support.
- To load a second model on 16 GB: `del` your model refs, then
  `generation.free_vram()`. Passing refs into a helper that dels them frees
  nothing.
- Gmail's `drafts.get` returns TRASHED drafts as if alive - reconciliation
  must check the TRASH label, not just 404 (bit us on the first live test).
- Network-share permissions bite twice: files copied from another OS can
  land with an owner the container can't read (chmod on the host fixes it),
  and systemd's `StandardOutput=append:` opens the log as root, which NFS
  root-squash turns into an unreadable anonymous-uid file (redirect inside
  ExecStart's shell instead - see nas/systemd/).
- INVARIANT: the draft text stored in the DB must byte-equal the Gmail
  draft (signature appended BEFORE record_draft) - reconciliation's
  sent_unedited/sent_edited distinction depends on it.
- API-created drafts don't get the Gmail signature; the bridge fetches the
  default sendAs signature per run and appends it. gmail.readonly covers
  the settings read.
- Keep the scheduled bridge's SQLite DB on host-local disk, NOT a network
  share: SQLite + two share-client hosts = lock corruption risk. If several
  machines run the CLI, decide which DB is canonical.
- Email cleaning is adversarial; artifacts found in the author's corpus
  (~380 pairs) include BOM-separated unquoted echoes of the original (29%
  of replies!), fused mid-line attributions, U+200A inside addresses. TDD
  every new pattern in tests/test_cleaning.py.
- The facts block is an attractor: image-only marketing HTML reduces to a
  near-empty body under html_to_text (text nodes only - no alt/href), and
  generation with nothing to condition on parrots facts.yaml into the draft
  (one live 10-char body did this 4/4 generations). Defenses, in order: the
  filter's MIN_BODY_WORDS layer, conditions on situational facts sections
  (gates are dampers, not switches, at 2B scale - A/B evidence in
  DECISIONS.md 2026-07-11), human review.

## Known future steps

1. Retrain once enough sent outcomes accumulate: build pairs from
   sent_replies (final text = ground truth), include the facts block in
   TRAINING prompts (removes the current inference-only mismatch), and
   downweight the corpus's dominant reply genre (>50% of v2 reply tokens -
   see DECISIONS.md 2026-07-11) so one topic can't become the fallback mode.
2. Gmail push notifications via GCP Pub/Sub (pull subscription works
   without a public endpoint) to replace polling.
3. deploy.yml activation: set the 4 NAS_* secrets and an SSH hostname for
   your Docker host (workflow stays dormant until then).
4. Weekly OAuth token refresh is manual in testing mode; publishing the
   OAuth app would remove it but triggers restricted-scope review.

## Layout

```
src/autoreply/
  pipeline/     mbox -> clean (incoming, reply) pairs   [build-pairs]
  training/     split, chat formatting, LoRA training,
                checkpoint comparison, GGUF export      [make-split, train-lora,
                                                         compare-replies, export-gguf]
  gmail/        OAuth, drafts, sender filter, payload
                parsing, inbox->draft bridge            [gmail-auth, draft-replies]
  facts.py      personal-facts prompt grounding (facts.yaml)
  generation.py shared model load + generate
  db.py         SQLite feedback store (emails/drafts/sent_replies/skipped/runs)
data/           mbox, jsonl, comparisons, autoreply.db  (gitignored)
models/         adapters + checkpoints                  (gitignored, share-synced)
nas/            compose + schedule templates for a CPU inference host
.github/        ci.yml (ruff+pytest), deploy.yml (SSH deploy, dormant)
```

## Conventions

- TDD: failing test first for anything with logic (cleaning, filter,
  pairing, db). GPU scripts are exempt but get smoke-tested before full runs.
- Tests never require GPU deps or network; CI runs `uv sync` (dev group
  only) + ruff + pytest.
- uv for everything: `uv run <entry-point>`, `uv sync --group train` for GPU
  work. Windows torch comes from the cu128 index automatically.
- mypy is intentionally NOT in CI yet (2026-07 decision: adopt later).
