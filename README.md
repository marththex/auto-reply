# auto-reply

Fine-tunes Gemma 4 E2B (Unsloth LoRA) on your own sent mail so it can draft
Gmail replies in your voice. Drafts land in your Gmail drafts folder via the
API for human review — **nothing is ever auto-sent**.

```
Takeout mbox ─▶ build-pairs ─▶ pairs.jsonl ─▶ make-split ─▶ train/eval
                                                              │
                                              train-lora (Unsloth LoRA) ─▶ adapter
                                                              │
Gmail inbox ─▶ draft-replies ─▶ filter ─▶ generate (+facts.yaml) ─▶ Gmail draft
                                   │                                    │
                                   └────────── SQLite feedback DB ◀─────┘
```

## Read this first: limitations and hard rules

- **Draft-only, never auto-send — keep it that way.** A model fine-tuned on
  your mail *sounds* exactly like you while confidently inventing personal
  facts — commitments, dates, relationships (confirmed failure mode of this
  project, not a hypothetical). Every draft needs human review before
  sending. No code path here calls Gmail send; the requested OAuth scopes
  (`gmail.readonly` + `gmail.compose`) cannot send mail. Do not add send
  scopes.
- **Your trained adapter is private data.** It is trained on your personal
  email and can reproduce personal information verbatim. Treat the adapter
  (and its GGUF exports) like the mail itself: never publish, commit, or
  upload the weights. The .gitignore already excludes `models/`, `data/`,
  `facts.yaml`, `allowlist.yaml`, and `credentials/` — keep it that way
  when you add files.
- **Facts grounding is a mitigation, not a fix.** `facts.yaml` reduces
  invented details; it does not eliminate them.
- **Email cleaning is adversarial.** Real mail clients produce artifacts
  you won't anticipate (this corpus had unquoted echoes of the original
  after a stray BOM in 29% of replies). Expect to add cleaning patterns for
  your own corpus — test-drive them in `tests/test_cleaning.py`.

## Prerequisites

- Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).
- **Training only**: an NVIDIA GPU with ~16 GB VRAM (development happened on
  an RTX 5080; 4-bit + LoRA via Unsloth fits in 16 GB). Inference can run
  CPU-only via llama.cpp.
- A Hugging Face account that has accepted Google's Gemma license — the
  base model downloads from HF, and the weights carry Google's own terms
  (see the note in LICENSE).
- A Google Cloud project for Gmail OAuth (free): Desktop-app credentials,
  no publishing required.

## Deployment options

- **A — one machine.** Train, generate, and draft on the GPU box.
  `draft-replies --backend local` and you're done; run it manually or on a
  schedule.
- **B — GPU box + always-on CPU host.** Train on the GPU box, export the
  adapter to GGUF, and serve it with llama.cpp on any Linux Docker host (a
  NAS VM, spare mini-PC, etc. — 2-10 s per draft on 9 modest vCPUs). The
  bridge then runs on that host every 5 minutes. Setup: [nas/README.md](nas/README.md).

## From clone to first draft

```sh
git clone https://github.com/marththex/auto-reply && cd auto-reply
uv sync                    # code + dev tools (no GPU deps)
uv run pytest              # should be all green
```

**1. Personal config** (both gitignored — copy the examples and edit):

```sh
cp facts.example.yaml facts.yaml           # personal facts for grounding
cp allowlist.example.yaml allowlist.yaml   # automated-sender filter exceptions
```

`identity.name` in facts.yaml matters twice: its first name becomes the
persona the prompts address ("Write the reply Jane would send…"), and the
whole file is injected into inference prompts so the model stops inventing
personal details. Without a facts file the prompts fall back to "the user"
and a warning is logged.

**2. Data pipeline.** Export your mail from
[Google Takeout](https://takeout.google.com) (Mail → All Mail: pairing
needs incoming messages too, not just Sent).

```sh
uv run build-pairs "path/to/export.mbox" --out data/pairs.jsonl
uv run make-split data/pairs.jsonl        # refuses to re-randomize; --force to override
```

Cleaning strips quoted chains, signatures, device boilerplate, and several
real-world artifacts (see tests/test_cleaning.py for the catalog). Skim
`data/pairs.jsonl` before training — your corpus will have artifacts this
one didn't.

**3. Training + checkpoint choice** (GPU required):

```sh
uv sync --group train                      # torch/unsloth (Windows: cu128 index is pinned)
uv run train-lora                          # LoRA r16, 4 epochs, checkpoint per epoch
uv run compare-replies --out data/comparison.md --max-seq-len 4096
uv run compare-replies --adapter models/adapter/checkpoints/checkpoint-N ...
```

`compare-replies` writes incoming / base / fine-tuned / actual side by
side. **Judge checkpoints on this, not eval loss** — loss was flat across
two epochs here while generated replies were 4× too long (see
DECISIONS.md). `--facts facts.yaml` tests grounding; `--limit N` for quick
passes.

**4. Gmail OAuth** (one-time):

1. Google Cloud Console → new project → enable the Gmail API.
2. OAuth consent screen: External, add yourself as a test user
   (testing-mode tokens expire weekly; re-run `gmail-auth` when they do).
3. Credentials → OAuth client ID → **Desktop app** → save the JSON as
   `credentials/client_secret.json`.
4. `uv run gmail-auth` (opens a browser; caches `credentials/token.json`).

Scopes: `gmail.readonly` + `gmail.compose`. Nothing can send.

**5. Draft:**

```sh
uv run draft-replies --dry-run    # generate + print only, no drafts created
uv run draft-replies              # create real Gmail drafts (still never sends)
```

Per message: an automated-sender filter (noreply patterns → bulk/auto
headers → automated phrases → scored marketing heuristics, with an
allowlist override) rejects before any model call; skips and drafts are
logged to the SQLite feedback DB (`data/autoreply.db`).

## Operations

Every run starts by **reconciling prior drafts** (one API call per pending
draft, no mailbox scans), then skips already-processed messages so nothing
is drafted twice. Outcome states:

| Status | Meaning | Training use |
|---|---|---|
| `pending` | draft sitting in Gmail | — |
| `sent_unedited` | sent as generated | training pair (sent text = ground truth) |
| `sent_edited` | edited, then sent | best training pair — the edits teach |
| `deleted` | discarded (incl. trashed) | excluded; kept for quality review |
| `dry_run` | never saved to Gmail | excluded from everything |

- **Weekly token expiry**: testing-mode OAuth tokens die after ~7 days —
  every scheduled run starts failing at once (obvious in the log). Re-run
  `uv run gmail-auth` and copy `credentials/token.json` back to wherever
  the bridge reads credentials. Publishing the OAuth app would remove the
  expiry but triggers Google's restricted-scope review.
- **Audit**: the `runs` table records a summary row per run; `skipped`
  records every filtered message with the reason.
- **Retrain cycle**: once enough sent outcomes accumulate, rebuild pairs
  (sent text is ground truth, edited drafts are the best signal), retrain,
  compare checkpoints behaviorally, then export + swap the adapter.

## Architecture

```
src/autoreply/
  pipeline/     mbox -> clean (incoming, reply) pairs   [build-pairs]
  training/     split, chat formatting, LoRA training,
                checkpoint comparison, GGUF export      [make-split, train-lora,
                                                         compare-replies, export-gguf]
  gmail/        OAuth, drafts, sender filter, payload
                parsing, inbox->draft bridge            [gmail-auth, draft-replies]
  facts.py      personal-facts prompt grounding (facts.yaml)
  generation.py shared model load + generate (local GPU or llama.cpp server)
  db.py         SQLite feedback store (emails/drafts/sent_replies/skipped/runs)
nas/            docker compose + schedule templates for a CPU inference host
.github/        ci.yml (ruff+pytest), deploy.yml (SSH deploy, dormant by default)
```

Design notes worth knowing before changing things: prompt bytes are pinned
(tests/test_prompt_pinning.py) because the deployed adapter is sensitive to
drift; the DB draft text must byte-equal the Gmail draft (signature
appended before recording) or reconciliation misclassifies edits; SQLite
stays on host-local disk. The full decision log with evidence is in
DECISIONS.md; conventions and hard-won gotchas in CLAUDE.md.

## License

MIT — see [LICENSE](LICENSE). Gemma model weights (and adapters derived
from them) are licensed separately by Google; you accept the Gemma Terms of
Use when downloading the base model from Hugging Face.
