# CPU inference host (llama.cpp)

Serve the fine-tuned model over HTTP from any always-on Linux Docker host
(NAS VM, spare mini-PC, etc.) so drafting works without the GPU machine.
Base model and LoRA adapter stay **separate** (llama.cpp `--lora` at
runtime), so a retrain only re-copies a ~50 MB adapter, not multi-GB.

All paths and ports below come from environment variables with
repo-relative defaults — copy the repo-root `.env.example` to `nas/.env`
and adjust if your models, config, or DB live elsewhere (e.g. on a share).

## Layout under MODELS_DIR (default `../models`)

```
gemma-4-e2b-base-q5.gguf            # base model, downloaded once
adapters/<name>.gguf                # your exported adapter
adapters/<name>.metadata.json       # provenance: checkpoint, eval loss, date
```

The base model comes from `unsloth/gemma-4-E2B-it-GGUF` on Hugging Face
(Q5_K_M; drop to Q4_K_M if throughput disappoints — same repo).

## After every retrain (run on the GPU machine)

```sh
# one-time: git clone --depth 1 https://github.com/ggml-org/llama.cpp
uv run export-gguf --name my-lora-vN-epochM \
    --source-checkpoint checkpoints/checkpoint-XXX --eval-loss X.XXX \
    --llama-cpp path/to/llama.cpp
```

This filters the adapter to language-model tensors (the audio/vision tower
LoRA weights don't apply to text-only inference), converts to GGUF, and
writes it plus metadata to `AUTOREPLY_MODELS_DIR/adapters/` — point that at
whatever volume this host mounts. Then set `ADAPTER_GGUF=<name>.gguf` in
`nas/.env` and restart the container.

## Start the server (on the Docker host)

```sh
cd <repo-path>/nas
docker compose up -d
curl -s http://localhost:${LLAMA_PORT:-8080}/health
```

The server binds host port `LLAMA_PORT` (default 8080 — pick any free
port). `LLAMA_THREADS` tunes CPU threads: leave a few vCPUs free and
benchmark before settling; memory bandwidth usually saturates before all
cores are busy.

## Benchmark

```sh
docker run --rm -v <models-dir>:/models ghcr.io/ggml-org/llama.cpp:full \
  --run -m /models/gemma-4-e2b-base-q5.gguf -p "benchmark" -n 128 -t <threads> 2>&1 | tail -5
```

Look at the `eval time ... tokens per second` line; a typical draft is
50-150 tokens, so 8 tok/s ≈ 6-20 s per draft. Illustrative numbers from
development (12-vCPU VM, Q5_K_M, 9 threads): 18.2 tok/s generation, 63
tok/s prompt processing — 2-10 s per typical draft.

## Scheduled drafting (every 5 min — disabled until you enable it)

Setup on the host: clone the repo, then
`cd <repo-path>/nas && docker compose build bridge`. The bridge reads
facts.yaml, allowlist.yaml, and credentials/ from `CONFIG_DIR` (default:
the repo root; the OAuth token refresh rewrites token.json, so credentials
are mounted rw).

Enable with EITHER the cron line from `crontab.example` OR the systemd
templates in `systemd/` — instantiate the placeholders first, e.g.:

```sh
sed -e 's|<user>|me|; s|<repo-path>|/home/me/auto-reply|; s|<log-dir>|/home/me/auto-reply/logs|' \
    systemd/auto-reply.service | sudo tee /etc/systemd/system/auto-reply.service
sudo cp systemd/auto-reply.timer /etc/systemd/system/
sudo systemctl enable --now auto-reply.timer
```

- Each run: reconcile prior drafts → filter new mail → draft (cap 3/run).
- Summary per run in the `runs` DB table; raw output appends to the log
  path you chose.
- Keep `DATA_DIR` (the SQLite volume) on **host-local disk**, not a
  network share: SQLite with two client hosts risks lock corruption. If
  you also run the CLI on your GPU machine, that machine keeps its own
  local DB for experiments — decide which one is canonical.
- Testing-mode OAuth tokens expire weekly: scheduled runs then fail until
  `gmail-auth` is re-run and token.json re-copied. The run log makes this
  obvious (every run errors).

## Gotchas

- **Network-share permissions**: files copied onto a share from another
  OS can land with an owner the container can't read (NFS root-squash,
  uid mismatches). If the server can't read a freshly copied adapter,
  `chmod -R a+rX` the models directory on the host; the durable fix is the
  share's default-permission setting.
- Marketing mail without bulk headers (rare) passes the filter and
  produces a junk draft — known behavior, cost is one deleted draft. The
  incoming body is capped at ~6000 chars before prompting so long
  newsletters can't blow the 4096-token context (this returned a bare 400
  from the server on first contact).

## Point the bridge at it (from any machine)

```sh
uv run draft-replies --backend remote --endpoint http://<host>:8080
# or: AUTOREPLY_BACKEND=remote AUTOREPLY_ENDPOINT=http://<host>:8080
```

Local GPU mode remains the default (`--backend local`).
