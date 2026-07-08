"""Inbox -> draft bridge: for each unread message that passes the
automated-sender filter, generate a reply with the LoRA adapter and save it
as a Gmail draft. NEVER sends - drafts only, reviewed by a human.

Usage: draft-replies [--dry-run] [--limit 10] [--query "is:unread in:inbox"]
"""

import argparse
import logging
import os
from pathlib import Path

import yaml

from autoreply import db
from autoreply.facts import DEFAULT_FACTS_PATH, load_facts, persona_name
from autoreply.gmail.auth import gmail_service
from autoreply.gmail.drafts import append_signature, create_draft, fetch_signature
from autoreply.gmail.filter import should_skip
from autoreply.gmail.inbox import extract_message
from autoreply.gmail.reconcile import reconcile
from autoreply.pipeline.cleaning import clean_reply_body

DEFAULT_ADAPTER = os.environ.get("AUTOREPLY_ADAPTER", "models/adapter")
DEFAULT_ALLOWLIST_PATH = Path("allowlist.yaml")

log = logging.getLogger("autoreply.bridge")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate reply drafts for unread mail. Drafts only, never sends."
    )
    parser.add_argument("--query", default="is:unread in:inbox",
                        help="Gmail search query selecting messages to draft for")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate and print replies; do not create Gmail drafts")
    parser.add_argument("--backend", choices=["local", "remote"],
                        default=os.environ.get("AUTOREPLY_BACKEND", "local"),
                        help="local = GPU inference via unsloth on this machine; "
                             "remote = llama.cpp server (NAS)")
    parser.add_argument("--endpoint",
                        default=os.environ.get("AUTOREPLY_ENDPOINT", "http://localhost:8080"),
                        help="llama.cpp server URL for --backend remote")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER,
                        help="Adapter dir for --backend local")
    parser.add_argument("--facts", default=str(DEFAULT_FACTS_PATH))
    parser.add_argument("--allowlist", default=str(DEFAULT_ALLOWLIST_PATH))
    parser.add_argument("--db", dest="db_path", default=str(db.DEFAULT_DB_PATH))
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--max-incoming-chars", type=int, default=6000,
                        help="Truncate incoming bodies beyond this before prompting "
                             "(~1500 tokens; keeps prompt + output inside the context)")
    parser.add_argument("--max-drafts-per-run", type=int, default=3,
                        help="Safety cap for unattended runs: stop after creating "
                             "this many drafts, leave the rest for the next run")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    service = gmail_service()
    conn = db.connect(args.db_path)

    # Always resolve prior draft outcomes before touching new mail.
    outcomes = reconcile(service, conn)
    log.info("reconciliation: %s", outcomes)

    listing = service.users().messages().list(
        userId="me", q=args.query, maxResults=args.limit).execute()
    ids = [m["id"] for m in listing.get("messages", [])]
    log.info("query %r matched %d message(s)", args.query, len(ids))
    if not ids:
        return

    already_processed = db.processed_email_ids(conn)
    allowlist = _load_allowlist(args.allowlist)

    kept = []
    n_processed_skips = n_filtered = 0
    for gmail_id in ids:
        if gmail_id in already_processed:
            log.info("SKIP %s: already processed (draft exists or resolved)", gmail_id)
            n_processed_skips += 1
            continue
        raw = service.users().messages().get(
            userId="me", id=gmail_id, format="full").execute()
        message = extract_message(raw)
        verdict = should_skip(
            sender=message["sender"], headers=message["headers"],
            subject=message["subject"], body=message["body"], allowlist=allowlist,
        )
        if verdict.skip:
            log.info("SKIP %s <%s> %r: %s", gmail_id, message["sender"],
                     message["subject"][:60], verdict.reason)
            db.record_skip(conn, gmail_id=gmail_id, sender=message["sender"],
                           reason=verdict.reason)
            n_filtered += 1
        else:
            kept.append(message)
    log.info("%d message(s) passed the filter", len(kept))

    capped = len(kept) > args.max_drafts_per_run
    if capped:
        log.warning("draft cap: %d passed, drafting only %d this run",
                    len(kept), args.max_drafts_per_run)
        kept = kept[: args.max_drafts_per_run]

    if not kept:
        db.record_run(conn, backend=args.backend, query=args.query,
                      matched=len(ids), filtered=n_filtered,
                      already_processed=n_processed_skips, drafted=0,
                      capped=False, reconciled=outcomes)
        return

    facts = load_facts(args.facts)
    if not facts:
        log.warning("no facts file at %s - drafts may invent personal details",
                    args.facts)
    name = persona_name(args.facts)
    signature = fetch_signature(service)

    if args.backend == "local":
        from autoreply.generation import generate_reply, load_model
        model, tokenizer = load_model(args.adapter, max_seq_len=args.max_seq_len)
        model_version = args.adapter
    else:
        from autoreply.generation import generate_reply_remote
        log.info("remote backend: %s", args.endpoint)
        model_version = f"remote:{args.endpoint}"

    for message in kept:
        record = {"incoming": {
            "from": message["sender"],
            "subject": message["subject"],
            "date": message["date"],
            "body": bounded_body(clean_reply_body(message["body"]),
                                 limit=args.max_incoming_chars),
        }}
        if args.backend == "local":
            reply = generate_reply(model, tokenizer, record, facts=facts,
                                   name=name, max_new_tokens=args.max_new_tokens)
        else:
            reply = generate_reply_remote(args.endpoint, record, facts=facts,
                                          name=name, max_new_tokens=args.max_new_tokens)
        # Append before recording so the stored text equals the Gmail draft -
        # reconciliation's sent_unedited comparison depends on that equality.
        reply = append_signature(reply, signature)
        db.record_email(conn, gmail_id=message["gmail_id"],
                        thread_id=message["thread_id"],
                        incoming_body=record["incoming"]["body"],
                        incoming_date=message["date"])
        if args.dry_run:
            db.record_draft(conn, email_id=message["gmail_id"], text=reply,
                            model_version=model_version, status="dry_run")
            print(f"\n=== DRY RUN: draft for {message['sender']} "
                  f"({message['subject']!r}) ===\n{reply}\n")
        else:
            created = create_draft(
                service,
                to=message["sender"],
                subject=_reply_subject(message["subject"]),
                body=reply,
                thread_id=message["thread_id"],
                in_reply_to=message["message_id_header"] or None,
            )
            db.record_draft(conn, email_id=message["gmail_id"], text=reply,
                            model_version=model_version,
                            gmail_draft_id=created.get("id"), status="pending")
            log.info("DRAFT created for <%s> %r", message["sender"],
                     message["subject"][:60])

    db.record_run(conn, backend=args.backend, query=args.query,
                  matched=len(ids), filtered=n_filtered,
                  already_processed=n_processed_skips, drafted=len(kept),
                  capped=capped, reconciled=outcomes)
    log.info("run summary: matched=%d filtered=%d already_processed=%d "
             "drafted=%d capped=%s reconciled=%s", len(ids), n_filtered,
             n_processed_skips, len(kept), capped, outcomes)


def bounded_body(text: str, *, limit: int) -> str:
    """Cap the incoming body so prompt + generation fit the context window.

    A 14k-char newsletter blew past the server's 4096-token context on the
    first real remote run; truncating the tail loses little (greetings and
    the ask are at the top) and keeps the request valid.
    """
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " [... truncated ...]"


def _reply_subject(subject: str) -> str:
    return subject if subject.lower().startswith("re:") else f"Re: {subject}"


def _load_allowlist(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return [str(entry) for entry in data] if data else []


if __name__ == "__main__":
    main()
