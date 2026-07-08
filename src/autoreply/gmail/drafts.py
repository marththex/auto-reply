"""Create Gmail drafts - the output end of the auto-reply pipeline."""

import base64
import re
from email.message import EmailMessage

from autoreply.pipeline.cleaning import html_to_text


def fetch_signature(service) -> str:
    """The default send-as address's signature, rendered to plain text.

    API-created drafts don't get the Gmail signature automatically (it's a
    compose-window feature), so the bridge appends it explicitly.
    """
    sendas = service.users().settings().sendAs().list(
        userId="me").execute().get("sendAs", [])
    return pick_default_signature(sendas)


def pick_default_signature(sendas: list[dict]) -> str:
    for entry in sendas:
        if entry.get("isDefault") and entry.get("signature"):
            text = html_to_text(entry["signature"]).strip()
            # HTML paragraph spacing renders as blank-line runs; a signature
            # block should be compact.
            return re.sub(r"\n{2,}", "\n", text)
    return ""


def append_signature(body: str, signature: str) -> str:
    if not signature:
        return body
    return body.rstrip() + "\n\n" + signature


def create_draft(
    service,
    *,
    to: str,
    subject: str,
    body: str,
    thread_id: str | None = None,
    in_reply_to: str | None = None,
) -> dict:
    """Create a draft in the authorized account.

    Pass thread_id (Gmail API thread id) and in_reply_to (the incoming
    message's Message-ID header) to attach the draft as a reply in its thread.
    Returns the Gmail API draft resource.
    """
    msg = EmailMessage()
    msg["To"] = to
    msg["Subject"] = subject
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
        msg["References"] = in_reply_to
    msg.set_content(body)

    draft = {"message": {"raw": base64.urlsafe_b64encode(msg.as_bytes()).decode()}}
    if thread_id:
        draft["message"]["threadId"] = thread_id
    return service.users().drafts().create(userId="me", body=draft).execute()
