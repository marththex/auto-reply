"""Gmail API OAuth via the desktop-app flow.

Requires credentials/client_secret.json (downloaded from Google Cloud Console,
see README). The resulting token is cached at credentials/token.json. Both
paths are gitignored.
"""

from pathlib import Path

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]
CREDENTIALS_DIR = Path("credentials")
CLIENT_SECRET_PATH = CREDENTIALS_DIR / "client_secret.json"
TOKEN_PATH = CREDENTIALS_DIR / "token.json"


def get_credentials() -> Credentials:
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_token(creds)
            return creds
        except RefreshError:
            pass  # token revoked or stale: fall through to a fresh consent flow
    creds = _run_consent_flow()
    _save_token(creds)
    return creds


def gmail_service():
    return build("gmail", "v1", credentials=get_credentials())


def _run_consent_flow() -> Credentials:
    if not CLIENT_SECRET_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CLIENT_SECRET_PATH}. Download an OAuth 'Desktop app' client "
            "secret from Google Cloud Console and save it there (see README)."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_PATH), SCOPES)
    return flow.run_local_server(port=0)


def _save_token(creds: Credentials) -> None:
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")


def main() -> None:
    """CLI (gmail-auth): run the OAuth flow and confirm the authorized account."""
    service = gmail_service()
    profile = service.users().getProfile(userId="me").execute()
    print(f"Authorized as {profile['emailAddress']}")
    print("Scopes: " + ", ".join(SCOPES))


if __name__ == "__main__":
    main()
