from cs329z_hw1.types import Email
from datetime import datetime

MAX_BODY_CHARS = 6000
MAX_TO_SHOWN = 5
MAX_ATTEMPTS = 7
RETRY_SLEEP_S = 3.0  # waits ~3, 6, 12, 24, 48, 60 s between attempts (about 2.5 min in total)
MAX_RETRY_SLEEP_S = 60.0

def _weekday(iso_date: str) -> str:
    try:
        return datetime.fromisoformat(iso_date.replace("Z", "+00:00")).strftime("%A")
    except (ValueError, AttributeError):
        return "unknown weekday"

def format_email(email: Email, max_body_chars: int = MAX_BODY_CHARS) -> str:
    """Render one email as readable headers plus body for the prompt."""
    date = email.get("date") or ""
    to = email.get("to") or []
    to_line = ", ".join(to[:MAX_TO_SHOWN])
    if len(to) > MAX_TO_SHOWN:
        to_line += f", ... ({len(to)} recipients in total)"
    body = (email.get("body") or "")[:max_body_chars]
    return (
        f"Date: {date} ({_weekday(date)})\n"
        f"From: {email.get('from') or ''}\n"
        f"To: {to_line}\n"
        f"Subject: {email.get('subject') or ''}\n"
        f"\n"
        f"{body}"
    )
