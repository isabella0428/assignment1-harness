"""Problem (priority): classify one email as urgent, normal, or ignore."""

from __future__ import annotations

import json
import random
import re
import time
from cs329z_hw1.llm import BudgetExceeded, ScriptExhausted
from cs329z_hw1.pipelines.utils import format_email
from cs329z_hw1.types import Email, LMCallable

LABELS = ("urgent", "normal", "ignore")
MAX_ATTEMPTS = 7
RETRY_SLEEP_S = 3.0  # waits ~3, 6, 12, 24, 48, 60 s between attempts (about 2.5 min in total)
MAX_RETRY_SLEEP_S = 60.0

PRIORITY_SYSTEM = """\
You triage email for one person: the recipient named in the To field. Decide what that person \
should do with this email at the moment it was sent, and give it exactly one category.

CATEGORIES
urgent: BOTH of these must be true:
  (1) The email asks the recipient to do or decide something, or reports a live problem the \
recipient is responsible for (an outage, a failed payment, a customer or manager escalating an issue).
  (2) There is time pressure: the email states OR CLEARLY IMPLIES that the action is needed the \
same day, the next working day, or by a deadline at most two working days after the email's date. \
Working days are Monday to Friday. Example: for an email sent on Friday, Monday is working day 1 \
and Tuesday is working day 2.
  Implied time pressure counts. "ASAP", "as soon as possible", "right away", "today", "tonight", \
"this morning", "before the meeting", or an emphatic personal plea for an immediate favor \
("Could you PLEASE call him and get his email? I need to reach him") all mean the action is \
needed now, so a direct request phrased that way is urgent even without a clock time.
  Emphasis counts too. When a person writing to this recipient asks them to do something and \
stresses it with capitalized words ("PLEASE", "NEED", "TODAY", "ASAP", "IMPORTANT"), repeated \
words, or several exclamation marks, treat that as a sign the sender needs it right away, and \
lean toward urgent. Emphasis only strengthens condition (2): it never turns an email that asks \
nothing of the recipient into urgent, and capitals in ads, newsletters or mass mailings mean nothing.

normal: Legitimate work or personal mail written for this recipient or a small working group, with \
no deadline in the next two working days. This covers status updates and FYIs, ordinary questions, \
documents sent for review, scheduling for later dates, replies in an ongoing discussion, and \
personal notes from friends, family or colleagues.

ignore: Mail the recipient can leave unread at no cost. This covers advertising, spam, newsletters, \
news digests, announcements to large employee lists (events, training, facilities notices), and \
automated notifications that ask nothing of the recipient.

RULES FOR HARD CASES
- The wording does not decide the category. Words like "URGENT", "act now" or "important" in an ad \
or a mass mailing still mean ignore. An email with URGENT in the subject that asks nothing of \
the recipient is not urgent. A calm "I need your signature by 3:00 PM today" is urgent.
- A request with no deadline and no sense of immediacy, or with a deadline more than two working \
days away, is normal.
- A date or time in the email is a deadline only if THE RECIPIENT'S action must be done by then. \
Dates of someone else's meeting, when a document or chart will arrive, or when a later step will \
happen are context, not deadlines. Ask: by when must the recipient themselves act? If the \
answer is after the next two working days, or the email does not say, the request is normal.
- A notice that something has changed (a new email address, phone number, office, contact person \
or procedure) is normal even when it says "effective immediately" or "starting today": that phrase \
says when the change starts, not that the recipient must act soon.
- An automated notice that asks this specific recipient to act (for example, an approval waiting \
on them) is normal, or urgent if it states a deadline within two working days.
- A short personal or social note from someone the recipient knows is normal, not ignore.
- Many recipients, a distribution-list address, or a greeting like "To all employees" suggest a \
mass mailing.

OUTPUT
Reply with a single JSON object and nothing else: no code fences, no text before or after it.
{"category": "<urgent, normal, or ignore>", "reason": "<one sentence naming the specific request \
and deadline, or why nothing is asked of the recipient>"}"""

def parse_priority_reply(reply: str) -> dict:
    """Turn the model's reply into {"category", "reason"}; never raises."""
    text = (reply or "").strip()

    # Cut out the {...} part in case the model added fences or a sentence around it.
    match = re.search(r"\{.*\}", text, re.S)
    try:
        obj = json.loads(match.group(0) if match else text)
        category = str(obj.get("category", "")).strip().lower()
        reason = str(obj.get("reason", "")).strip()
    except (ValueError, AttributeError, TypeError):  # not valid JSON, or not a dict
        category, reason = "", ""

    # Fallback for a reply that is not JSON but still names a label.
    if category not in LABELS:
        word = re.search(r"\b(urgent|normal|ignore)\b", text, re.I)
        category = word.group(1).lower() if word else "unknown"

    if not reason:
        reason = text or "The model reply stated no category."
    return {"category": category, "reason": reason}


def run_priority(email: Email, lm: LMCallable) -> dict:
    messages = [
        {"role": "system", "content": PRIORITY_SYSTEM},
        {"role": "user", "content": format_email(email)},
    ]
    error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            return parse_priority_reply(lm(messages))
        except (BudgetExceeded, ScriptExhausted) as e:  # retrying cannot help
            error = e
            break
        except Exception as e:  # transient API error, rate limit, OutputTruncated, ...
            error = e
            if attempt < MAX_ATTEMPTS - 1:
                delay = min(RETRY_SLEEP_S * 2**attempt, MAX_RETRY_SLEEP_S)
                # jitter so parallel calls that failed together don't all retry at the same moment
                time.sleep(delay * random.uniform(0.75, 1.25))
    # The function must not raise; say what went wrong in the reason.
    return {"category": "unknown", "reason": f"Model call failed: {type(error).__name__}: {error}"[:300]}
