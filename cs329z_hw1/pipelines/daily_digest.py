"""Problem (daily_digest): a morning summary of one day's email.

Code does the aggregation:
  1. classify every email concurrently with run_priority (unknown counts as normal),
  2. drop the ignore pile,
  3. keep one item per thread,
  4. order the items by urgency.
One final LLM call then writes the text from that ordered list.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from cs329z_hw1.pipelines.priority import run_priority
from cs329z_hw1.pipelines.utils import format_email
from cs329z_hw1.types import Email, LMCallable

PRIORITY_WORKERS = 8      # parallel classification calls per day
RANK = {"urgent": 0, "normal": 1}  # lower = more urgent; unknown is treated as normal

MAX_WORDS = 300           # the test's hard limit (len(digest.split()))
TARGET_WORDS = 200        # what we ask for, leaving room because models miscount
MIN_LINE_WORDS = 5        # an urgent line still needs "<name>: <action>"
MAX_LINE_WORDS = 20
URGENT_BODY_CHARS = 3000  # urgent items keep most of their body
OTHER_BODY_CHARS = 500    # other items only need enough to mention them
SHORTEN_ATTEMPTS = 2      # extra model calls to rewrite a digest that is too long
EMPTY_DIGEST = "No emails that need your attention today."

DIGEST_SYSTEM = """\
You write a short morning digest of one person's email from one day. You receive a numbered list \
of items, one per email thread, already ordered from most to least urgent. Each item is marked \
[URGENT] (needs action within about two working days) or [OTHER] (ordinary mail). Junk has \
already been removed.

Write the digest in exactly this shape:

Urgent:
- <one line per [URGENT] item, in the given order: the company, project, site or counterparty it \
is about, by name, then what the reader must do and by when>

Other:
- <one short line per [OTHER] item, in the given order, for as many as fit in the word budget; \
skip the rest>

RULES
- HARD LIMIT: stay within the word budget given with the items. Count before you answer.
- Every [URGENT] item gets its own line, and that line must name the company, project or site the \
email is about (for example "Acme Pipeline: approve the corrected invoice by 2 PM today"). With \
many urgent items, use just "<name>: <action>, <deadline>".
- Never drop an [URGENT] item to save words; drop [OTHER] items from the end of the list instead.
- Keep the order of the list. Do not merge items or regroup them by topic.
- If there are no items of a kind, write "None." under that heading.
- Plain text only. No greeting, no title, no closing remarks."""

SHORTEN_SYSTEM = """\
You shorten a morning email digest. Keep the "Urgent:" and "Other:" headings. Keep every urgent \
line and the company, project or site name on it; shorten those lines and drop lines from the end \
of the Other section first. Reply with the shortened digest only."""


def daily_digest(emails: list[Email], lm: LMCallable) -> str:
    if not emails:
        return EMPTY_DIGEST

    # 1. Classify concurrently: the calls are independent. pool.map keeps the input order.
    with ThreadPoolExecutor(max_workers=PRIORITY_WORKERS) as pool:
        labels = list(pool.map(lambda e: run_priority(e, lm)["category"], emails))

    # 2-4. Drop ignore, one item per thread, order by urgency.
    items = _aggregate(emails, labels)
    if not items:
        return EMPTY_DIGEST

    messages = [
        {"role": "system", "content": DIGEST_SYSTEM},
        {"role": "user", "content": _render_items(items)},
    ]
    try:
        digest = lm(messages)
    except Exception:  # the function must always return a string
        return EMPTY_DIGEST
    return _within_limit(digest, lm)


def _aggregate(emails: list[Email], labels: list[str]) -> list[dict]:
    """Return one item per thread, most urgent first.

    Each item is {"email", "label", "count"}. A thread takes its most urgent
    label; the email shown for it is the most urgent one, the latest among ties.
    """
    threads: dict[str, dict] = {}
    for email, label in zip(emails, labels):
        if label == "ignore":
            continue
        label = label if label in RANK else "normal"  # unknown -> normal
        key = email.get("thread_id") or email.get("id") or str(id(email))
        item = threads.get(key)
        if item is None:
            threads[key] = {"email": email, "label": label, "count": 1}
            continue
        item["count"] += 1
        more_urgent = RANK[label] < RANK[item["label"]]
        same_but_later = RANK[label] == RANK[item["label"]] and (email.get("date") or "") > (item["email"].get("date") or "")
        if more_urgent or same_but_later:
            item["email"], item["label"] = email, label

    return sorted(threads.values(), key=lambda it: (RANK[it["label"]], it["email"].get("date") or ""))


def _render_items(items: list[dict]) -> str:
    n_urgent = sum(it["label"] == "urgent" for it in items)
    n_other = len(items) - n_urgent
    other_words = 60 if n_urgent < 8 else 30
    per_line = (TARGET_WORDS - other_words) // max(n_urgent, 1)
    per_line = max(MIN_LINE_WORDS, min(MAX_LINE_WORDS, per_line))

    header = (
        f"WORD BUDGET: at most {TARGET_WORDS} words in total. Each of the {n_urgent} urgent lines at "
        f"most {per_line} words. The Other section at most {other_words} words in total "
        f"({n_other} other items; list them in order until the budget runs out).\n"
    )
    blocks = []
    for i, it in enumerate(items, start=1):
        tag = "URGENT" if it["label"] == "urgent" else "OTHER"
        limit = URGENT_BODY_CHARS if tag == "URGENT" else OTHER_BODY_CHARS
        thread = f" ({it['count']} messages in this thread; the most urgent, latest one is shown)" if it["count"] > 1 else ""
        blocks.append(f"[{i}] [{tag}]{thread}\n{format_email(it['email'], limit)}")
    return header + "\n\n" + "\n\n----------\n\n".join(blocks)


def _within_limit(digest: str, lm: LMCallable) -> str:
    """If the model wrote more than MAX_WORDS words, ask it to rewrite shorter.

    The test counts the words the model wrote, so the text is never cut in code.
    """
    for _ in range(SHORTEN_ATTEMPTS):
        n = len(digest.split())
        if n <= MAX_WORDS:
            break
        try:
            shorter = lm([
                {"role": "system", "content": SHORTEN_SYSTEM},
                {"role": "user", "content": f"This digest has {n} words. Rewrite it in at most "
                                            f"{TARGET_WORDS} words.\n\n{digest}"},
            ])
        except Exception:
            break
        if shorter and shorter.strip():
            digest = shorter
    return digest
