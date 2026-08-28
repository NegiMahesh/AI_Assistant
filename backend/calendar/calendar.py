# =========================================================
# LOCAL CALENDAR TOOL — PHASE 14
# =========================================================

import json
import os
import re
import uuid
from datetime import datetime, timedelta


CALENDAR_FILE = os.path.join(
    os.path.dirname(__file__),
    "events.json"
)


# =========================================================
# STORAGE
# =========================================================

def _load_events():
    if not os.path.exists(CALENDAR_FILE):
        return []

    try:
        with open(CALENDAR_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []
    except Exception as error:
        print("Calendar load error:", error)
        return []


def _save_events(events):
    directory = os.path.dirname(CALENDAR_FILE)
    os.makedirs(directory, exist_ok=True)

    temp_file = CALENDAR_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(events, file, indent=2, ensure_ascii=False)

    os.replace(temp_file, CALENDAR_FILE)


# =========================================================
# DATE / TIME PARSING
# =========================================================

def _parse_time(text):
    match = re.search(
        r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
        text,
        re.IGNORECASE
    )

    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = (match.group(3) or "").lower()

    if meridiem:
        if hour < 1 or hour > 12:
            return None
        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
    elif hour > 23 or minute > 59:
        return None

    return hour, minute


def _parse_date(text, now=None):
    now = now or datetime.now()
    lowered = text.lower()

    if re.search(r"\btomorrow\b", lowered):
        return (now + timedelta(days=1)).date()

    if re.search(r"\btoday\b", lowered):
        return now.date()

    if re.search(r"\bday after tomorrow\b", lowered):
        return (now + timedelta(days=2)).date()

    # YYYY-MM-DD
    match = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if match:
        try:
            return datetime(
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3))
            ).date()
        except ValueError:
            return None

    # Month name + day, e.g. September 5 / 5 September
    months = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12,
    }

    match = re.search(
        r"\b(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|"
        r"august|aug|september|sep|sept|october|oct|november|nov|december|dec)"
        r"\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s*,?\s*(20\d{2}))?\b",
        lowered,
        re.IGNORECASE
    )
    if match:
        month = months[match.group(1).lower()]
        day = int(match.group(2))
        year = int(match.group(3)) if match.group(3) else now.year
        try:
            candidate = datetime(year, month, day).date()
            if not match.group(3) and candidate < now.date():
                candidate = datetime(year + 1, month, day).date()
            return candidate
        except ValueError:
            return None

    match = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
        r"(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|"
        r"august|aug|september|sep|sept|october|oct|november|nov|december|dec)"
        r"(?:\s*,?\s*(20\d{2}))?\b",
        lowered,
        re.IGNORECASE
    )
    if match:
        day = int(match.group(1))
        month = months[match.group(2).lower()]
        year = int(match.group(3)) if match.group(3) else now.year
        try:
            candidate = datetime(year, month, day).date()
            if not match.group(3) and candidate < now.date():
                candidate = datetime(year + 1, month, day).date()
            return candidate
        except ValueError:
            return None

    return None


def parse_event_datetime(text):
    now = datetime.now()
    event_date = _parse_date(text, now)
    event_time = _parse_time(text)

    if event_date is None:
        return None, "Please specify a date, such as today, tomorrow, or September 5."

    if event_time is None:
        event_time = (9, 0)

    hour, minute = event_time
    event_datetime = datetime(
        event_date.year,
        event_date.month,
        event_date.day,
        hour,
        minute
    )

    return event_datetime, None


# =========================================================
# EVENT TITLE EXTRACTION
# =========================================================

def extract_event_title(text):
    cleaned = text.strip()

    patterns = [
        r"^(?:add|create|schedule|set|put|remember)\s+(?:an?\s+)?(?:event|meeting|appointment|reminder|task)?\s*(?:called|named)\s+(.+?)(?=\s+(?:today|tomorrow|on\s+|at\s+\d)|$)",
        r"^(?:add|create|schedule|set|put|remember)\s+(.+?)(?=\s+(?:today|tomorrow|on\s+|for\s+|at\s+\d)|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            title = match.group(1).strip(" .,!?:")
            title = re.sub(r"^(?:an?\s+)?(?:event|meeting|appointment|reminder|task)\s+", "", title, flags=re.IGNORECASE)
            if title:
                return title

    return None


# =========================================================
# PUBLIC CALENDAR FUNCTIONS
# =========================================================

def add_event(title, event_datetime, description=""):
    events = _load_events()

    event = {
        "id": str(uuid.uuid4())[:8],
        "title": title.strip(),
        "datetime": event_datetime.isoformat(timespec="minutes"),
        "description": description.strip(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    events.append(event)
    events.sort(key=lambda item: item.get("datetime", ""))
    _save_events(events)
    return event


def get_events(start=None, end=None):
    events = _load_events()
    result = []

    for event in events:
        try:
            event_dt = datetime.fromisoformat(event["datetime"])
        except Exception:
            continue

        if start and event_dt < start:
            continue
        if end and event_dt >= end:
            continue

        result.append(event)

    result.sort(key=lambda item: item["datetime"])
    return result


def get_today_events():
    now = datetime.now()
    start = datetime(now.year, now.month, now.day)
    end = start + timedelta(days=1)
    return get_events(start, end)


def get_upcoming_events(limit=10):
    events = get_events(start=datetime.now())
    return events[:max(1, min(int(limit), 50))]


def delete_event(event_id):
    events = _load_events()
    remaining = [event for event in events if event.get("id") != event_id]

    if len(remaining) == len(events):
        return False

    _save_events(remaining)
    return True


def clear_events():
    _save_events([])


def format_event(event):
    try:
        dt = datetime.fromisoformat(event["datetime"])
        formatted = dt.strftime("%A, %d %B %Y at %I:%M %p")
    except Exception:
        formatted = event.get("datetime", "unknown time")

    return f"{event.get('title', 'Untitled')} — {formatted} [ID: {event.get('id', '?')}]"


def format_events(events):
    if not events:
        return "You have no calendar events for that period."

    return "\n".join(
        f"{index}. {format_event(event)}"
        for index, event in enumerate(events, start=1)
    )


# =========================================================
# NATURAL LANGUAGE CALENDAR COMMANDS
# =========================================================

def is_calendar_question(text):
    text = text.lower().strip()

    patterns = [

        # Date questions
        r"\bwhat date is today\b",
        r"\bwhat is today's date\b",
        r"\bwhat's today's date\b",
        r"\bwhat day is today\b",
        r"\bwhat date are we\b",
        r"\bwhat day are we\b",
        r"\bwhat is the date\b",
        r"\btoday's date\b",

        # Explicit scheduling commands
        r"\b(add|create|schedule|set|put|save|book)\b"
        r".*\b"
        r"(tomorrow|today|tonight|"
        r"at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?|"
        r"@\s*\d{1,2})\b",

        # Natural scheduling statements
        r"\b(i have|i've got|i need|i want|"
        r"i'm going to|i am going to)\b"
        r".*\b"
        r"(tomorrow|today|tonight)\b"
        r".*\b"
        r"(at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?|"
        r"@\s*\d{1,2})\b",

        # Common event/activity words
        r"\b(meeting|event|appointment|class|"
        r"reminder|call|task|gym|workout|"
        r"doctor|study|exam|interview|"
        r"college|office|project|shopping)\b"
        r".*\b"
        r"(tomorrow|today|tonight)\b",

        # Calendar lookup
        r"\bwhat do i have\b",
        r"\bwhat have i got\b",
        r"\bwhat's on my calendar\b",
        r"\bwhat is on my calendar\b",
        r"\bshow my calendar\b",
        r"\bshow my events\b",
        r"\bshow upcoming events\b",
        r"\bshow my upcoming events\b",
        r"\bupcoming events\b",
        r"\bmy upcoming events\b",
        r"\bwhat are my events\b",
        r"\bmy schedule\b",
        r"\bwhat is my schedule\b",
        r"\bwhat's my schedule\b",
        r"\bwhat do i have tomorrow\b",
        r"\bwhat do i have today\b",
        r"\bwhat's happening tomorrow\b",
        r"\bwhat is happening tomorrow\b",
        r"\bwhat's happening today\b",
        r"\bwhat is happening today\b",
        r"\bwhat events do i have\b",
    ]

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


def calendar_response(text):
    lowered = text.lower().strip()

    # Delete by ID
    delete_match = re.search(r"\b(?:delete|remove)\b.*?\b(?:event|meeting|appointment|reminder)\s*(?:id\s*)?([a-f0-9]{8})\b", lowered)
    if delete_match:
        event_id = delete_match.group(1)
        if delete_event(event_id):
            return f"Calendar event {event_id} deleted successfully."
        return f"I couldn't find calendar event {event_id}."

    # Clear all events
    if re.search(r"\b(?:clear|delete|remove)\s+(?:all|everything)\s+(?:calendar\s+)?(?:events?|meetings?|appointments?|reminders?)\b", lowered):
        clear_events()
        return "All local calendar events have been deleted."

    # Show today's events
    if re.search(r"\b(?:today|today's)\b", lowered) and re.search(r"\b(?:calendar|events?|meetings?|appointments?)\b", lowered):
        events = get_today_events()
        return format_events(events)

    # Show tomorrow's events
    if re.search(r"\btomorrow\b", lowered) and re.search(r"\b(?:calendar|events?|meetings?|appointments?)\b", lowered):
        now = datetime.now() + timedelta(days=1)
        start = datetime(now.year, now.month, now.day)
        events = get_events(start, start + timedelta(days=1))
        return format_events(events)

    # Generic upcoming calendar request
    if re.search(r"\b(?:show|list|see|view|what(?:'s| is))\b", lowered) and re.search(r"\b(?:calendar|events?|meetings?|appointments?)\b", lowered):
        return format_events(get_upcoming_events(10))

    # Add event
    if re.search(r"\b(?:add|create|schedule|set|put|remember)\b", lowered):
        event_datetime, error = parse_event_datetime(text)
        if error:
            return error

        title = extract_event_title(text)
        if not title:
            return "What should I call the event? For example: add project meeting tomorrow at 10 AM."

        event = add_event(title, event_datetime)
        return "Event added successfully: " + format_event(event)

    return "I can manage your local calendar. You can add, view, or delete events."
