import sqlite3
from pathlib import Path
from datetime import datetime
import re


# =========================================================
# DATABASE CONFIGURATION
# =========================================================

MEMORY_DIR = Path(__file__).resolve().parent
DATABASE_FILE = MEMORY_DIR / "memory.db"


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_connection():
    """
    Create and return a connection to the SQLite database.
    """

    connection = sqlite3.connect(DATABASE_FILE)
    connection.row_factory = sqlite3.Row

    return connection


# =========================================================
# INITIALIZE DATABASE
# =========================================================

def initialize_database():

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                created_at TEXT NOT NULL
            )
            """
        )

        connection.commit()

    finally:

        connection.close()


# =========================================================
# SAVE MEMORY
# =========================================================

def save_memory(memory, category="general"):

    memory = str(memory).strip()
    category = str(category).strip() or "general"

    if not memory:

        raise ValueError(
            "Memory cannot be empty."
        )

    connection = get_connection()

    try:

        cursor = connection.cursor()

        created_at = datetime.now().isoformat(
            timespec="seconds"
        )

        cursor.execute(
            """
            INSERT INTO memories
            (memory, category, created_at)
            VALUES (?, ?, ?)
            """,
            (
                memory,
                category,
                created_at
            )
        )

        connection.commit()

        return cursor.lastrowid

    finally:

        connection.close()


# =========================================================
# SAVE OR UPDATE MEMORY
# =========================================================

def save_or_update_memory(
    memory,
    category="general"
):

    memory = str(memory).strip()
    category = str(category).strip() or "general"

    if not memory:

        raise ValueError(
            "Memory cannot be empty."
        )

    connection = get_connection()

    try:

        cursor = connection.cursor()

        # -------------------------------------------------
        # EXACT MATCH
        # -------------------------------------------------

        cursor.execute(
            """
            SELECT id
            FROM memories
            WHERE LOWER(memory) = LOWER(?)
              AND LOWER(category) = LOWER(?)
            LIMIT 1
            """,
            (
                memory,
                category
            )
        )

        existing = cursor.fetchone()

        if existing:

            cursor.execute(
                """
                UPDATE memories
                SET memory = ?
                WHERE id = ?
                """,
                (
                    memory,
                    existing["id"]
                )
            )

            connection.commit()

            return {
                "action": "updated",
                "id": existing["id"]
            }

        # -------------------------------------------------
        # INSERT
        # -------------------------------------------------

        created_at = datetime.now().isoformat(
            timespec="seconds"
        )

        cursor.execute(
            """
            INSERT INTO memories
            (memory, category, created_at)
            VALUES (?, ?, ?)
            """,
            (
                memory,
                category,
                created_at
            )
        )

        connection.commit()

        return {
            "action": "created",
            "id": cursor.lastrowid
        }

    finally:

        connection.close()


# =========================================================
# GET ALL MEMORIES
# =========================================================

def get_all_memories():

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                memory,
                category,
                created_at
            FROM memories
            ORDER BY id ASC
            """
        )

        rows = cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        connection.close()


# =========================================================
# GET RECENT MEMORIES
# =========================================================

def get_recent_memories(limit=10):

    limit = max(
        1,
        int(limit)
    )

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                memory,
                category,
                created_at
            FROM memories
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        )

        rows = cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        connection.close()


# =========================================================
# SEARCH MEMORIES
# =========================================================

def search_memories(
    query,
    limit=10
):

    query = str(query).strip()

    if not query:

        return []

    limit = max(
        1,
        int(limit)
    )

    connection = get_connection()

    try:

        cursor = connection.cursor()

        search_pattern = f"%{query}%"

        cursor.execute(
            """
            SELECT
                id,
                memory,
                category,
                created_at
            FROM memories
            WHERE memory LIKE ?
               OR category LIKE ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (
                search_pattern,
                search_pattern,
                limit
            )
        )

        rows = cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        connection.close()


# =========================================================
# SEARCH MEMORIES BY KEYWORDS
# =========================================================

def search_memories_by_keywords(
    query,
    limit=10
):

    query = str(query).strip()

    if not query:

        return []

    limit = max(
        1,
        int(limit)
    )

    # -----------------------------------------------------
    # STOP WORDS
    # -----------------------------------------------------

    stop_words = {

        "what",
        "what's",
        "is",
        "are",
        "the",
        "my",
        "me",
        "i",
        "do",
        "you",
        "know",
        "tell",
        "about",
        "please",
        "can",
        "could",
        "would",
        "your",
        "a",
        "an",
        "of",
        "to",
        "for",
        "in",
        "on",
        "and",
        "or",
        "it",
        "that",
        "this",
        "who",
        "when",
        "where",
        "why",
        "how"
    }

    words = re.findall(
        r"[A-Za-z0-9+#.]+",
        query.lower()
    )

    keywords = [

        word

        for word in words

        if word not in stop_words
        and len(word) > 1

    ]

    if not keywords:

        return []

    connection = get_connection()

    try:

        cursor = connection.cursor()

        conditions = []
        parameters = []

        for keyword in keywords:

            conditions.append(
                """
                (
                    LOWER(memory) LIKE ?
                    OR LOWER(category) LIKE ?
                )
                """
            )

            pattern = f"%{keyword}%"

            parameters.append(pattern)
            parameters.append(pattern)

        sql = f"""
            SELECT
                id,
                memory,
                category,
                created_at
            FROM memories
            WHERE {" OR ".join(conditions)}
            ORDER BY id DESC
            LIMIT ?
        """

        parameters.append(limit)

        cursor.execute(
            sql,
            parameters
        )

        rows = cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        connection.close()


# =========================================================
# GET RELEVANT MEMORIES
# =========================================================

def get_relevant_memories(
    query,
    limit=5
):

    query = str(query).strip()

    if not query:

        return []

    exact_results = search_memories(
        query,
        limit
    )

    if exact_results:

        return exact_results

    return search_memories_by_keywords(
        query,
        limit
    )


# =========================================================
# GET MEMORY BY ID
# =========================================================

def get_memory(memory_id):

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                memory,
                category,
                created_at
            FROM memories
            WHERE id = ?
            """,
            (memory_id,)
        )

        row = cursor.fetchone()

        if row is None:

            return None

        return dict(row)

    finally:

        connection.close()


# =========================================================
# DELETE MEMORY
# =========================================================

def delete_memory(memory_id):

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            DELETE FROM memories
            WHERE id = ?
            """,
            (memory_id,)
        )

        connection.commit()

        return cursor.rowcount > 0

    finally:

        connection.close()


# =========================================================
# DELETE ALL MEMORIES
# =========================================================

def delete_all_memories():

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            DELETE FROM memories
            """
        )

        connection.commit()

    finally:

        connection.close()


# =========================================================
# MEMORY COUNT
# =========================================================

def get_memory_count():

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM memories
            """
        )

        result = cursor.fetchone()

        return int(result[0])

    finally:

        connection.close()


# =========================================================
# MEMORY DETECTION
# =========================================================

def detect_memory(text):

    text = str(text).strip()

    if not text:

        return None

    lower = text.lower()

    # =====================================================
    # IGNORE QUESTIONS / COMMANDS
    # =====================================================

    ignored_patterns = [

        r"^what is ",
        r"^what's ",
        r"^who is ",
        r"^who are ",
        r"^where is ",
        r"^where are ",
        r"^when is ",
        r"^when are ",
        r"^why is ",
        r"^why are ",
        r"^how is ",
        r"^how are ",
        r"^how do ",
        r"^how can ",
        r"^can you ",
        r"^could you ",
        r"^would you ",
        r"^tell me ",
        r"^calculate ",
        r"^solve ",
        r"^compute ",
        r"^open ",
        r"^close ",
        r"^launch ",
        r"^start ",
        r"^stop ",
        r"^run "
    ]

    for pattern in ignored_patterns:

        if re.search(
            pattern,
            lower
        ):

            return None

    # =====================================================
    # NAME
    # =====================================================

    name_patterns = [

        r"\bmy name is ([A-Za-z][A-Za-z ]{1,40})\b",

        r"\bcall me ([A-Za-z][A-Za-z ]{1,40})\b",

        r"\bi am ([A-Za-z][A-Za-z ]{1,40})\b"

    ]

    for pattern in name_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            name = match.group(1).strip()

            if len(name.split()) <= 4:

                return {

                    "memory":
                        f"The user's name is {name}.",

                    "category":
                        "identity"

                }

    # =====================================================
    # FAVORITES / PREFERENCES
    # =====================================================

    preference_patterns = [

        (
            r"\bmy favorite ([A-Za-z ]+?) is (.+)",
            "preference"
        ),

        (
            r"\bi prefer (.+)",
            "preference"
        ),

        (
            r"\bi like (.+)",
            "preference"
        ),

        (
            r"\bi love (.+)",
            "preference"
        ),

        (
            r"\bi enjoy (.+)",
            "preference"
        )

    ]

    for pattern, category in preference_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            return {

                "memory":
                    text,

                "category":
                    category

            }

    # =====================================================
    # DISLIKES
    # =====================================================

    dislike_patterns = [

        r"\bi don't like (.+)",

        r"\bi dislike (.+)",

        r"\bi hate (.+)"

    ]

    for pattern in dislike_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            return {

                "memory":
                    text,

                "category":
                    "dislike"

            }

    # =====================================================
    # PERSONAL FACTS
    # =====================================================

    personal_patterns = [

        r"\bi am a (.+)",

        r"\bi'm a (.+)",

        r"\bi work as (.+)",

        r"\bi study (.+)",

        r"\bi am studying (.+)",

        r"\bi'm studying (.+)",

        r"\bi live in (.+)",

        r"\bi'm from (.+)",

        r"\bi am from (.+)"

    ]

    for pattern in personal_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            return {

                "memory":
                    text,

                "category":
                    "personal"

            }

    # =====================================================
    # EXPLICIT MEMORY
    # =====================================================

    remember_patterns = [

        r"\bremember that (.+)",

        r"\bremember this (.+)",

        r"\bplease remember (.+)",

        r"\bkeep in mind that (.+)"

    ]

    for pattern in remember_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            memory_text = match.group(1).strip()

            return {

                "memory":
                    memory_text,

                "category":
                    "important"

            }

    return None


# =========================================================
# AUTOMATIC MEMORY PROCESSING
# =========================================================

def process_memory(text):

    detected = detect_memory(text)

    if detected is None:

        return None

    result = save_or_update_memory(

        detected["memory"],

        detected["category"]

    )

    return {

        "saved":
            True,

        "action":
            result["action"],

        "id":
            result["id"],

        "memory":
            detected["memory"],

        "category":
            detected["category"]

    }


# =========================================================
# BUILD MEMORY CONTEXT
# =========================================================

def build_memory_context(memories):
    """
    Safely convert memory records into AI context.

    Accepts both dictionaries and strings.
    """

    if not memories:

        return ""

    lines = []

    for item in memories:

        # -------------------------------------------------
        # DICTIONARY MEMORY
        # -------------------------------------------------

        if isinstance(
            item,
            dict
        ):

            memory = item.get(
                "memory",
                ""
            )

            category = item.get(
                "category",
                "general"
            )

        # -------------------------------------------------
        # SQLITE ROW
        # -------------------------------------------------

        elif isinstance(
            item,
            sqlite3.Row
        ):

            memory = item["memory"]

            category = item["category"]

        # -------------------------------------------------
        # STRING MEMORY
        # -------------------------------------------------

        elif isinstance(
            item,
            str
        ):

            memory = item

            category = "general"

        # -------------------------------------------------
        # UNKNOWN TYPE
        # -------------------------------------------------

        else:

            continue

        memory = str(
            memory
        ).strip()

        category = str(
            category
        ).strip() or "general"

        if memory:

            lines.append(
                f"- {memory} "
                f"(category: {category})"
            )

    if not lines:

        return ""

    return (
        "Useful information remembered "
        "about the user:\n"
        + "\n".join(lines)
    )


# =========================================================
# INITIALIZE DATABASE
# =========================================================

initialize_database()


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "PHASE 13C — SQLITE MEMORY TEST"
    )

    print("=" * 60)

    print()

    print(
        "Database:"
    )

    print(
        DATABASE_FILE
    )

    # -----------------------------------------------------
    # TEST
    # -----------------------------------------------------

    test_messages = [

        "My favorite programming language is C++.",

        "I prefer dark mode.",

        "I like Python.",

        "What is artificial intelligence?",

        "Calculate 25 * 4",

        "Remember that I am building an AI assistant."

    ]

    print()

    print(
        "Testing memory detection..."
    )

    for message in test_messages:

        print()

        print(
            "User:",
            message
        )

        result = process_memory(
            message
        )

        if result:

            print(
                "MEMORY:",
                result["memory"]
            )

            print(
                "CATEGORY:",
                result["category"]
            )

            print(
                "ACTION:",
                result["action"]
            )

        else:

            print(
                "No long-term memory detected."
            )

    # -----------------------------------------------------
    # SHOW ALL
    # -----------------------------------------------------

    print()

    print(
        "All memories:"
    )

    memories = get_all_memories()

    for item in memories:

        print(
            f"[{item['id']}] "
            f"{item['memory']} "
            f"({item['category']})"
        )

    # -----------------------------------------------------
    # SEARCH
    # -----------------------------------------------------

    print()

    print(
        "Searching for programming language..."
    )

    results = get_relevant_memories(

        "What programming language do I prefer?"

    )

    for item in results:

        print(
            f"[{item['id']}] "
            f"{item['memory']}"
        )

    # -----------------------------------------------------
    # CONTEXT
    # -----------------------------------------------------

    print()

    print(
        "Generated AI memory context:"
    )

    context = build_memory_context(
        results
    )

    print(
        context
    )

    # -----------------------------------------------------
    # COUNT
    # -----------------------------------------------------

    print()

    print(
        "Total memories:",
        get_memory_count()
    )

    print()

    print("=" * 60)

    print(
        "MEMORY TEST COMPLETE"
    )

    print("=" * 60)