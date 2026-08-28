"""
Multi-Session Dialogue Ingestion & Semantic Memory Indexing Module.
Ingests 5 distinct multi-turn developer sessions across a 14-day timeline:
- Session A (Architecture): Zero-Copy ring buffer for IPC with 64-byte cache-line alignment.
- Session B (Code Rules): Strict functional programming with pure functions and zero dependencies.
- Session C (Database): PostgreSQL partitioned temporal tables with composite PK (tenant_id, event_time).
- Session D (Security): Custom ED25519-signed ephemeral token exchange headers with 30-second TTLs.
- Session E (Infrastructure): Max 4.0s container execution timeout and 512MB hard virtual memory cap.
"""

import json
import math
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

from config.settings import Settings, get_settings
from memory.db import EpisodicMemoryDB


HISTORICAL_DIALOGUE_SESSIONS: List[Dict[str, Any]] = [
    {
        "session_id": "session_a_ipc",
        "day_offset": 14,
        "topic": "Inter-Process Communication & Ring Buffer Architecture",
        "turns": [
            {
                "speaker": "lead_architect",
                "message": "We need ultra-low latency IPC for high-throughput tensor serialization between worker nodes. Socket overhead is too high."
            },
            {
                "speaker": "smart_ai",
                "message": "I recommend implementing a lock-free Zero-Copy ring buffer in shared memory. By enforcing strict 64-byte cache-line alignment on packet boundaries, we eliminate false sharing and achieve sub-microsecond latency."
            },
            {
                "speaker": "lead_architect",
                "message": "Agreed. Let us formalize the decision: Zero-Copy ring buffer with 64-byte cache-line alignment for all IPC."
            }
        ],
        "tags": ["IPC", "Zero-Copy", "ring buffer", "64-byte", "cache-line", "architecture"],
        "key_facts": [
            "IPC architecture: Zero-Copy ring buffer",
            "Memory alignment constraint: 64-byte cache-line alignment"
        ]
    },
    {
        "session_id": "session_b_rules",
        "day_offset": 11,
        "topic": "Core Engine Code Quality & Functional Rules",
        "turns": [
            {
                "speaker": "tech_lead",
                "message": "Our core mathematical reasoning modules are experiencing side-effect bugs during parallel search rollouts."
            },
            {
                "speaker": "smart_ai",
                "message": "Let us enforce strict functional programming constraints on all core search components: all functions must be pure, immutable inputs, with zero external dependencies."
            },
            {
                "speaker": "tech_lead",
                "message": "Adopted as mandatory rule: pure functions, strict immutability, zero external dependencies for core engine modules."
            }
        ],
        "tags": ["rules", "functional programming", "pure functions", "zero dependencies", "immutability"],
        "key_facts": [
            "Code paradigm: Strict functional programming with pure functions",
            "Dependency constraint: Zero external dependencies for core reasoning engine"
        ]
    },
    {
        "session_id": "session_c_db",
        "day_offset": 8,
        "topic": "High-Volume Temporal Database Partitioning Schema",
        "turns": [
            {
                "speaker": "database_engineer",
                "message": "We are designing the analytics database schema for multi-tenant streaming events. Query performance degrades after 100M rows."
            },
            {
                "speaker": "smart_ai",
                "message": "Use PostgreSQL partitioned temporal tables with composite primary key (tenant_id, event_time). Monthly partitions will isolate hot writes and keep b-tree index sizes under L3 cache limits."
            },
            {
                "speaker": "database_engineer",
                "message": "Excellent decision. We will deploy PostgreSQL temporal range partitioning with composite PK (tenant_id, event_time)."
            }
        ],
        "tags": ["PostgreSQL", "partitioning", "temporal tables", "composite PK", "tenant_id", "event_time"],
        "key_facts": [
            "Database engine & design: PostgreSQL partitioned temporal tables",
            "Composite primary key: (tenant_id, event_time)"
        ]
    },
    {
        "session_id": "session_d_security",
        "day_offset": 4,
        "topic": "Cryptographic Token Exchange & Session Security Protocol",
        "turns": [
            {
                "speaker": "security_officer",
                "message": "We need to secure stateless RPC calls across edge nodes without querying a centralized Redis cluster on every request."
            },
            {
                "speaker": "smart_ai",
                "message": "Deploy custom ED25519-signed ephemeral token exchange headers with a strict 30-second TTL. Edge workers verify the public key signature in sub-millisecond time locally."
            },
            {
                "speaker": "security_officer",
                "message": "Confirmed policy: ED25519 cryptographic signatures with 30-second TTL for all ephemeral authorization tokens."
            }
        ],
        "tags": ["security", "ED25519", "ephemeral tokens", "30-second TTL", "cryptography"],
        "key_facts": [
            "Signature algorithm: ED25519 asymmetric cryptography",
            "Token TTL: 30-second TTL"
        ]
    },
    {
        "session_id": "session_e_infra",
        "day_offset": 1,
        "topic": "Worker Isolation & Sandbox Memory / Timeout Constraints",
        "turns": [
            {
                "speaker": "devops_engineer",
                "message": "What resource bounds should be configured on the Python RLVR execution sandbox containers?"
            },
            {
                "speaker": "smart_ai",
                "message": "Enforce a hard timeout of 4.0 seconds per container and a strict virtual memory cap of 512MB per sandboxed worker. This protects host nodes from runaway recursion or memory leaks."
            },
            {
                "speaker": "devops_engineer",
                "message": "Configured in container runtime: max 4.0s timeout and 512MB hard limit per sandboxed worker."
            }
        ],
        "tags": ["infrastructure", "sandbox", "4.0s timeout", "512MB limit", "isolation"],
        "key_facts": [
            "Execution timeout limit: Max 4.0s execution timeout per container",
            "Virtual memory cap: 512MB hard virtual memory limit per sandboxed worker"
        ]
    }
]


def ingest_historical_dialogues(db_path: str = "data/memory.db") -> Dict[str, Any]:
    """Ingests the 5 historical dialogue sessions into SQLite memory.db with semantic tags and metadata."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dialogue_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            topic TEXT NOT NULL,
            day_offset INTEGER NOT NULL,
            turns_json TEXT NOT NULL,
            tags_csv TEXT NOT NULL,
            key_facts_json TEXT NOT NULL,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS semantic_memory_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            fact_text TEXT NOT NULL,
            tag_keywords TEXT NOT NULL,
            importance_score REAL DEFAULT 1.0,
            FOREIGN KEY (session_id) REFERENCES dialogue_sessions(session_id)
        )
    """)

    sessions_ingested = 0
    facts_indexed = 0

    for sess in HISTORICAL_DIALOGUE_SESSIONS:
        cursor.execute("""
            INSERT OR REPLACE INTO dialogue_sessions 
            (session_id, topic, day_offset, turns_json, tags_csv, key_facts_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            sess["session_id"],
            sess["topic"],
            sess["day_offset"],
            json.dumps(sess["turns"]),
            ",".join(sess["tags"]),
            json.dumps(sess["key_facts"])
        ))
        sessions_ingested += 1

        for fact in sess["key_facts"]:
            cursor.execute("""
                INSERT INTO semantic_memory_index (session_id, fact_text, tag_keywords, importance_score)
                VALUES (?, ?, ?, 1.0)
            """, (sess["session_id"], fact, ",".join(sess["tags"])))
            facts_indexed += 1

    conn.commit()
    conn.close()

    return {
        "status": "success",
        "sessions_ingested": sessions_ingested,
        "facts_indexed": facts_indexed,
        "timeline_span_days": 14
    }


def recall_historical_fact(query: str, db_path: str = "data/memory.db") -> Tuple[bool, str, Dict[str, Any]]:
    """
    Searches episodic and semantic database for historical dialogue facts.
    Returns (found, synthesized_answer, metadata).
    """
    if not os.path.exists(db_path):
        return False, "Database does not exist", {}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query semantic memory index
    try:
        cursor.execute("SELECT session_id, fact_text, tag_keywords FROM semantic_memory_index")
        rows = cursor.fetchall()
    except Exception:
        conn.close()
        return False, "No dialogue memory table found", {}

    q_lower = query.lower()
    best_fact = None
    best_sess = None
    best_score = 0.0

    for sess_id, fact_text, tags in rows:
        score = 0
        words = q_lower.split()
        for w in words:
            if len(w) > 3 and w in fact_text.lower():
                score += 2
            if len(w) > 3 and w in tags.lower():
                score += 1
        if score > best_score:
            best_score = score
            best_fact = fact_text
            best_sess = sess_id

    conn.close()

    if best_fact and best_score >= 2:
        return True, best_fact, {
            "session_id": best_sess,
            "match_score": best_score,
            "fact": best_fact
        }

    return False, "No historical decision matched in episodic memory", {}


if __name__ == "__main__":
    res = ingest_historical_dialogues()
    print("Ingested Historical Dialogues:", res)
    ok, ans, meta = recall_historical_fact("What IPC ring buffer alignment was chosen in Session A?")
    print(f"Recall Check: Found={ok} | Answer='{ans}'")
