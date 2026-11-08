"""Dashboard insights: what customers asked and how the assistant answered."""

import logging

from sqlalchemy import text

from app.database.connection import engine


logger = logging.getLogger(__name__)


ANSWERED = {"KNOWN", "CASUAL", "CONFIG_ONLY"}
UNANSWERED = {"RELATED_BUT_UNKNOWN", "KNOWLEDGE_GAP", "OUT_OF_SCOPE", "NOT_ANSWERABLE"}

# Gap types worth showing as "questions the assistant could not answer".
GAP_TYPES = ("RELATED_BUT_UNKNOWN", "KNOWLEDGE_GAP", "OUT_OF_SCOPE")


def outcome_for(action: str, status: str) -> str:
    if action == "handoff":
        return "handoff"
    if status == "CLARIFICATION_REQUIRED":
        return "clarification"
    if status in ANSWERED:
        return "answered"
    if status in UNANSWERED:
        return "unanswered"
    return "error"


def record_chat(
    client_id: int,
    request_id: str | None,
    conversation_id: str,
    message: str,
    answer: str,
    action: str,
    status: str,
    latency_ms: int,
    intent: str | None = None,
    intent_reason: str | None = None,
    best_distance: float | None = None,
):
    # Reporting only: never fail the customer's answer over it.
    try:
        with engine.begin() as connection:
            connection.execute(
                text("""
                    INSERT INTO chat_log (
                        client_id, request_id, conversation_id, message, answer,
                        action, status, outcome, latency_ms,
                        intent, intent_reason, best_distance
                    )
                    VALUES (
                        :client_id, :request_id, :conversation_id, :message, :answer,
                        :action, :status, :outcome, :latency_ms,
                        :intent, :intent_reason, :best_distance
                    )
                """),
                {
                    "client_id": client_id,
                    "conversation_id": conversation_id or None,
                    "message": message,
                    "answer": answer or "",
                    "action": action,
                    "status": status,
                    "outcome": outcome_for(action, status),
                    "latency_ms": latency_ms,
                    "request_id": request_id,
                    "intent": intent,
                    "intent_reason": intent_reason,
                    "best_distance": best_distance,
                },
            )
    except Exception:
        logger.exception("Failed to record chat: client_id=%s", client_id)


def _scope(client_id: int | None, alias: str) -> str:
    return f"AND {alias}.client_id = :client_id" if client_id is not None else ""


def get_insights(client_id: int | None, days: int) -> dict:
    """Activity for the last `days` days; client_id None means all providers."""
    params = {"client_id": client_id, "days": days}

    with engine.connect() as connection:
        daily = connection.execute(
            text(f"""
                WITH d AS (
                    SELECT generate_series(
                        current_date - (:days - 1), current_date, interval '1 day'
                    )::date AS day
                ),
                chats AS (
                    SELECT l.created_at::date AS day,
                           count(*) AS messages,
                           count(*) FILTER (WHERE outcome = 'answered') AS answered,
                           count(*) FILTER (WHERE outcome = 'handoff') AS handoff,
                           count(*) FILTER (WHERE outcome = 'unanswered') AS unanswered,
                           count(*) FILTER (WHERE outcome = 'clarification') AS clarification,
                           count(*) FILTER (WHERE outcome = 'error') AS error
                    FROM chat_log l
                    WHERE l.created_at >= current_date - (:days - 1) {_scope(client_id, "l")}
                    GROUP BY 1
                ),
                usage AS (
                    SELECT u.created_at::date AS day,
                           count(*) AS calls,
                           sum(input_tokens) AS input_tokens,
                           sum(output_tokens) AS output_tokens,
                           sum(cost_usd) AS cost_usd
                    FROM claude_usage u
                    WHERE u.created_at >= current_date - (:days - 1) {_scope(client_id, "u")}
                    GROUP BY 1
                )
                SELECT d.day,
                       COALESCE(c.messages, 0) AS messages,
                       COALESCE(c.answered, 0) AS answered,
                       COALESCE(c.handoff, 0) AS handoff,
                       COALESCE(c.unanswered, 0) AS unanswered,
                       COALESCE(c.clarification, 0) AS clarification,
                       COALESCE(c.error, 0) AS error,
                       COALESCE(u.calls, 0) AS calls,
                       COALESCE(u.input_tokens, 0) AS input_tokens,
                       COALESCE(u.output_tokens, 0) AS output_tokens,
                       COALESCE(u.cost_usd, 0) AS cost_usd
                FROM d
                LEFT JOIN chats c ON c.day = d.day
                LEFT JOIN usage u ON u.day = d.day
                ORDER BY d.day
            """),
            params,
        ).fetchall()

        period = connection.execute(
            text(f"""
                SELECT count(DISTINCT conversation_id) AS conversations,
                       avg(latency_ms) AS avg_latency_ms,
                       percentile_cont(0.9) WITHIN GROUP (ORDER BY latency_ms) AS p90_latency_ms
                FROM chat_log l
                WHERE l.created_at >= current_date - (:days - 1) {_scope(client_id, "l")}
            """),
            params,
        ).fetchone()

        tracking_since = connection.execute(
            text(f"SELECT min(created_at) FROM chat_log l WHERE true {_scope(client_id, 'l')}"),
            params,
        ).scalar()

        gaps = _gaps(connection, client_id, days)

        gap_counts = connection.execute(
            text(f"""
                SELECT count(*) AS asked,
                       count(DISTINCT lower(btrim(message))) AS distinct_questions
                FROM ai_gaps g
                WHERE g.created_at >= current_date - (:days - 1)
                  AND g.type = ANY(:gap_types)
                  {_scope(client_id, "g")}
            """),
            params | {"gap_types": list(GAP_TYPES)},
        ).fetchone()

        by_purpose = connection.execute(
            text(f"""
                SELECT COALESCE(u.purpose, 'untracked') AS purpose,
                       count(*) AS calls,
                       sum(input_tokens) AS input_tokens,
                       sum(output_tokens) AS output_tokens,
                       sum(cost_usd) AS cost_usd
                FROM claude_usage u
                WHERE u.created_at >= current_date - (:days - 1) {_scope(client_id, "u")}
                GROUP BY 1
                ORDER BY sum(cost_usd) DESC
            """),
            params,
        ).fetchall()

        # Cost of the messages whose Claude calls are linked to them.
        linked = connection.execute(
            text(f"""
                SELECT count(DISTINCT l.id) AS messages, sum(u.cost_usd) AS cost_usd
                FROM chat_log l
                JOIN claude_usage u ON u.request_id = l.request_id
                WHERE l.created_at >= current_date - (:days - 1) {_scope(client_id, "l")}
            """),
            params,
        ).fetchone()

        recent = connection.execute(
            text(f"""
                SELECT l.id, l.conversation_id, l.message, l.answer, l.outcome,
                       l.status, l.latency_ms, l.created_at, c.name AS provider,
                       l.intent, l.intent_reason, l.best_distance,
                       u.calls, u.input_tokens, u.output_tokens, u.cost_usd
                FROM chat_log l
                JOIN clients c ON c.id = l.client_id
                LEFT JOIN LATERAL (
                    SELECT count(*) AS calls,
                           sum(input_tokens) AS input_tokens,
                           sum(output_tokens) AS output_tokens,
                           sum(cost_usd) AS cost_usd
                    FROM claude_usage cu
                    WHERE cu.request_id = l.request_id
                ) u ON l.request_id IS NOT NULL
                WHERE true {_scope(client_id, "l")}
                ORDER BY l.created_at DESC
                LIMIT 30
            """),
            params,
        ).fetchall()

    series = []
    totals = {
        "messages": 0, "answered": 0, "handoff": 0, "unanswered": 0,
        "clarification": 0, "error": 0, "calls": 0,
        "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
    }

    for row in daily:
        cost = float(row.cost_usd)
        point = {
            "date": row.day.isoformat(),
            "messages": row.messages,
            "answered": row.answered,
            "handoff": row.handoff,
            "unanswered": row.unanswered,
            "clarification": row.clarification,
            "error": row.error,
            "calls": row.calls,
            "cost_usd": round(cost, 4),
        }
        series.append(point)
        for key in totals:
            totals[key] += cost if key == "cost_usd" else getattr(row, key)

    totals["cost_usd"] = round(totals["cost_usd"], 4)
    totals["conversations"] = period.conversations or 0
    totals["cost_per_message_usd"] = (
        round(float(linked.cost_usd) / linked.messages, 6) if linked.messages else None
    )
    totals["gaps_asked"] = gap_counts.asked
    totals["gaps_distinct"] = gap_counts.distinct_questions
    totals["avg_latency_ms"] = round(period.avg_latency_ms) if period.avg_latency_ms is not None else None
    totals["p90_latency_ms"] = round(period.p90_latency_ms) if period.p90_latency_ms is not None else None

    return {
        "days": days,
        "tracking_since": tracking_since.isoformat() if tracking_since else None,
        "totals": totals,
        "daily": series,
        "gaps": gaps,
        "cost_by_purpose": [
            {
                "purpose": r.purpose,
                "calls": r.calls,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cost_usd": round(float(r.cost_usd or 0), 6),
            }
            for r in by_purpose
        ],
        "recent": [
            {
                "id": r.id,
                "conversation_id": r.conversation_id,
                "message": r.message,
                "answer": r.answer,
                "outcome": r.outcome,
                "status": r.status,
                "latency_ms": r.latency_ms,
                "created_at": r.created_at.isoformat(),
                "provider": r.provider,
                "intent": r.intent,
                "intent_reason": r.intent_reason,
                "best_distance": r.best_distance,
                # Only for messages recorded after usage was linked to requests.
                "usage": {
                    "calls": r.calls,
                    "input_tokens": r.input_tokens or 0,
                    "output_tokens": r.output_tokens or 0,
                    "cost_usd": round(float(r.cost_usd or 0), 6),
                } if r.calls else None,
            }
            for r in recent
        ],
    }


def _gaps(connection, client_id: int | None, days: int) -> list[dict]:
    # The same question asked several times shows once, with a count.
    rows = connection.execute(
        text(f"""
            SELECT min(g.message) AS message,
                   (array_agg(g.type ORDER BY g.created_at DESC))[1] AS type,
                   count(*) AS times,
                   max(g.created_at) AS last_asked,
                   (array_agg(c.name ORDER BY g.created_at DESC))[1] AS provider
            FROM ai_gaps g
            JOIN clients c ON c.id = g.client_id
            WHERE g.created_at >= current_date - (:days - 1)
              AND g.type = ANY(:gap_types)
              {_scope(client_id, "g")}
            GROUP BY lower(btrim(g.message))
            ORDER BY count(*) DESC, max(g.created_at) DESC
            LIMIT 25
        """),
        {"client_id": client_id, "days": days, "gap_types": list(GAP_TYPES)},
    ).fetchall()

    return [
        {
            "message": r.message,
            "type": r.type,
            "times": r.times,
            "last_asked": r.last_asked.isoformat() if r.last_asked else None,
            "provider": r.provider,
        }
        for r in rows
    ]


# ----------------------------------------------------------------------
# Knowledge entries
# ----------------------------------------------------------------------

def list_knowledge(client_id: int, query: str, limit: int, offset: int) -> dict:
    where = "client_id = :client_id"
    params = {"client_id": client_id, "limit": limit, "offset": offset}

    if query:
        where += " AND content ILIKE :q"
        params["q"] = f"%{query}%"

    with engine.connect() as connection:
        total = connection.execute(
            text(f"SELECT count(*) FROM knowledge WHERE {where}"), params
        ).scalar()
        rows = connection.execute(
            text(f"""
                SELECT id, content FROM knowledge
                WHERE {where}
                ORDER BY id DESC
                LIMIT :limit OFFSET :offset
            """),
            params,
        ).fetchall()

    return {"total": total, "items": [{"id": r.id, "content": r.content} for r in rows]}


def delete_knowledge(client_id: int, entry_id: int) -> bool:
    with engine.begin() as connection:
        result = connection.execute(
            text("DELETE FROM knowledge WHERE id = :id AND client_id = :client_id"),
            {"id": entry_id, "client_id": client_id},
        )
    return result.rowcount > 0
