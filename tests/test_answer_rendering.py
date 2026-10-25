"""What survives of the two answer services: source-card splitting and stored-date rendering.

The router used to answer questions itself, from candidates the model named or from a
snapshot of the whole project. The agent searches and cites instead, so only these two
pieces are still used, and the time policy they carry is unchanged: a date or time shown
to the user is read from SQLite, never written by the model.
"""

import unittest
from datetime import date, datetime, timezone

from captureflow.answer_rendering import answer_source_references, with_stored_dates
from captureflow.models import CalendarEvent, ProjectRecord, SavedItem


def _event(event_id: str, title: str, start_at, end_at=None) -> CalendarEvent:
    return CalendarEvent(
        id=event_id,
        conversation_id="chat",
        title=title,
        source_message_id="source",
        source_quote="source",
        start_at=start_at,
        end_at=end_at,
        all_day=not isinstance(start_at, datetime),
        timezone="America/Vancouver",
    )


def _record(record_id: str, content: str, due_at=None) -> ProjectRecord:
    return ProjectRecord(
        id=record_id,
        project_id="project",
        conversation_id="chat",
        kind="task",
        content=content,
        state="active",
        due_at=due_at,
        source_message_id="source",
        source_quote="source",
    )


class AnswerSourceReferencesTests(unittest.TestCase):
    def test_sources_split_into_record_ids_and_local_references_in_order(self) -> None:
        sources = [
            _event("event-1", "Design review", date(2026, 9, 18)),
            _record("record-1", "Measure latency"),
            SavedItem(
                id="item-1",
                conversation_id="chat",
                kind="note",
                content="Latency notes",
                source_message_id="source",
                source_quote="source",
            ),
        ]

        record_ids, references = answer_source_references(sources)

        self.assertEqual(record_ids, ["record-1"])
        self.assertEqual(
            [(reference.source_type, reference.source_id) for reference in references],
            [("calendar_event", "event-1"), ("saved_item", "item-1")],
        )

    def test_no_sources_produce_no_cards(self) -> None:
        self.assertEqual(answer_source_references([]), ([], []))


class StoredDateRenderingTests(unittest.TestCase):
    def test_a_reply_without_dated_rows_is_left_alone(self) -> None:
        reply = "The review is scheduled."

        self.assertEqual(with_stored_dates(reply, (_record("r", "No deadline"),), ()), reply)

    def test_each_cited_row_contributes_its_stored_date(self) -> None:
        start = datetime(2026, 9, 15, 10, 0, tzinfo=timezone.utc)
        end = datetime(2026, 9, 15, 11, 0, tzinfo=timezone.utc)

        rendered = with_stored_dates(
            "Two things.",
            (_record("r", "Ship the report", due_at=datetime(2026, 9, 18, tzinfo=timezone.utc)),),
            (_event("e", "Architecture review", start, end),),
        )

        self.assertIn("Stored dates:", rendered)
        self.assertIn(f"- Task deadline {datetime(2026, 9, 18, tzinfo=timezone.utc).isoformat()}", rendered)
        self.assertIn(f"- Calendar {start.isoformat()} to {end.isoformat()}: Architecture review", rendered)
        self.assertTrue(rendered.startswith("Two things."))

    def test_a_block_the_model_copied_is_replaced_by_the_stored_one(self) -> None:
        # The model sees earlier replies in the conversation and has copied this block from
        # one. Python owns the rendering, so a copy of it does not get through: the dates a
        # user reads come from SQLite, not from the model.
        copied = (
            "已创建日程：和导师开会。\n"
            "\n"
            "Stored dates:\n"
            "- Calendar 2026-09-16T15:00:00-07:00: 和导师开会"
        )
        start = datetime(2026, 9, 16, 15, 0, tzinfo=timezone.utc)

        rendered = with_stored_dates(copied, (), (_event("e", "和导师开会", start),))

        self.assertEqual(rendered.count("Stored dates:"), 1)
        self.assertIn(f"- Calendar {start.isoformat()}: 和导师开会", rendered)
        self.assertNotIn("2026-09-16T15:00:00-07:00", rendered)

    def test_a_copied_block_goes_even_when_nothing_is_cited(self) -> None:
        copied = "没找到。\n\nStored dates:\n- Calendar 2026-09-16: 编造的事件"

        self.assertEqual(with_stored_dates(copied, (), ()), "没找到。")

    def test_prose_around_a_copied_block_survives(self) -> None:
        copied = "两件事。\n\nStored dates:\n- Calendar 2026-09-16: 会议\n\n还需要确认地点。"

        rendered = with_stored_dates(copied, (), ())

        self.assertIn("两件事。", rendered)
        self.assertIn("还需要确认地点。", rendered)
        self.assertNotIn("Stored dates:", rendered)

    def test_the_words_in_ordinary_prose_are_left_alone(self) -> None:
        reply = "我核对了 stored dates 之后再答复你。"

        self.assertEqual(with_stored_dates(reply, (), ()), reply)

    def test_an_all_day_event_renders_without_an_end(self) -> None:
        rendered = with_stored_dates("One day.", (), (_event("e", "Evidence day", date(2026, 9, 17)),))

        self.assertIn("- Calendar 2026-09-17: Evidence day", rendered)
        self.assertNotIn(" to ", rendered)


if __name__ == "__main__":
    unittest.main()
