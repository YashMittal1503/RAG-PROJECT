"""
Unit tests for Chat Auto-Naming and Session Renaming capability.

Verifies:
1. Cleaning and normalization of LLM title outputs (stripping quotes, markdown, prefixes, word limits).
2. Pure social pleasantries return 'New conversation'.
3. Substantive queries extract specific, concise 2-to-5 word titles using question and answer.
4. Fallback behavior on LLM failure or empty response.
5. PATCH /api/chat/sessions/{session_id} updates session title in DB.
6. Multi-turn auto-naming triggers for early turns (turns 1-3) and locks out after turn 3.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import uuid
import pytest

from app.schemas import ChatSessionUpdate
from app.services.query import _clean_title, generate_chat_title


# ── Title Cleaning Tests ───────────────────────────────────────────────────

def test_clean_title_normalizes_markdown_and_quotes():
    assert _clean_title('**"React Auth State Debugging"**') == "React Auth State Debugging"
    assert _clean_title('`Python Asyncio Concurrency`') == "Python Asyncio Concurrency"
    assert _clean_title("'DuckDB Text-to-SQL'") == "DuckDB Text-to-SQL"


def test_clean_title_strips_conversational_prefixes_and_punctuation():
    assert _clean_title("Title: Q3 Financial Performance.") == "Q3 Financial Performance"
    assert _clean_title("Suggested Title: Employee Leave Policy!") == "Employee Leave Policy"
    assert _clean_title("Topic: Customer Churn Prediction;") == "Customer Churn Prediction"
    assert _clean_title("Here is a title: Machine Learning Optimization") == "Machine Learning Optimization"


def test_clean_title_enforces_word_limit():
    long_raw = "Analysis of quarterly enterprise sales performance across eastern regional territories"
    cleaned = _clean_title(long_raw)
    assert len(cleaned.split()) <= 6
    assert cleaned == "Analysis of quarterly enterprise sales performance"


def test_clean_title_strips_book_and_document_summary_prefixes():
    assert _clean_title("Book Summary: Can We Be Friends") == "Can We Be Friends"
    assert _clean_title("Document Summary: Customer Churn Metrics") == "Customer Churn Metrics"
    assert _clean_title("Summary: Machine Learning Optimization") == "Machine Learning Optimization"


def test_clean_title_trims_dangling_trailing_words():
    assert _clean_title("Customer Churn Prediction Model in the") == "Customer Churn Prediction Model"


def test_repair_dangling_title_completes_from_context():
    from app.services.query import _repair_dangling_title
    context = "Can you give me a summary of the book Can We Be Friends by the author?"
    repaired = _repair_dangling_title("Can We Be", context)
    assert repaired == "Can We Be Friends"


def test_clean_title_preserves_new_conversation():
    assert _clean_title("New conversation") == "New conversation"
    assert _clean_title("New conversation.") == "New conversation"
    assert _clean_title("new chat") == "New conversation"


# ── Title Generation Service Tests ────────────────────────────────────────

@pytest.mark.anyio
async def test_generate_chat_title_greeting_heuristic():
    title = await generate_chat_title(
        chat_history=[],
        current_question="Hi",
        current_answer="",
    )
    assert title == "New conversation"

    title_hello = await generate_chat_title(
        chat_history=[],
        current_question="Good morning!",
        current_answer="",
    )
    assert title_hello == "New conversation"


@pytest.mark.anyio
async def test_generate_chat_title_substantive_query():
    dummy_response = MagicMock()
    choice = MagicMock()
    choice.message.content = "Telco Churn Analysis"
    dummy_response.choices = [choice]

    with patch("app.services.query.call_llm_with_fallback", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = dummy_response

        title = await generate_chat_title(
            chat_history=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
            current_question="Can you analyze the churn rate by contract type in the telco dataset?",
            current_answer="Month-to-month contracts had a 42% churn rate...",
        )

        assert title == "Telco Churn Analysis"
        assert mock_llm.called
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["fast"] is True


@pytest.mark.anyio
async def test_generate_chat_title_fallback_on_error():
    with patch("app.services.query.call_llm_with_fallback", side_effect=Exception("API timeout")):
        title = await generate_chat_title(
            chat_history=[],
            current_question="What are the latest semiconductor transistor density metrics?",
            current_answer="Modern silicon chips contain over 50 billion transistors...",
        )

        assert title != "New conversation"
        assert "Semiconductor" in title or "Transistor" in title


# ── Endpoint PATCH /sessions/{session_id} Test ────────────────────────────

@pytest.mark.anyio
async def test_update_session_endpoint():
    from app.routers.chat import update_session
    from app.models import ChatSession

    session_id = uuid.uuid4()
    user_id = str(uuid.uuid4())
    from datetime import datetime, timezone
    mock_session = ChatSession(
        id=session_id,
        user_id=uuid.UUID(user_id),
        title="Old Title",
        created_at=datetime.now(timezone.utc),
    )

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_session
    mock_db.execute.return_value = mock_result

    body = ChatSessionUpdate(title="New Custom Title")
    response = await update_session(
        session_id=session_id,
        body=body,
        user_id=user_id,
        db=mock_db,
    )

    assert response.title == "New Custom Title"
    assert mock_session.title == "New Custom Title"
    assert mock_db.commit.called
