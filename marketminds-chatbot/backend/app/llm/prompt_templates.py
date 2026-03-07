"""Prompt templates for MarketMinds Chatbot LLM interactions.

This module contains all the system and user prompt templates used in
interactions with large language models (LLMs).

Supports optional chat_history for multi-turn conversations.
"""

from typing import Optional


def system_prompt() -> str:
    """
    System-level instructions for the LLM.
    """
    return (
        "You are MarketMinds, an advanced AI assistant for market and financial analysis.\n"
        "Follow these rules strictly:\n"
        "1. If the provided CONTEXT contains relevant information, you MUST base your answer on it.\n"
        "2. If the CONTEXT does NOT contain sufficient information, you MAY use your general knowledge.\n"
        "3. If you use general knowledge, clearly state that the answer is based on general knowledge, not the documents.\n"
        "4. Do NOT contradict or override information found in the CONTEXT.\n"
        "5. Do NOT fabricate numbers, dates, or events.\n"
        "6. Be concise, factual, and clear.\n"
        "7. Use markdown formatting for readability (bold, lists, headings).\n"
        "- Do NOT invent stock prices, ratios, or financial data.\n"
        "- If data is missing, explicitly say so.\n"
        "- Use cautious, analytical language.\n"
        "- Prefer explanation over speculation.\n"
        "- If a ticker or company is unknown, say it clearly.\n"
    )


def user_prompt(question: str) -> str:
    """Format the user's question into a prompt for the LLM.

    Args:
        question: The user's question.

    Returns:
        The formatted user prompt.
    """
    return f"Question:\n{question}"


def format_chat_history(chat_history: list[dict]) -> str:
    """Format chat history into a readable conversation block.

    Args:
        chat_history: List of dicts with 'role' and 'content' keys.
                      role is 'user' or 'assistant'.

    Returns:
        Formatted conversation string.
    """
    if not chat_history:
        return ""

    lines = ["=== Previous Conversation ==="]
    for msg in chat_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prefix = "User" if role == "user" else "MarketMinds"
        lines.append(f"{prefix}: {content}")
    lines.append("=== End of Previous Conversation ===")

    return "\n".join(lines)


def full_prompt(
    question: str,
    context: Optional[str] = None,
    chat_history: Optional[list[dict]] = None,
) -> str:
    """Combines system prompt, chat history, context, and user question
    into a full prompt for the LLM.

    Args:
        question:     The user's current question.
        context:      Optional retrieved context (RAG / fundamentals).
        chat_history: Optional list of previous messages for multi-turn support.

    Returns:
        The complete prompt for the LLM.
    """
    prompt_parts = [system_prompt()]

    # Add conversation history if available
    if chat_history:
        history_block = format_chat_history(chat_history)
        if history_block:
            prompt_parts.append(history_block)

    # Add retrieved context
    if context:
        prompt_parts.append(f"Context:\n{context}")

    # Add the current question
    prompt_parts.append(user_prompt(question))

    return "\n\n".join(prompt_parts)