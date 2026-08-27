"""Conversation management: history, active topics, summaries, context."""

from conversation.manager import ConversationManager, ConversationState
from conversation.reference import Resolution, resolve_reference

__all__ = ["ConversationManager", "ConversationState", "Resolution", "resolve_reference"]
