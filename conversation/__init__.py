"""Conversation management: history, active topics, summaries, context."""

from conversation.context import ConversationContext, classify_intent
from conversation.entities import Entity, EntityResolution, EntityTracker
from conversation.manager import ConversationManager, ConversationState
from conversation.pending import (
    PendingQuestion,
    PendingResolution,
    detect_pending_question,
    resolve_pending,
)
from conversation.reference import Resolution, resolve_reference
from conversation.topics import Topic, TopicStack

__all__ = [
    "ConversationManager", "ConversationState", "Resolution", "resolve_reference",
    "Entity", "EntityResolution", "EntityTracker", "Topic", "TopicStack",
    "PendingQuestion", "PendingResolution", "detect_pending_question", "resolve_pending",
    "ConversationContext", "classify_intent",
]
