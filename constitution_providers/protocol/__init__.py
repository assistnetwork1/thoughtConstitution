# constitution_providers/protocol/__init__.py
from .retriever import RetrieverProvider
from .reasoner import ReasoningProvider

# If runner expects ProposalProvider, define it explicitly:
ProposalProvider = ReasoningProvider

__all__ = ["RetrieverProvider", "ReasoningProvider", "ProposalProvider"]
