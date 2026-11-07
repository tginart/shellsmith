"""Detached, Studio-owned jobs controlled through bounded SSH requests."""

from .client import Endpoint, JobClient

__all__ = ["Endpoint", "JobClient"]
