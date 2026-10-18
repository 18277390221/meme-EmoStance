"""Shared local rules for addressed standby and wake commands."""

from .commands import classify_command, is_sleep_command, is_wake_command

__all__ = ["classify_command", "is_sleep_command", "is_wake_command"]
