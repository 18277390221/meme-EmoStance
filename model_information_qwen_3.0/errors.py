"""Public inference errors independent of Torch and the model runtime."""


class PromptTooLong(ValueError):
    """The current turn cannot fit while preserving system instructions."""
