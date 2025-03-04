class UserValidationError(ValueError):
    """Exception raised for user validation errors."""


class SubtaskValidationError(ValueError):
    """Exception raised for subtask validation errors."""


class DependencyValidationError(ValueError):
    """Exception raised for dependency validation errors."""


class TaskValidationError(ValueError):
    """Exception raised for task validation errors."""


class IngestionError(ValueError):
    """Exception raised for ingestion errors."""
