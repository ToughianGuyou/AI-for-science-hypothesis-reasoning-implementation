"""Serializable error types shared by the prototype."""


class HypothesisReasoningError(Exception):
    """Base exception for expected domain and pipeline failures."""


class ContractValidationError(HypothesisReasoningError):
    """Raised when an input or output violates a domain contract."""


class InputValidationError(ContractValidationError):
    """Raised when a case input is missing, malformed, or internally inconsistent."""


class InputReferenceError(ContractValidationError):
    """Raised when a case input refers to unavailable or disallowed records."""


class TopicDiscoveryError(ContractValidationError):
    """Raised when topic discovery remains invalid after one repair attempt."""


class TopicSelectionError(ContractValidationError):
    """Raised when a topic or score batch violates the selection contract."""


class RefinementLimitError(HypothesisReasoningError):
    """Raised before a hypothesis lineage would receive a second revision."""


class RefinementValidationError(ContractValidationError):
    """Raised when a proposed revision violates lineage or gate constraints."""


class PipelineConfigurationError(HypothesisReasoningError):
    """Raised when an explicit experiment mode lacks a required component."""


class AuditValidationError(ContractValidationError):
    """Raised when a candidate source or audit record is not reproducible."""


class SkillAdaptationError(ContractValidationError):
    """Raised when an external skill cannot be reduced to safe prompt-only instructions."""


class BudgetConfigurationError(HypothesisReasoningError):
    """Raised when a budget partition or amount is invalid."""


class BudgetExceededError(HypothesisReasoningError):
    """Raised before a model call would exceed its budget partition."""


class PricingConfigurationError(HypothesisReasoningError):
    """Raised when model pricing is missing, stale, or malformed."""


class GatewayConfigurationError(HypothesisReasoningError):
    """Raised when the Qwen gateway is not safe to call."""


class GatewayResponseError(HypothesisReasoningError):
    """Raised when an API response is unsuccessful or incomplete."""


class StructuredOutputError(GatewayResponseError):
    """Raised when JSON output remains invalid after the single repair attempt."""


class ModelChangedError(GatewayResponseError):
    """Raised when a rolling model changes inside one controlled experiment batch."""
