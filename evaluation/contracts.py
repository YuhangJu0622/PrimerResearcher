"""Versioned, runtime-independent contracts consumed by the LLM judge."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DATASET_FORMAT_VERSION = "deepresearch-pairwise-judge/v1"


class JudgeToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    arguments: dict[str, Any]


class JudgePathStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=1)
    reasoning_content: str = ""
    tool_calls: list[JudgeToolCall] = Field(default_factory=list)


class JudgeTrajectory(BaseModel):
    """The judge-visible view: no system/user messages or tool observations."""

    model_config = ConfigDict(extra="forbid")

    path: list[JudgePathStep]
    answer: str = Field(min_length=1)

    @model_validator(mode="after")
    def steps_are_contiguous(self) -> "JudgeTrajectory":
        if [step.step for step in self.path] != list(range(1, len(self.path) + 1)):
            raise ValueError("path step numbers must be contiguous and start at 1")
        return self


class SampleProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The original format only supported a reference embedded in test.jsonl.
    # Keep those fields optional so existing judge datasets remain readable,
    # while allowing another run directory to provide the reference trajectory.
    reference_type: Literal["test_file", "run_dir"] = "test_file"
    test_file: str | None = None
    test_line: int | None = Field(default=None, ge=1)
    test_row_sha256: str | None = None
    run_dir: str
    trajectory_file: str
    trajectory_sha256: str
    reference_run_dir: str | None = None
    reference_trajectory_file: str | None = None
    reference_trajectory_sha256: str | None = None

    @model_validator(mode="after")
    def reference_source_is_complete(self) -> "SampleProvenance":
        if self.reference_type == "test_file":
            if not (self.test_file and self.test_line and self.test_row_sha256):
                raise ValueError("test_file provenance is incomplete")
        elif not (
            self.reference_run_dir
            and self.reference_trajectory_file
            and self.reference_trajectory_sha256
        ):
            raise ValueError("run_dir provenance is incomplete")
        return self


class EvaluationSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format_version: Literal["deepresearch-pairwise-judge/v1"] = DATASET_FORMAT_VERSION
    id: str = Field(min_length=1)
    question_number: int = Field(ge=1)
    query: str = Field(min_length=1)
    candidate: JudgeTrajectory
    reference: JudgeTrajectory
    provenance: SampleProvenance
