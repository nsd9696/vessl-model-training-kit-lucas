from typing import Literal

from pydantic import BaseModel, Field


class MCQ_QA_BaseResponse(BaseModel):
    answer: Literal["a", "b", "c", "d", "e"]


class MCQ_QA_ReasoningResponse(MCQ_QA_BaseResponse):
    reasoning: str = Field(
        ...,
        description="3 Sentences Reasoning for the answer in same language as the question",
    )
