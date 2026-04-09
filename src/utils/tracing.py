"""Trace context and span creation for observability.

Each span records: component name, start/end time, model_id, latency,
token_count, full prompt and response text. Writes to JSONL file.
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Span:
    component: str
    model_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    latency_ms: float = 0.0
    token_count: int = 0
    prompt: str = ""
    response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def finish(self) -> None:
        self.end_time = time.time()
        self.latency_ms = (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "component": self.component,
            "model_id": self.model_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "latency_ms": round(self.latency_ms, 2),
            "token_count": self.token_count,
            "prompt": self.prompt,
            "response": self.response,
            "metadata": self.metadata,
        }


class TraceContext:
    """Container for a query's trace spans. Writes to JSONL on flush."""

    def __init__(self, trace_dir: str = "traces") -> None:
        self.trace_id: str = uuid.uuid4().hex[:16]
        self.spans: list[Span] = []
        self.trace_dir = Path(trace_dir)

    def create_span(self, component: str, model_id: str = "") -> Span:
        span = Span(component=component, model_id=model_id, start_time=time.time())
        self.spans.append(span)
        return span

    def flush(self) -> Path:
        """Write all spans to a JSONL file and return the path."""
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.trace_dir / f"{self.trace_id}.jsonl"
        with open(path, "w") as f:
            for span in self.spans:
                f.write(json.dumps(span.to_dict()) + "\n")
        return path

    def to_list(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.spans]
