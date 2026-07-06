"""Provider for Warp-Ingest PARSE."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
)
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import PageIR, ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType

_DEFAULT_PARSE_OPTIONS: dict[str, Any] = {
    "apply_ocr": False,
    "render_format": "all",
}

ParsePayloadFn = Callable[..., dict[str, Any]]
RenderPagesFn = Callable[[dict[str, Any]], list[tuple[int, str]]]


def _load_markdown_exporter() -> tuple[ParsePayloadFn, RenderPagesFn]:
    try:
        from warp_ingest.ingestor.markdown_exporter import (
            parse_to_markdown_payload,
            render_pages,
        )
    except ImportError as exc:
        raise ProviderConfigError(
            "warp-ingest>=1.0.2 not installed. Run: pip install 'warp-ingest>=1.0.2'"
        ) from exc

    return parse_to_markdown_payload, render_pages


@register_provider("warp_ingest")
class WarpIngestProvider(Provider):
    """Provider for Warp-Ingest (local, no API key)."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        configured_parse_options = self.base_config.get("parse_options")
        if configured_parse_options is None:
            configured_parse_options = {}
        if not isinstance(configured_parse_options, dict):
            raise ProviderConfigError("warp_ingest parse_options must be a dictionary")
        self._parse_options = {**_DEFAULT_PARSE_OPTIONS, **configured_parse_options}
        self._include_native_tables = bool(self.base_config.get("include_native_tables", True))

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"WarpIngestProvider only supports PARSE, got {request.product_type}")

        pdf_path = Path(request.source_file_path)
        if pdf_path.suffix.lower() != ".pdf":
            raise ProviderPermanentError(f"WarpIngestProvider only supports .pdf files, got {pdf_path.suffix}")
        if not pdf_path.exists():
            raise ProviderPermanentError(f"PDF file not found: {pdf_path}")

        parse_to_markdown_payload = _load_markdown_exporter()[0]

        started_at = datetime.now()
        try:
            raw_output = parse_to_markdown_payload(
                str(pdf_path),
                parse_options=self._parse_options,
                include_native_tables=self._include_native_tables,
            )
        except Exception as exc:
            raise ProviderPermanentError(f"Warp-Ingest parse error: {exc}") from exc
        completed_at = datetime.now()

        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=raw_output,
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=int((completed_at - started_at).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"WarpIngestProvider only supports PARSE, got {raw_result.product_type}")

        render_pages = _load_markdown_exporter()[1]

        try:
            rendered = render_pages(raw_result.raw_output)
        except Exception as exc:
            raise ProviderPermanentError(f"Warp-Ingest normalize error: {exc}") from exc

        pages = [PageIR(page_index=page_index, markdown=markdown) for page_index, markdown in rendered]
        full_text = "\n\n".join(markdown for _, markdown in rendered)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            markdown=full_text,
        )

        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_result.raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )
