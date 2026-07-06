"""Tests for the Warp-Ingest parse provider wrapper."""

from __future__ import annotations

import sys
from datetime import datetime
from types import ModuleType
from typing import Any

import pytest

import parse_bench.inference.providers.parse.warp_ingest as warp_ingest_module
from parse_bench.evaluation.layout_adapters.adapters import WarpIngestLayoutAdapter
from parse_bench.inference.providers.base import ProviderConfigError
from parse_bench.inference.providers.parse.warp_ingest import WarpIngestProvider
from parse_bench.schemas.parse_output import ParseOutput
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from parse_bench.schemas.product import ProductType


def _pipeline() -> PipelineSpec:
    return PipelineSpec(
        pipeline_name="warp_ingest",
        provider_name="warp_ingest",
        product_type=ProductType.PARSE,
        config={},
    )


def _request(pdf_path: str) -> InferenceRequest:
    return InferenceRequest(
        example_id="example-1",
        source_file_path=pdf_path,
        product_type=ProductType.PARSE,
    )


def test_run_inference_delegates_to_upstream_markdown_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    calls: list[dict[str, Any]] = []

    def fake_parse_to_markdown_payload(
        doc_location: str,
        *,
        parse_options: dict[str, Any],
        include_native_tables: bool,
    ) -> dict[str, Any]:
        calls.append(
            {
                "doc_location": doc_location,
                "parse_options": parse_options,
                "include_native_tables": include_native_tables,
            }
        )
        return {"pages": [{"page_index": 0, "blocks": []}], "source": "warp"}

    def fake_render_pages(_payload: dict[str, Any]) -> list[tuple[int, str]]:
        return []

    monkeypatch.setattr(
        warp_ingest_module,
        "_load_markdown_exporter",
        lambda: (fake_parse_to_markdown_payload, fake_render_pages),
    )

    provider = WarpIngestProvider("warp_ingest", {"parse_options": {"apply_ocr": True}})
    raw_result = provider.run_inference(_pipeline(), _request(str(pdf_path)))

    assert raw_result.raw_output == {"pages": [{"page_index": 0, "blocks": []}], "source": "warp"}
    assert calls == [
        {
            "doc_location": str(pdf_path),
            "parse_options": {"apply_ocr": True, "render_format": "all"},
            "include_native_tables": True,
        }
    ]


def test_normalize_delegates_rendering_to_upstream_exporter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    def fake_parse_to_markdown_payload(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"payload": "raw"}

    def fake_render_pages(payload: dict[str, Any]) -> list[tuple[int, str]]:
        assert payload == {"payload": "raw"}
        return [(0, "# Title"), (1, "<table><tr><td>x</td></tr></table>")]

    monkeypatch.setattr(
        warp_ingest_module,
        "_load_markdown_exporter",
        lambda: (fake_parse_to_markdown_payload, fake_render_pages),
    )

    provider = WarpIngestProvider("warp_ingest")
    raw_result = provider.run_inference(_pipeline(), _request(str(pdf_path)))
    result = provider.normalize(raw_result)

    assert isinstance(result.output, ParseOutput)
    assert [page.page_index for page in result.output.pages] == [0, 1]
    assert [page.markdown for page in result.output.pages] == [
        "# Title",
        "<table><tr><td>x</td></tr></table>",
    ]
    assert result.output.markdown == "# Title\n\n<table><tr><td>x</td></tr></table>"


def test_invalid_parse_options_are_rejected() -> None:
    with pytest.raises(ProviderConfigError, match="parse_options"):
        WarpIngestProvider("warp_ingest", {"parse_options": "apply_ocr=true"})


def test_layout_adapter_delegates_to_upstream_generic_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_render_layout_predictions(
        payload: dict[str, Any],
        *,
        page_filter: int | None = None,
    ) -> dict[str, Any]:
        calls.append({"payload": payload, "page_filter": page_filter})
        return {
            "image_width": 612,
            "image_height": 792,
            "predictions": [
                {
                    "bbox": [1, 2, 3, 4],
                    "score": 1.0,
                    "label": "Text",
                    "page": 1,
                    "content": {"type": "text", "text": "Hello"},
                    "order_index": 7,
                },
                {
                    "bbox": [5, 6, 7, 8],
                    "score": 1.0,
                    "label": "Table",
                    "page": 1,
                    "content": {"type": "table", "html": "<table></table>"},
                    "order_index": 8,
                },
            ],
        }

    warp_package = ModuleType("warp_ingest")
    warp_package.__path__ = []
    ingestor_package = ModuleType("warp_ingest.ingestor")
    ingestor_package.__path__ = []
    exporter_module = ModuleType("warp_ingest.ingestor.markdown_exporter")
    exporter_module.render_layout_predictions = fake_render_layout_predictions
    monkeypatch.setitem(sys.modules, "warp_ingest", warp_package)
    monkeypatch.setitem(sys.modules, "warp_ingest.ingestor", ingestor_package)
    monkeypatch.setitem(sys.modules, "warp_ingest.ingestor.markdown_exporter", exporter_module)

    request = InferenceRequest(
        example_id="layout-1",
        source_file_path="/tmp/doc.pdf",
        product_type=ProductType.PARSE,
    )
    output = ParseOutput(
        task_type="parse",
        example_id=request.example_id,
        pipeline_name="warp_ingest",
        pages=[],
        markdown="",
    )
    result = InferenceResult(
        request=request,
        pipeline_name="warp_ingest",
        product_type=ProductType.PARSE,
        raw_output={"blocks": [{"block_text": "Hello"}]},
        output=output,
        started_at=datetime.now(),
        completed_at=datetime.now(),
        latency_in_ms=1,
    )

    layout = WarpIngestLayoutAdapter().to_layout_output(result, page_filter=1)

    assert calls == [{"payload": {"blocks": [{"block_text": "Hello"}]}, "page_filter": 1}]
    assert layout.image_width == 612
    assert layout.image_height == 792
    assert [prediction.label for prediction in layout.predictions] == ["Text", "Table"]
    assert layout.predictions[0].content is not None
    assert layout.predictions[0].content.type == "text"
    assert layout.predictions[1].content is not None
    assert layout.predictions[1].content.type == "table"
    assert layout.predictions[0].provider_metadata["order_index"] == 7
