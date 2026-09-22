"""Verify persisted evidence, model identity, and an independent lexical reference."""

import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from retrieval_bench.beir import import_beir
from retrieval_bench.cli import app
from retrieval_bench.config import ModelConfig
from retrieval_bench.lexical import BM25Index, evaluate_bm25
from retrieval_bench.models import BenchmarkResults, Document, RetrievalQuery
from retrieval_bench.provenance import build_provenance


def records():
    return [
        Document(id="a", title="fruit", text="apple apple"),
        Document(id="b", title="fruit", text="orange orange"),
    ]


def test_bm25_formula_and_independent_ranking():
    index = BM25Index(records())
    ranked = index.search("APPLE", 2)
    # Equal document lengths simplify the denominator to tf + k1.
    expected = math.log(2) * 2 * 2.2 / (2 + 1.2)
    assert ranked[0][0] == "a"
    assert ranked[0][1] == pytest.approx(expected)
    assert ranked[1] == ("b", 0)
    assert index.search("unknown", 9) == [("a", 0), ("b", 0)]


@pytest.mark.parametrize("kwargs", [{"k1": 0}, {"k1": float("nan")}, {"b": 2}])
def test_bm25_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        BM25Index(records(), **kwargs)


def test_baseline_records_query_evidence():
    ticks = iter([1.0, 1.125])
    query = RetrievalQuery(id="q", query="orange", relevant_doc_ids=["b"])
    output = evaluate_bm25(records(), [query], k_values=[1, 10], clock=lambda: next(ticks))
    assert output["metrics"]["mrr@10"] == 1
    assert output["query_results"][0]["ranked_doc_ids"] == ["b", "a"]
    assert output["query_results"][0]["latency_ms"] == 125
    assert output["provenance"]["document_count"] == 2
    with pytest.raises(ValueError):
        evaluate_bm25(records(), [], k_values=[10])
    with pytest.raises(ValueError):
        evaluate_bm25(records(), [query], k_values=[])
    with pytest.raises(ValueError):
        BM25Index(records()).search("a", 0)


def test_provenance_identifies_actual_data_and_unpinned_model():
    first = build_provenance(records(), [], model_name="example", model_revision=None)
    changed = records()
    changed[0] = changed[0].model_copy(update={"text": "changed"})
    second = build_provenance(changed, [], model_name="example", model_revision=None)
    assert first["corpus_sha256"] != second["corpus_sha256"]
    assert first["queries_sha256"] == second["queries_sha256"]
    assert first["model_revision_pinned"] is False


def test_revision_requires_immutable_commit():
    ModelConfig(name="model", batch_size=4, revision="a" * 40)
    with pytest.raises(ValueError):
        ModelConfig(name="model", batch_size=4, revision="main")


def beir_fixture(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "corpus.jsonl").write_text(
        "\n".join(json.dumps({"_id": d.id, "title": d.title, "text": d.text}) for d in records()),
        encoding="utf-8",
    )
    (source / "queries.jsonl").write_text(
        '{"_id":"q","text":"apple"}\n{"_id":"unused","text":"unjudged"}', encoding="utf-8"
    )
    qrels = source / "test.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\nq\ta\t2\nq\tb\t0\n", encoding="utf-8")
    return source, qrels


def test_beir_import_keeps_negatives_and_selected_split(tmp_path):
    source, qrels = beir_fixture(tmp_path)
    destination = tmp_path / "output"
    assert import_beir(source, qrels, destination) == (2, 1)
    assert len((destination / "corpus.jsonl").read_text().splitlines()) == 2
    query = json.loads((destination / "queries.jsonl").read_text())
    assert query["relevant_doc_ids"] == ["a"]
    with pytest.raises(FileExistsError):
        import_beir(source, qrels, destination)


@pytest.mark.parametrize(
    "contents",
    [
        "wrong\theader\n",
        "query-id\tcorpus-id\tscore\n",
        "query-id\tcorpus-id\tscore\nq\tmissing\t1\n",
    ],
)
def test_beir_rejects_invalid_qrels_without_output(tmp_path, contents):
    source, qrels = beir_fixture(tmp_path)
    qrels.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        import_beir(source, qrels, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_new_cli_paths_and_historical_artifact_compatibility(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["bm25", "--output", str(tmp_path / "baseline.json")])
    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / "baseline.json").read_text())["query_results"]
    source, qrels = beir_fixture(tmp_path)
    args = ["import-beir", str(source), "--qrels", str(qrels), "--output", str(tmp_path / "out")]
    assert runner.invoke(app, args).exit_code == 0
    assert runner.invoke(app, args).exit_code == 1
    assert runner.invoke(app, ["bm25", "--config", str(tmp_path / "absent")]).exit_code == 1
    old = BenchmarkResults.model_validate_json(Path("artifacts/results.json").read_text())
    assert old.provenance == {} and old.results[0].query_results == []


@pytest.mark.parametrize("row", ["q\ta\tnan", "q\ta\tinf", "q\ta", "\ta\t1"])
def test_beir_rejects_malformed_scores_and_incomplete_rows(tmp_path, row):
    source, qrels = beir_fixture(tmp_path)
    qrels.write_text("query-id\tcorpus-id\tscore\n" + row + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        import_beir(source, qrels, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_beir_rejects_normalized_query_collisions_before_writing(tmp_path):
    source, qrels = beir_fixture(tmp_path)
    (source / "queries.jsonl").write_text(
        '{"_id":"q","text":"apple"}\n{"_id":" q ","text":"orange"}', encoding="utf-8"
    )
    qrels.write_text("query-id\tcorpus-id\tscore\nq\ta\t1\n q \tb\t1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="normalized"):
        import_beir(source, qrels, tmp_path / "out")
    assert not (tmp_path / "out").exists()
