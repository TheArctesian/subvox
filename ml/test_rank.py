import json

from subvox_ml.cli import rank


def test_rank_preserves_unrelated_reports_in_output_directory(tmp_path):
    run = tmp_path / "frozen-run"
    run.mkdir()
    (run / "comparison.json").write_text('{"dataset":"fixture"}')
    unrelated = tmp_path / ".rank-unrelated.json"
    unrelated.write_text('{"saved":true}')
    output = tmp_path / "report.json"

    rank(run, output)

    assert json.loads(output.read_text())["scores"] == {}
    assert json.loads(unrelated.read_text()) == {"saved": True}
