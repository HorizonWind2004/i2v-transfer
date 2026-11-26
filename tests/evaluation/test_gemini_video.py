import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from i2vt.evaluation import gemini_video, rubrics, score

REPLY = "Instruction Compliance: 4\nVisual Quality & Stability: 3\nConsistency & Detail Fidelity: 4"


@pytest.fixture
def client(monkeypatch):
    client = Mock()
    factory = MagicMock()
    factory.return_value.__enter__.return_value = client
    monkeypatch.setattr(gemini_video, "create_client", factory)
    monkeypatch.setattr(gemini_video.time, "sleep", lambda _: None)
    client.files.upload.side_effect = [
        SimpleNamespace(name="original", state="PROCESSING"),
        SimpleNamespace(name="edited", state="ACTIVE"),
    ]
    client.files.get.return_value = SimpleNamespace(name="original", state="ACTIVE")
    client.models.generate_content.return_value = SimpleNamespace(text=REPLY)
    return client


def evaluate():
    return gemini_video.evaluate_video_pair(
        Path("source.mp4"),
        Path("edited.mp4"),
        "official prompt",
        api_key="test-key",
        model="gemini-2.5-pro",
        timeout=120,
        parse_response=score.check_format,
    )


def test_uploads_two_complete_videos_and_waits_before_scoring(client):
    scores, text = evaluate()
    assert scores == [4, 3, 4] and text == REPLY
    assert [c.kwargs["file"] for c in client.files.upload.call_args_list] == ["source.mp4", "edited.mp4"]
    client.files.get.assert_called_once_with(name="original")
    args = client.models.generate_content.call_args.kwargs
    assert args["model"] == "gemini-2.5-pro"
    assert args["config"] == {"system_instruction": "official prompt"}
    assert [f.name for f in args["contents"]] == ["original", "edited"]
    assert all(f.state == "ACTIVE" for f in args["contents"])
    assert [c.kwargs["name"] for c in client.files.delete.call_args_list] == ["original", "edited"]


def test_second_upload_failure_cleans_up_first_file(client):
    client.files.upload.side_effect = [SimpleNamespace(name="first", state="ACTIVE"), RuntimeError("upload failed")]
    with pytest.raises(RuntimeError, match="upload failed"):
        evaluate()
    client.files.delete.assert_called_once_with(name="first")
    client.models.generate_content.assert_not_called()


def test_failed_video_processing_never_scores(client):
    client.files.get.return_value = SimpleNamespace(name="original", state="FAILED")
    with pytest.raises(RuntimeError, match="processing failed"):
        evaluate()
    client.models.generate_content.assert_not_called()
    assert client.files.delete.call_count == 2


def test_processing_has_a_deadline(client, monkeypatch):
    monkeypatch.setattr(gemini_video.time, "monotonic", Mock(side_effect=[0, 121]))
    with pytest.raises(TimeoutError):
        evaluate()
    client.models.generate_content.assert_not_called()
    assert client.files.delete.call_count == 2


def test_bad_responses_are_retried_without_reuploading(client):
    client.models.generate_content.side_effect = [
        RuntimeError("temporary"),
        SimpleNamespace(text="bad"),
        SimpleNamespace(text=REPLY),
    ]
    assert evaluate()[0] == [4, 3, 4]
    assert client.files.upload.call_count == 2
    assert client.models.generate_content.call_count == 3


def test_scoring_failure_cleans_files(client):
    client.models.generate_content.return_value = SimpleNamespace(text="bad")
    with pytest.raises(RuntimeError, match="three attempts"):
        evaluate()
    assert client.files.delete.call_count == 2


def setup_case(tmp_path):
    bench = tmp_path / "bench.csv"
    bench.write_text("edited_type,original_video,prompt\nglobal_style,videos/a.mp4,paint\n")
    source = tmp_path / "videos/a.mp4"
    source.parent.mkdir()
    source.write_bytes(b"source video")
    case = tmp_path / "results/global_style/a"
    case.mkdir(parents=True)
    (case / "gen_a.mp4").write_bytes(b"edited video")
    return [
        "--video_paths",
        str(tmp_path / "results"),
        "--bench-csv",
        str(bench),
        "--bench-dir",
        str(tmp_path),
        "--filter-types",
        "global_style",
    ]


def test_cli_scores_once_and_resumes(tmp_path, monkeypatch, client):
    argv = setup_case(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    # OpenVE-Bench categories use the official rubrics, whose global_style axes have their own names.
    client.models.generate_content.return_value = SimpleNamespace(
        text="Style Fidelity: 4\nContent Preservation: 4\nTemporal Stability: 3"
    )
    path = score.score_path(tmp_path / "results", "gemini-2.5-pro")
    # A row from another protocol never marks the case done.
    path.write_text(json.dumps({"base": "a", "scores": [1, 1, 1], "protocol": "other"}) + "\n")
    score.main(argv)
    row = json.loads(path.read_text().splitlines()[-1])
    assert row["protocol"] == "gemini_video" and row["model"] == "gemini-2.5-pro"
    assert row["scores"] == [4, 3, 4]
    assert row["rubric"] == rubrics.OFFICIAL_SOURCE
    assert row["original_video"] == str(tmp_path / "videos/a.mp4")
    before = path.read_bytes()
    score.main(argv)
    assert path.read_bytes() == before
    client.models.generate_content.assert_called_once()


def test_failed_scoring_is_reported(tmp_path, monkeypatch, client):
    argv = setup_case(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client.models.generate_content.return_value = SimpleNamespace(text="bad")
    with pytest.raises(RuntimeError, match="Scoring failed"):
        score.main(argv)


def test_api_keys_come_from_a_file_or_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "key-a, key-b")
    assert score.load_api_keys() == ["key-a", "key-b"]
    keys = tmp_path / "keys.txt"
    keys.write_text("# comment\nkey-c\n")
    assert score.load_api_keys(str(keys)) == ["key-c"]
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(SystemExit):
        score.load_api_keys()
