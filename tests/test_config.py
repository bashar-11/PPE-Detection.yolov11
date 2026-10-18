"""
Tests for the shared helpers and the dataset definition.

These deliberately avoid importing ultralytics or torch, so the suite runs in
a couple of seconds on a clean CI runner.
"""

import pytest

from src.utils import (
    format_metrics,
    load_yaml,
    project_path,
    resolve_device,
    summarize_metrics,
)


class TestLoadYaml:
    def test_returns_empty_dict_when_file_missing(self):
        assert load_yaml(project_path("configs", "does-not-exist.yaml")) == {}

    def test_returns_empty_dict_for_zero_byte_file(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        assert load_yaml(empty) == {}

    def test_reads_a_real_config(self):
        config = load_yaml(project_path("configs", "data.yaml"))
        assert config["nc"] == 2
        assert config["names"] == {0: "Reflective-Jacket", 1: "Safety-Helmet"}


class TestProjectPath:
    def test_builds_path_under_root(self):
        assert project_path("models", "best.pt").name == "best.pt"

    def test_accepts_a_single_component(self):
        assert project_path("README.md").name == "README.md"


class TestResolveDevice:
    def test_honours_an_explicit_request(self):
        assert resolve_device("0") == "0"
        assert resolve_device("cpu") == "cpu"

    def test_falls_back_to_a_string(self):
        # CPU-only machines and missing torch both land on "cpu".
        assert resolve_device(None) in {"cpu", "0"}


class FakeBox:
    mp = 0.8123
    mr = 0.7456
    map50 = 0.7890
    map = 0.5432


class FakeMetrics:
    box = FakeBox()


class TestSummarizeMetrics:
    def test_extracts_the_four_headline_numbers(self):
        summary = summarize_metrics(FakeMetrics())
        assert summary["precision"] == pytest.approx(0.8123)
        assert summary["recall"] == pytest.approx(0.7456)
        assert summary["map50"] == pytest.approx(0.7890)
        assert summary["map50_95"] == pytest.approx(0.5432)

    def test_returns_empty_when_no_box_attribute(self):
        class NoBox:
            pass

        assert summarize_metrics(NoBox()) == {}


class TestFormatMetrics:
    def test_renders_each_metric(self):
        text = format_metrics(summarize_metrics(FakeMetrics()))
        assert "Precision" in text
        assert "0.8123" in text
        assert "mAP@50-95" in text

    def test_handles_empty_summary(self):
        assert "No metrics" in format_metrics({})


class TestShippedConfigs:
    def test_train_config_has_required_keys(self):
        config = load_yaml(project_path("configs", "train_config.yaml"))
        for key in ["model", "data", "epochs", "imgsz", "batch", "workers"]:
            assert key in config, f"train_config.yaml is missing {key}"

    def test_train_config_points_at_a_config_that_exists(self):
        config = load_yaml(project_path("configs", "train_config.yaml"))
        assert project_path(config["data"]).exists()

    def test_inference_config_has_required_keys(self):
        config = load_yaml(project_path("configs", "inference_config.yaml"))
        for key in ["imgsz", "conf", "iou"]:
            assert key in config, f"inference_config.yaml is missing {key}"

    def test_confidence_thresholds_are_in_range(self):
        config = load_yaml(project_path("configs", "inference_config.yaml"))
        assert 0.0 < config["conf"] < 1.0
        assert 0.0 < config["iou"] <= 1.0


class TestRepoHygiene:
    def test_dataset_is_not_committed(self):
        # The dataset is redistributed by Roboflow under CC BY 4.0 and must
        # never end up back in the repository.
        assert not project_path("data").exists()

    def test_model_weights_are_present(self):
        # best.pt is the artefact the app serves, so it is committed on purpose.
        assert project_path("models", "best.pt").exists()
