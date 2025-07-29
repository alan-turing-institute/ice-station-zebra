import re

from click.testing import Result
from typer.testing import CliRunner

from ice_station_zebra.cli import app

runner = CliRunner()


def plain_lines(result: Result) -> list[str]:
    """Helper to strip colour codes from the output."""
    colorstrip = re.compile(r"\x1b[^m]*m")
    return [colorstrip.sub("", line) for line in result.output.split("\n")]


def test_help():
    result = runner.invoke(app, ["--help"])
    expected_patterns = [
        r"Usage: zebra [OPTIONS] COMMAND [ARGS]...",
        r"Entrypoint for zebra application commands",
        r"--install-completion\s+Install completion for the current shell.",
        r"--show-completion\s+Show completion for the current shell, to copy it or customize the installation.",
        r"--help\s+-h\s+Show this message and exit.",
        r"datasets\s+Manage datasets",
        r"evaluate\s+Evaluate a model",
        r"train\s+Train a model",
    ]
    lines = plain_lines(result)
    for pattern in expected_patterns:
        assert any([re.search(pattern, line) for line in lines])


def test_datasets_help():
    result = runner.invoke(app, ["datasets", "--help"])
    expected_patterns = [
        r"Usage: zebra datasets [OPTIONS] COMMAND [ARGS]...",
        r"Manage datasets",
        r"--help\s+-h\s+Show this message and exit.",
        r"create\s+Create all datasets",
        r"inspect\s+Inspect all datasets",
    ]
    lines = plain_lines(result)
    for pattern in expected_patterns:
        assert any([re.search(pattern, line) for line in lines])


def test_evaluate_help():
    result = runner.invoke(app, ["train", "--help"])
    expected_patterns = [
        r"Usage: zebra evaluate [OPTIONS] [OVERRIDES]...",
        r"Evaluate a model",
        r"overrides\s+[OVERRIDES]...\s+Apply space-separated Hydra config overrides (https://hydra.cc/docs/advanced/override_grammar/basic/) [default: None]",
        r"--checkpoint\s+TEXT\s+Specify the path to a trained model checkpoint [default: None] [required]",
        r"--config-name\s+TEXT\s+Specify the name of a file to load from the config directory [default: zebra]",
        r"--help\s+-h\s+Show this message and exit.",
    ]
    lines = plain_lines(result)
    for pattern in expected_patterns:
        assert any([re.search(pattern, line) for line in lines])


def test_train_help():
    result = runner.invoke(app, ["train", "--help"])
    expected_patterns = [
        r"Usage: zebra train [OPTIONS] [OVERRIDES]...",
        r"Train a model",
        r"overrides\s+[OVERRIDES]...\s+Apply space-separated Hydra config overrides (https://hydra.cc/docs/advanced/override_grammar/basic/) [default: None]",
        r"--config-name\s+TEXT\s+Specify the name of a file to load from the config directory [default: zebra]",
        r"--help\s+-h\s+Show this message and exit.",
    ]
    lines = plain_lines(result)
    for pattern in expected_patterns:
        assert any([re.search(pattern, line) for line in lines])
