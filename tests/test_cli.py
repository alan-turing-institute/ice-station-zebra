import re

from typer.testing import CliRunner

from ice_station_zebra.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    expected_patterns = [
        r"│ datasets\s+Manage datasets",
        r"│ train",
    ]
    lines = result.output.split("\n")
    for pattern in expected_patterns:
        assert any([re.search(pattern, line) for line in lines])
