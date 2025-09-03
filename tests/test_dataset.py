import pytest
from typer.testing import CliRunner

from gogo_test.dataset import app

runner = CliRunner()


def test_app():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "main" in result.stdout
