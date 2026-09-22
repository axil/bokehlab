from copy import deepcopy

import pytest

import bokehlab
from bokehlab import config


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch, tmp_path):
    original = deepcopy(config.CONFIG)
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "bokehlab.yaml")
    monkeypatch.setattr(config, "CONFIG_LOADED", False)
    bokehlab.FIGURES.clear()
    yield
    bokehlab.FIGURES.clear()
    config.CONFIG.clear()
    config.CONFIG.update(original)
