import pytest
import yaml

from bokehlab import Figure, config


@pytest.mark.parametrize("setting, expected", [
    ("resources='inline'", {"resources": {"mode": "inline"}}),
    ("figure.width=200", {"figure": {"width": 200}}),
    ("width=400 height=250", {"figure": {"width": 400, "height": 250}}),
])
def test_save_reload(setting, expected):
    config.configure("-g " + setting)
    assert yaml.safe_load(config.CONFIG_FILE.read_text()) == expected
    config.CONFIG.clear()
    config.CONFIG_LOADED = False
    config.load_config()
    assert config.CONFIG == expected


def test_local_settings_do_not_write():
    config.configure("figure.width=200")
    assert config.CONFIG["figure"]["width"] == 200
    assert not config.CONFIG_FILE.exists()
    assert Figure().width == 200
    assert Figure(width=500).width == 500


@pytest.mark.parametrize("key", ["figure.width", "figure", "resources.mode"])
def test_delete_global(key):
    config.configure("-g figure.width=200 resources='inline'")
    config.configure("-g -d " + key)
    saved = yaml.safe_load(config.CONFIG_FILE.read_text())
    section, _, option = key.partition(".")
    if option:
        assert option not in saved.get(section, {})
        assert option not in config.CONFIG.get(section, {})
    else:
        assert section not in saved
        assert section not in config.CONFIG


def test_delete_local_preserves_disk():
    config.configure("-g figure.width=200")
    config.configure("-d figure.width")
    assert yaml.safe_load(config.CONFIG_FILE.read_text()) == {"figure": {"width": 200}}
    assert "width" not in config.CONFIG["figure"]


def test_clear():
    config.configure("-g figure.width=200")
    config.configure("--clear --force")
    assert not config.CONFIG_FILE.exists()


def test_empty_file():
    config.CONFIG_FILE.touch()
    config.load_config()
    assert config.read_config() == {}


def test_invalid_resource_does_not_replace_default():
    config.configure("resources='invalid'")
    assert config.CONFIG["resources"]["mode"] == "cdn"
