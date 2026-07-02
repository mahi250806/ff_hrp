import pytest
from pathlib import Path
from ffhrp.config import load_config, PortfolioConfig, RunConfig, VALID_FACTOR_MODELS, VALID_FREQUENCIES

CONFIG_PATH = Path(__file__).parent.parent / "config" / "portfolio.yaml"


def test_load_default_config():
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg, PortfolioConfig)
    assert cfg.run.factor_model == "carhart4"
    assert cfg.run.frequency == "daily"
    assert cfg.run.lookback_years == 2
    assert cfg.run.min_obs_ratio == 20
    assert len(cfg.assets) == 10


def test_all_tickers_present():
    cfg = load_config(CONFIG_PATH)
    tickers = {a.ticker for a in cfg.assets}
    expected = {"LLY", "MSFT", "GOOGL", "AVGO", "NVDA", "AMZN", "MU", "LITE", "COCO", "ARKG"}
    assert tickers == expected


def test_bad_factor_model_raises(tmp_path):
    yaml_content = """
run:
  frequency: daily
  factor_model: invalid_model
assets: []
"""
    p = tmp_path / "bad.yaml"
    p.write_text(yaml_content)
    with pytest.raises(ValueError, match="factor_model"):
        load_config(p)


def test_bad_frequency_raises(tmp_path):
    yaml_content = """
run:
  frequency: weekly
  factor_model: ff3
assets: []
"""
    p = tmp_path / "bad.yaml"
    p.write_text(yaml_content)
    with pytest.raises(ValueError, match="frequency"):
        load_config(p)


def test_all_valid_factor_models(tmp_path):
    for fm in VALID_FACTOR_MODELS:
        yaml_content = f"""
run:
  frequency: daily
  factor_model: {fm}
assets: []
"""
        p = tmp_path / f"{fm}.yaml"
        p.write_text(yaml_content)
        cfg = load_config(p)
        assert cfg.run.factor_model == fm


def test_all_valid_frequencies(tmp_path):
    for freq in VALID_FREQUENCIES:
        yaml_content = f"""
run:
  frequency: {freq}
  factor_model: ff3
assets: []
"""
        p = tmp_path / f"{freq}.yaml"
        p.write_text(yaml_content)
        cfg = load_config(p)
        assert cfg.run.frequency == freq
