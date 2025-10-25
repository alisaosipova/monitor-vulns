import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deal_monitor import (
    ConfigError,
    Deal,
    DealMonitor,
    ItemQuery,
    MarketFetcher,
    MarketOffer,
    Settings,
    SteamQuote,
    build_item_query,
    build_settings,
    expand_env,
    format_money,
    iter_items_by_keys,
    load_config,
    normalize_wear,
    parse_cli_item,
    parse_market_hash_name,
    parse_optional_bool,
    resolve_steam_currency_id,
    to_string_list,
    to_string_set,
    coerce_float,
    coerce_int,
)


def test_resolve_steam_currency_id_known():
    assert resolve_steam_currency_id("usd") == 1


def test_resolve_steam_currency_id_unknown():
    with pytest.raises(ConfigError):
        resolve_steam_currency_id("zzz")


def test_normalize_wear_alias_and_case():
    assert normalize_wear("  MW  ") == "Minimal Wear"
    assert normalize_wear(None) is None
    assert normalize_wear(" ") is None


def test_parse_market_hash_name_variants():
    attrs = parse_market_hash_name("StatTrak™ Souvenir ★ AK-47 | Redline (Field-Tested)")
    assert attrs.weapon == "AK-47"
    assert attrs.skin == "Redline"
    assert attrs.wear == "Field-Tested"
    assert attrs.stattrak is True
    assert attrs.souvenir is True


def test_to_string_helpers():
    assert to_string_list("a, b ,c") == ["a", "b", "c"]
    assert to_string_list(["x", "y"]) == ["x", "y"]
    assert to_string_set({"a", "b"}) == {"a", "b"}
    assert to_string_list(None) == []


@pytest.mark.parametrize(
    "value,expected",
    [
        ("12,34", 12.34),
        ("1 234,56", 1234.56),
        ("1,234.50", None),
        (None, None),
        ("abc", None),
    ],
)
def test_coerce_float(value, expected):
    assert coerce_float(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [("1 234", 1234), ("abc", None), (12.0, 12), (None, None)],
)
def test_coerce_int(value, expected):
    assert coerce_int(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("true", True),
        ("No", False),
        ("1", True),
        (0, False),
        (None, None),
    ],
)
def test_parse_optional_bool_valid(value, expected):
    assert parse_optional_bool(value) == expected


def test_parse_optional_bool_invalid():
    with pytest.raises(ConfigError):
        parse_optional_bool("maybe")


def test_format_money():
    assert format_money(1234.5, "USD") == "1 234.50 USD"
    assert format_money(-5.5, "EUR") == "-5.50 EUR"
    assert format_money(10, "USD", signed=True) == "+10.00 USD"
    assert format_money(None, "USD") == "–"


def test_item_query_matches_offer_filters():
    query = ItemQuery(
        market_hash_name="AK-47 | Redline (Field-Tested)",
        wear_filters={"Field-Tested"},
        weapon_filters={"ak-47"},
        include_keywords={"redline"},
        exclude_keywords={"souvenir"},
        stattrak=False,
        souvenir=False,
    )
    offer = MarketOffer(
        market="steam",
        offer_id="1",
        market_hash_name="AK-47 | Redline (Field-Tested)",
        price=10.0,
        url="https://example.com",
    )
    assert query.matches_offer(offer)


def test_item_query_matches_offer_respects_filters():
    query = ItemQuery(
        market_hash_name="AK-47 | Redline (Field-Tested)",
        wear_filters={"Minimal Wear"},
        weapon_filters={"m4a1-s"},
    )
    offer = MarketOffer(
        market="steam",
        offer_id="1",
        market_hash_name="AK-47 | Redline (Field-Tested)",
        price=10.0,
        url="https://example.com",
    )
    assert not query.matches_offer(offer)


def test_item_query_dynamic_resolve_steam_name():
    query = ItemQuery(
        market_hash_name="AK-47 | Redline (Field-Tested)",
        dynamic=True,
    )
    offer = MarketOffer(
        market="cs.money",
        offer_id="1",
        market_hash_name="AK-47 | Redline (Field-Tested)",
        price=10.0,
        url="https://example.com",
    )
    assert query.resolve_steam_name(offer) == "AK-47 | Redline (Field-Tested)"


def test_expand_env_recursive(monkeypatch):
    monkeypatch.setenv("TEST_TOKEN", "secret")
    data = {
        "path": "${TEST_TOKEN}/dir",
        "nested": ["value", {"inner": "${TEST_TOKEN}"}],
    }
    expanded = expand_env(data)
    assert expanded["path"].endswith("secret/dir")
    assert expanded["nested"][1]["inner"] == "secret"


def test_load_config_reads_and_expands(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "123")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("key: ${API_KEY}")
    config = load_config(str(config_path))
    assert config["key"] == "123"


def test_iter_items_by_keys_extracts_lists():
    payload = {
        "data": {
            "items": [{"id": 1}, {"id": 2}],
            "nested": {"sellOrders": [{"id": 3}]},
        }
    }
    collected = list(iter_items_by_keys(payload, {"items", "sellOrders"}))
    assert collected == [[{"id": 1}, {"id": 2}], [{"id": 3}]]


def test_deal_dataclass_roundtrip():
    query = ItemQuery(market_hash_name="AK-47 | Redline (Field-Tested)")
    offer = MarketOffer(
        market="steam",
        offer_id="offer-1",
        market_hash_name=query.market_hash_name,
        price=20.0,
        url="https://example.com",
    )
    quote = SteamQuote(
        median_price=30.0,
        lowest_price=25.0,
        volume=10,
        fetched_at=datetime.now(timezone.utc),
    )
    # Проверяем базовую инициализацию и свойства Deal.
    deal = Deal(
        item=query,
        offer=offer,
        steam_quote=quote,
        target_sell_price=29.0,
        profit=9.0,
        roi=0.45,
        discount=0.1,
        score=0.5,
    )
    assert deal.market == "steam"


def test_settings_defaults_are_reasonable():
    settings = Settings()
    assert settings.refresh_interval == 60
    assert settings.currency == "USD"


def test_build_settings_applies_cli_overrides():
    raw = {
        "settings": {
            "refresh_interval": 120,
            "currency": "usd",
            "steam_fee": 0.15,
            "deal_score_weights": {"roi": 0.9, "discount": 0.05},
            "max_displayed_deals": 5,
        }
    }
    overrides = argparse.Namespace(refresh=45, currency="eur", min_roi=0.2, min_profit=3.5)
    settings = build_settings(raw, overrides)
    assert settings.refresh_interval == 45
    assert settings.currency == "EUR"
    assert settings.default_min_roi == 0.2
    assert settings.default_min_profit == 3.5
    assert settings.deal_score_weights["roi"] == 0.9
    assert settings.deal_score_weights["discount"] == 0.05
    assert settings.max_displayed_deals == 5


def test_build_item_query_from_config_section():
    settings = Settings(default_min_roi=0.15, default_min_profit=2.5)
    item_cfg = {
        "market_hash_name": "AK-47 | Redline (Field-Tested)",
        "min_price": 9,
        "max_price": 25,
        "markets": ["skinport", "steam"],
        "aliases": {"steam": "AK-47 | Redline (Field-Tested)"},
        "filters": {
            "wears": ["field-tested", "minimal wear"],
            "weapons": ["ak-47"],
            "stattrak": "false",
            "souvenir": "no",
            "contains": ["Redline"],
            "exclude": ["Souvenir"],
        },
    }
    query = build_item_query(item_cfg, settings)
    assert query.market_hash_name == "AK-47 | Redline (Field-Tested)"
    assert query.min_price == 9
    assert query.max_price == 25
    assert query.min_roi == 0.15
    assert query.min_profit == 2.5
    assert query.watch_markets == {"skinport", "steam"}
    assert query.wear_filters == {"Field-Tested", "Minimal Wear"}
    assert query.weapon_filters == {"ak-47"}
    assert query.stattrak is False
    assert query.souvenir is False
    assert query.include_keywords == {"redline"}
    assert query.exclude_keywords == {"souvenir"}


def test_parse_cli_item_supports_dynamic_queries():
    settings = Settings(default_min_roi=0.1, default_min_profit=1.5)
    expr = (
        "search=ak-47;dynamic=true;min_price=7;min_roi=0.4;min_profit=2;"
        "stattrak=yes;wears=Field-Tested;weapons=AK-47;contains=Redline;"
        "exclude=Souvenir;alias:steam=AK-47 | Redline (Field-Tested)"
    )
    query = parse_cli_item(expr, settings)
    assert query.dynamic is True
    assert query.search_term == "ak-47"
    assert query.min_price == 7
    assert query.min_roi == 0.4
    assert query.min_profit == 2
    assert query.stattrak is True
    assert query.wear_filters == {"Field-Tested"}
    assert query.weapon_filters == {"ak-47"}
    assert query.include_keywords == {"redline"}
    assert query.exclude_keywords == {"souvenir"}
    assert query.aliases["steam"] == "AK-47 | Redline (Field-Tested)"
    assert query.exact_name is False
    assert query.steam_name is None


class DummyFetcher(MarketFetcher):
    name = "dummy"
    supports_dynamic = True

    def __init__(self, settings: Settings, offers):
        super().__init__(settings)
        self._offers = list(offers)

    def fetch_offers(self, query: ItemQuery):
        return list(self._offers)


class DummySteamClient:
    def __init__(self, quotes):
        self.quotes = quotes
        self.requested = []

    def fetch(self, item_name: str):
        self.requested.append(item_name)
        return self.quotes.get(item_name)


def test_deal_monitor_poll_produces_scored_deals():
    settings = Settings(steam_fee=0.1, refresh_interval=5, currency="USD")
    query = ItemQuery(
        market_hash_name="AK-47 | Redline (Field-Tested)",
        min_price=5,
        max_price=15,
        min_roi=0.1,
        min_profit=2,
        steam_name="AK-47 | Redline (Field-Tested)",
    )
    offer = MarketOffer(
        market="skinport",
        offer_id="42",
        market_hash_name="AK-47 | Redline (Field-Tested)",
        price=10.0,
        url="https://example.com",
    )
    steam_quote = SteamQuote(
        median_price=20.0,
        lowest_price=18.0,
        volume=120,
        fetched_at=datetime.now(timezone.utc),
    )
    steam_client = DummySteamClient({query.steam_name: steam_quote})
    fetcher = DummyFetcher(settings, [offer])
    monitor = DealMonitor(settings, [fetcher], steam_client)

    deals, quotes, errors = monitor.poll([query])

    assert not errors
    assert steam_client.requested == [query.steam_name]
    assert quotes[query.steam_name] == steam_quote
    assert len(deals) == 1
    deal = deals[0]
    assert deal.offer.offer_id == "42"
    assert deal.profit == pytest.approx(8.0)
    assert deal.roi == pytest.approx(0.8)
    assert deal.discount == pytest.approx(0.5)
    assert deal.score > 0


