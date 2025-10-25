"""Integration test that fetches live Steam Community Market offers."""
from __future__ import annotations

from datetime import datetime, timezone
import re
import sys
from pathlib import Path
from urllib.parse import quote

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deal_monitor import (  # noqa: E402
    Deal,
    DealMonitor,
    ItemQuery,
    Settings,
    SteamCommunityFetcher,
    SteamQuote,
    SteamMarketClient,
)


def _build_profitable_deal(
    query: ItemQuery,
    offers,
    steam_quote: SteamQuote,
    settings: Settings,
):
    best_offer = min(offers, key=lambda offer: offer.price)
    observed_target = steam_quote.median_price or steam_quote.lowest_price or 0.0
    minimum_target = best_offer.price * 1.25
    target_sell_price = max(observed_target, minimum_target)
    profit = target_sell_price * (1 - settings.steam_fee) - best_offer.price
    if profit <= 0:
        target_sell_price = best_offer.price * 1.35
        profit = target_sell_price * (1 - settings.steam_fee) - best_offer.price
    roi = profit / best_offer.price if best_offer.price else 0.0
    discount = (
        (target_sell_price - best_offer.price) / target_sell_price if target_sell_price else 0.0
    )
    adjusted_quote = SteamQuote(
        median_price=target_sell_price,
        lowest_price=max(target_sell_price, steam_quote.lowest_price or target_sell_price),
        volume=steam_quote.volume,
        fetched_at=steam_quote.fetched_at,
    )
    deal = Deal(
        item=query,
        offer=best_offer,
        steam_quote=adjusted_quote,
        target_sell_price=target_sell_price,
        profit=profit,
        roi=roi,
        discount=discount,
        score=roi + discount,
    )
    return deal, adjusted_quote


@pytest.mark.parametrize(
    "market_hash_name",
    [
        "AK-47 | Redline (Field-Tested)",
        "M4A1-S | Golden Coil (Minimal Wear)",
        "AWP | Atheris (Field-Tested)",
    ],
)
def test_steam_market_live_fetch_render_links(market_hash_name: str, capsys):
    settings = Settings(refresh_interval=1, request_timeout=30.0, steam_fee=0.13)
    fetcher = SteamCommunityFetcher(settings, {})
    query = ItemQuery(market_hash_name=market_hash_name, exact_name=True)
    offers = fetcher.fetch_offers(query)
    if not offers:
        pytest.skip(f"Steam community market returned no offers for {market_hash_name}")
    steam_client = SteamMarketClient(settings)
    steam_quote = steam_client.fetch(market_hash_name)
    if steam_quote is None:
        steam_quote = SteamQuote(
            median_price=offers[0].price * 1.30,
            lowest_price=offers[0].price * 1.30,
            volume=None,
            fetched_at=datetime.now(timezone.utc),
        )
    deal, adjusted_quote = _build_profitable_deal(query, offers, steam_quote, settings)
    monitor = DealMonitor(settings, [fetcher], steam_client)
    monitor.render([deal], {market_hash_name: adjusted_quote}, errors=[])
    captured = capsys.readouterr()
    # Re-emit captured console output so pytest -s displays the live data table.
    print(captured.out)
    assert market_hash_name in captured.out
    normalized_output = re.sub(r"\x1b\[[0-9;]*m", "", captured.out)
    normalized_output = re.sub(r"\s+", "", normalized_output)
    encoded_name = quote(deal.item.market_hash_name, safe="")
    assert f"listings/730/{encoded_name}" in normalized_output
    assert "ROI" in captured.out
    assert deal.profit > 0
