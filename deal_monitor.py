#!/usr/bin/env python3
"""Мониторинг выгодных офферов CS:GO across cs.money, lis-skins.com, market.csgo.com, skinport.com и Steam."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import cloudscraper
import requests
import yaml
from requests import Response
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from urllib3.util.retry import Retry

console = Console()

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

STEAM_CURRENCY_IDS: Dict[str, int] = {
    "USD": 1,
    "GBP": 2,
    "EUR": 3,
    "CHF": 4,
    "RUB": 5,
    "BRL": 7,
    "JPY": 8,
    "NOK": 9,
    "IDR": 10,
    "MYR": 11,
    "PHP": 12,
    "SGD": 13,
    "THB": 14,
    "KRW": 16,
    "TRY": 18,
    "MXN": 19,
    "CAD": 20,
    "AUD": 21,
    "NZD": 22,
    "CNY": 23,
    "UAH": 36,
}


class ConfigError(RuntimeError):
    """Ошибки конфигурации."""


@dataclass
class Settings:
    refresh_interval: int = 60
    currency: str = "USD"
    steam_fee: float = 0.13
    default_min_roi: float = 0.05
    default_min_profit: float = 1.0
    request_timeout: float = 25.0
    verify_tls: bool = True
    proxies: Dict[str, str] = field(default_factory=dict)


@dataclass
class ItemQuery:
    market_hash_name: str
    min_price: float = 0.0
    max_price: Optional[float] = None
    min_roi: float = 0.05
    min_profit: float = 1.0
    watch_markets: Optional[Set[str]] = None
    aliases: Dict[str, str] = field(default_factory=dict)

    def name_for(self, market: str) -> str:
        return self.aliases.get(market, self.market_hash_name)


@dataclass
class SteamQuote:
    median_price: Optional[float]
    lowest_price: Optional[float]
    volume: Optional[int]
    fetched_at: datetime


@dataclass
class MarketOffer:
    market: str
    offer_id: str
    market_hash_name: str
    price: float
    url: str
    quantity: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass
class Deal:
    item: ItemQuery
    offer: MarketOffer
    steam_quote: SteamQuote
    target_sell_price: float
    profit: float
    roi: float

    @property
    def market(self) -> str:
        return self.offer.market


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return expand_env(data)


def coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = "".join(ch for ch in value if ch.isdigit() or ch in ",.")
        if not cleaned:
            return None
        if cleaned.count(",") > 1 and "." not in cleaned:
            cleaned = cleaned.replace(",", "")
        elif cleaned.count(",") == 1 and "." not in cleaned:
            cleaned = cleaned.replace(",", ".")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        if not digits:
            return None
        return int(digits)
    return None


def format_money(value: Optional[float], currency: str, signed: bool = False) -> str:
    if value is None:
        return "–"
    amount = abs(value)
    sign = ""
    if signed:
        sign = "+" if value >= 0 else "-"
    elif value < 0:
        sign = "-"
    formatted = f"{amount:,.2f}".replace(",", " ")
    return f"{sign}{formatted} {currency}"


def create_retry_session(verify: bool, proxies: Optional[Dict[str, str]] = None) -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.6, status_forcelist=(429, 500, 502, 503, 504))
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.verify = verify
    if proxies:
        session.proxies.update(proxies)
    return session


class SteamMarketClient:
    BASE_URL = "https://steamcommunity.com/market/priceoverview/"

    def __init__(self, settings: Settings):
        self.settings = settings
        currency_id = STEAM_CURRENCY_IDS.get(settings.currency.upper())
        if currency_id is None:
            raise ConfigError(f"Валюта {settings.currency!r} не поддерживается Steam API.")
        self.currency_id = currency_id
        self.session = create_retry_session(settings.verify_tls, settings.proxies)
        self.session.headers.update(
            {
                "User-Agent": DEFAULT_USER_AGENT,
                "Accept": "application/json, text/javascript",
                "X-Requested-With": "XMLHttpRequest",
            }
        )

    def fetch(self, item_name: str) -> Optional[SteamQuote]:
        params = {"appid": 730, "market_hash_name": item_name, "currency": self.currency_id}
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=self.settings.request_timeout)
            response.raise_for_status()
        except RequestException as exc:
            logging.warning("Steam запрос не удался для %s: %s", item_name, exc)
            return None
        try:
            data = response.json()
        except ValueError as exc:
            logging.warning("Steam вернул некорректный JSON для %s: %s", item_name, exc)
            return None
        if not data or not data.get("success", True):
            logging.info("Steam не нашёл предмет %s (ответ: %s)", item_name, data)
            return None
        median = coerce_float(data.get("median_price"))
        lowest = coerce_float(data.get("lowest_price"))
        volume = coerce_int(data.get("volume"))
        return SteamQuote(median, lowest, volume, datetime.now(timezone.utc))


class MarketFetcher:
    name: str = "abstract"

    def __init__(self, settings: Settings, config: Optional[Dict[str, Any]] = None):
        self.settings = settings
        self.config = config or {}

    def fetch_offers(self, query: ItemQuery) -> List[MarketOffer]:
        raise NotImplementedError

    def adjust_price(self, price: float) -> float:
        multiplier = float(self.config.get("price_multiplier", 1.0))
        return price * multiplier


class HTTPFetcher(MarketFetcher):
    def __init__(self, settings: Settings, config: Optional[Dict[str, Any]] = None):
        super().__init__(settings, config)
        self.session = create_retry_session(
            settings.verify_tls,
            self.config.get("proxies") or settings.proxies,
        )
        headers = {"User-Agent": self.config.get("user_agent", DEFAULT_USER_AGENT)}
        headers.update(self.config.get("headers") or {})
        self.session.headers.update(headers)
        cookies = self.config.get("cookies")
        if cookies:
            self.session.cookies.update(cookies)


class CloudflareFetcher(MarketFetcher):
    def __init__(self, settings: Settings, config: Optional[Dict[str, Any]] = None):
        super().__init__(settings, config)
        browser_cfg = {
            "browser": self.config.get("browser", "firefox"),
            "platform": self.config.get("platform", "windows"),
            "mobile": bool(self.config.get("mobile", False)),
        }
        self.scraper = cloudscraper.create_scraper(browser=browser_cfg)
        headers = {"User-Agent": self.config.get("user_agent", DEFAULT_USER_AGENT)}
        headers.update(self.config.get("headers") or {})
        self.scraper.headers.update(headers)
        cookies = self.config.get("cookies")
        if cookies:
            self.scraper.cookies.update(cookies)

    def request(self, method: str, url: str, **kwargs) -> Response:
        timeout = kwargs.pop("timeout", self.settings.request_timeout)
        try:
            return self.scraper.request(method, url, timeout=timeout, **kwargs)
        except RequestException as exc:
            raise RuntimeError(f"{self.name}: сетевой сбой — {exc}") from exc


def iter_items_by_keys(value: Any, keys: Iterable[str]):
    key_set = set(keys)
    yield from _iter_items_recursive(value, key_set)


def _iter_items_recursive(value: Any, key_set: Set[str]):
    if isinstance(value, dict):
        for key, val in value.items():
            if key in key_set and isinstance(val, list):
                yield val
            yield from _iter_items_recursive(val, key_set)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_items_recursive(item, key_set)


class SkinportFetcher(HTTPFetcher):
    name = "skinport"
    API_URL = "https://api.skinport.com/v1/item"

    def fetch_offers(self, query: ItemQuery) -> List[MarketOffer]:
        params = {
            "app_id": 730,
            "market_hash_name": query.name_for(self.name),
            "currency": self.config.get("currency", self.settings.currency),
        }
        if self.config.get("tradable_only", True):
            params["tradable"] = 1
        response = self.session.get(self.API_URL, params=params, timeout=self.settings.request_timeout)
        if response.status_code == 404:
            return []
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "errors" in data:
            raise RuntimeError(f"Skinport: {data['errors']}")
        items = data if isinstance(data, list) else [data]
        offers: List[MarketOffer] = []
        target = query.name_for(self.name).lower()
        for entry in items:
            name = (entry.get("market_hash_name") or entry.get("market_name") or "").lower()
            if name and name != target:
                continue
            price = coerce_float(entry.get("min_price") or entry.get("price"))
            if price is None:
                continue
            adjusted = self.adjust_price(price)
            offer_id = str(entry.get("item_id") or entry.get("market_hash_name"))
            url = self.config.get("item_url_template", "https://skinport.com/item/{name}").format(
                name=urllib.parse.quote(query.name_for(self.name))
            )
            offers.append(
                MarketOffer(
                    self.name,
                    offer_id,
                    query.market_hash_name,
                    adjusted,
                    url,
                    quantity=entry.get("quantity"),
                    extra={
                        "suggested": coerce_float(entry.get("suggested_price")),
                        "instant_sale": entry.get("instant_sale"),
                    },
                    raw=entry,
                )
            )
        return offers


class CSGOMarketFetcher(HTTPFetcher):
    name = "market.csgo.com"
    API_URL = "https://market.csgo.com/api/v2/search-list-items-by-hash-name-all"

    def __init__(self, settings: Settings, config: Optional[Dict[str, Any]] = None):
        super().__init__(settings, config)
        self.api_key = self.config.get("api_key") or os.getenv("CSGO_MARKET_API_KEY")
        if not self.api_key:
            raise ConfigError(
                "market.csgo.com требует API ключ (markets.market_csgo.api_key или переменная окружения CSGO_MARKET_API_KEY)."
            )
        self.config.setdefault("price_multiplier", 0.01)

    def fetch_offers(self, query: ItemQuery) -> List[MarketOffer]:
        params = {
            "key": self.api_key,
            "hash_name": query.name_for(self.name),
            "page": 1,
            "limit": self.config.get("limit", 60),
        }
        response = self.session.get(self.API_URL, params=params, timeout=self.settings.request_timeout)
        response.raise_for_status()
        data = response.json()
        if not data.get("success", False):
            raise RuntimeError(f"market.csgo.com: {data.get('error') or data.get('message') or 'неизвестная ошибка'}")
        items = data.get("items") or data.get("data") or []
        offers: List[MarketOffer] = []
        for entry in items:
            raw_price = entry.get("price") or entry.get("price_usd") or entry.get("price_float")
            price = coerce_float(raw_price)
            if price is None:
                continue
            adjusted = self.adjust_price(price)
            offer_id = str(entry.get("id") or entry.get("assetid") or entry.get("classid") or entry.get("instanceid"))
            url = entry.get("link") or self.config.get(
                "item_url_template",
                "https://market.csgo.com/item/{name}",
            ).format(name=urllib.parse.quote(query.name_for(self.name)))
            offers.append(
                MarketOffer(
                    self.name,
                    offer_id,
                    query.market_hash_name,
                    adjusted,
                    url,
                    quantity=entry.get("count") or entry.get("amount") or entry.get("total"),
                    extra={
                        "float": entry.get("float"),
                        "phase": entry.get("phase"),
                    },
                    raw=entry,
                )
            )
        return offers


class LisSkinsFetcher(CloudflareFetcher):
    name = "lis-skins.com"

    def fetch_offers(self, query: ItemQuery) -> List[MarketOffer]:
        page_url = self.config.get("page_url", "https://lis-skins.com/market/csgo")
        params = {"search": query.name_for(self.name)}
        response = self.request("GET", page_url, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"lis-skins.com вернул {response.status_code}")
        html = response.text
        marker = '<script id="__NEXT_DATA__" type="application/json">'
        start = html.find(marker)
        if start == -1:
            raise RuntimeError("lis-skins.com: не найден блок __NEXT_DATA__ (обновите headers/cookies).")
        start += len(marker)
        end = html.find("</script>", start)
        if end == -1:
            raise RuntimeError("lis-skins.com: не найден конец блока __NEXT_DATA__.")
        payload = json.loads(html[start:end])
        items: List[Dict[str, Any]] = []
        for candidate in iter_items_by_keys(payload, {"items", "sellOrders", "lots"}):
            if isinstance(candidate, list):
                items.extend(candidate)
        offers: List[MarketOffer] = []
        needle = query.name_for(self.name).lower()
        seen: Set[str] = set()
        for entry in items:
            name = (entry.get("market_hash_name") or entry.get("fullName") or entry.get("name") or "").lower()
            if needle not in name:
                continue
            price_value = entry.get("price") or entry.get("priceUsd") or entry.get("min_price")
            price = coerce_float(price_value)
            if price is None:
                continue
            adjusted = self.adjust_price(price)
            offer_id = str(
                entry.get("id")
                or entry.get("_id")
                or entry.get("assetid")
                or entry.get("steamId")
                or entry.get("hash")
                or name
            )
            if offer_id in seen:
                continue
            seen.add(offer_id)
            url = self.config.get(
                "item_url_template",
                "https://lis-skins.com/market/csgo?search={name}",
            ).format(name=urllib.parse.quote(query.name_for(self.name)))
            offers.append(
                MarketOffer(
                    self.name,
                    offer_id,
                    query.market_hash_name,
                    adjusted,
                    url,
                    quantity=entry.get("count") or entry.get("quantity") or entry.get("available"),
                    extra={
                        "float": entry.get("floatValue") or entry.get("wear"),
                        "stickers": entry.get("stickers"),
                    },
                    raw=entry,
                )
            )
        return offers


class CSMoneyFetcher(CloudflareFetcher):
    name = "cs.money"

    def fetch_offers(self, query: ItemQuery) -> List[MarketOffer]:
        if api_url := self.config.get("api_url"):
            return self._fetch_via_api(api_url, query)
        return self._fetch_via_page(query)

    def _fetch_via_api(self, url: str, query: ItemQuery) -> List[MarketOffer]:
        params = {
            "limit": self.config.get("limit", 60),
            "offset": 0,
            "sort": self.config.get("sort", "price"),
            "order": self.config.get("order", "asc"),
            "search": query.name_for(self.name),
        }
        response = self.request("GET", url, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"cs.money API вернул {response.status_code}")
        data = response.json()
        items = data.get("items") or data.get("sellOrders") or data.get("data") or []
        return self._parse_items(items, query)

    def _fetch_via_page(self, query: ItemQuery) -> List[MarketOffer]:
        page_url = self.config.get("page_url", "https://cs.money/csgo/trade")
        params = {"search": query.name_for(self.name)}
        response = self.request("GET", page_url, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"cs.money страница вернула {response.status_code}")
        html = response.text
        marker = 'id="__NEXT_DATA__" type="application/json">'
        start = html.find(marker)
        if start == -1:
            raise RuntimeError("cs.money: не удалось найти __NEXT_DATA__ (нужны свежие cookie/headers).")
        start += len(marker)
        end = html.find("</script>", start)
        if end == -1:
            raise RuntimeError("cs.money: повреждённый ответ, не найден </script>.")
        payload = json.loads(html[start:end])
        items: List[Dict[str, Any]] = []
        for candidate in iter_items_by_keys(payload, {"items", "sellOrders", "list"}):
            if isinstance(candidate, list):
                items.extend(candidate)
        return self._parse_items(items, query)

    def _parse_items(self, items: List[Dict[str, Any]], query: ItemQuery) -> List[MarketOffer]:
        offers: List[MarketOffer] = []
        target = query.name_for(self.name).lower()
        seen: Set[str] = set()
        for entry in items:
            name = (entry.get("market_hash_name") or entry.get("fullName") or entry.get("name") or "").lower()
            if target not in name:
                continue
            price_value = (
                entry.get("price")
                or entry.get("priceUsd")
                or entry.get("priceUSD")
                or (entry.get("price") or {}).get("usd")
            )
            price = coerce_float(price_value)
            if price is None:
                continue
            adjusted = self.adjust_price(price)
            offer_id = str(
                entry.get("id")
                or entry.get("item_id")
                or entry.get("assetid")
                or entry.get("steamId")
                or entry.get("id64")
                or name
            )
            if offer_id in seen:
                continue
            seen.add(offer_id)
            url = self.config.get(
                "item_url_template",
                "https://cs.money/csgo/trade/?search={name}",
            ).format(name=urllib.parse.quote(query.name_for(self.name)))
            offers.append(
                MarketOffer(
                    self.name,
                    offer_id,
                    query.market_hash_name,
                    adjusted,
                    url,
                    quantity=entry.get("count") or entry.get("available") or entry.get("amount"),
                    extra={
                        "float": entry.get("floatValue") or entry.get("wear"),
                        "stickers": entry.get("stickers"),
                    },
                    raw=entry,
                )
            )
        return offers


class DealMonitor:
    def __init__(self, settings: Settings, fetchers: Sequence[MarketFetcher], steam_client: SteamMarketClient):
        self.settings = settings
        self.fetchers = list(fetchers)
        self.steam_client = steam_client
        self.last_seen: Dict[str, float] = {}

    def run(self, queries: Sequence[ItemQuery]) -> None:
        console.print(f"[bold]Старт мониторинга — {len(queries)} предмет(ов)[/bold]")
        while True:
            started = time.monotonic()
            deals, quotes, errors = self.poll(queries)
            self.render(deals, quotes, errors)
            elapsed = time.monotonic() - started
            sleep_for = max(0.0, self.settings.refresh_interval - elapsed)
            try:
                time.sleep(sleep_for)
            except KeyboardInterrupt:
                raise

    def poll(self, queries: Sequence[ItemQuery]):
        deals: List[Deal] = []
        quotes: Dict[str, SteamQuote] = {}
        errors: List[str] = []
        for query in queries:
            steam_quote = self.steam_client.fetch(query.name_for("steam"))
            if steam_quote is None:
                errors.append(f"Steam: нет цены для {query.market_hash_name}")
                continue
            quotes[query.market_hash_name] = steam_quote
            for fetcher in self.fetchers:
                if query.watch_markets and fetcher.name not in query.watch_markets:
                    continue
                try:
                    offers = fetcher.fetch_offers(query)
                except Exception as exc:
                    logging.exception("Ошибка в %s для %s", fetcher.name, query.market_hash_name)
                    errors.append(f"{fetcher.name}: {exc}")
                    continue
                for offer in offers:
                    deal = self.evaluate_offer(query, steam_quote, offer)
                    if deal:
                        key = f"{deal.market}:{deal.offer.offer_id}"
                        self.last_seen[key] = time.time()
                        deals.append(deal)
        deals.sort(key=lambda d: (d.roi, d.profit), reverse=True)
        now = time.time()
        for key, ts in list(self.last_seen.items()):
            if now - ts > self.settings.refresh_interval * 3:
                self.last_seen.pop(key, None)
        return deals, quotes, errors

    def evaluate_offer(self, query: ItemQuery, steam_quote: SteamQuote, offer: MarketOffer) -> Optional[Deal]:
        target_price = steam_quote.median_price or steam_quote.lowest_price
        if target_price is None:
            return None
        sell_after_fee = target_price * (1 - self.settings.steam_fee)
        profit = sell_after_fee - offer.price
        if offer.price < query.min_price:
            return None
        if query.max_price is not None and offer.price > query.max_price:
            return None
        if profit < query.min_profit:
            return None
        roi = profit / offer.price if offer.price else 0.0
        if roi < query.min_roi:
            return None
        if sell_after_fee <= 0:
            return None
        return Deal(query, offer, steam_quote, target_price, profit, roi)

    def render(self, deals: Sequence[Deal], quotes: Dict[str, SteamQuote], errors: Sequence[str]) -> None:
        console.clear()
        header = f"Мониторинг офферов — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        table = Table(title=header, box=box.SIMPLE_HEAVY, show_lines=False)
        table.add_column("Маркет")
        table.add_column("Предмет", style="bold")
        table.add_column("Покупка", justify="right")
        table.add_column("Продажа (Steam)", justify="right")
        table.add_column("Прибыль", justify="right")
        table.add_column("ROI", justify="right")
        table.add_column("Ссылка", overflow="fold")
        if not deals:
            table.add_row("—", "Нет выгодных предложений", "—", "—", "—", "—", "—")
        for deal in deals:
            table.add_row(
                deal.market,
                deal.item.market_hash_name,
                format_money(deal.offer.price, self.settings.currency),
                format_money(deal.target_sell_price, self.settings.currency),
                format_money(deal.profit, self.settings.currency, signed=True),
                f"{deal.roi * 100:+.2f}%",
                deal.offer.url,
            )
        console.print(table)
        steam_table = Table(title="Steam показатели", box=box.SIMPLE_HEAD)
        steam_table.add_column("Предмет")
        steam_table.add_column("Median", justify="right")
        steam_table.add_column("Lowest", justify="right")
        steam_table.add_column("Volume", justify="right")
        for name, quote in quotes.items():
            steam_table.add_row(
                name,
                format_money(quote.median_price, self.settings.currency),
                format_money(quote.lowest_price, self.settings.currency),
                f"{quote.volume:,}".replace(",", " ") if quote.volume is not None else "–",
            )
        console.print(steam_table)
        if errors:
            console.print(Panel("\n".join(errors), title="Ошибки за цикл", style="red"))


def build_settings(raw: Dict[str, Any], overrides: argparse.Namespace) -> Settings:
    data = raw.get("settings", {})
    settings = Settings(
        refresh_interval=int(data.get("refresh_interval", 60)),
        currency=(data.get("currency") or "USD").upper(),
        steam_fee=float(data.get("steam_fee", 0.13)),
        default_min_roi=float(data.get("default_min_roi", 0.05)),
        default_min_profit=float(data.get("default_min_profit", 1.0)),
        request_timeout=float(data.get("request_timeout", 25.0)),
        verify_tls=bool(data.get("verify_tls", True)),
        proxies=data.get("proxies") or {},
    )
    if overrides.refresh:
        settings.refresh_interval = overrides.refresh
    if overrides.currency:
        settings.currency = overrides.currency.upper()
    if overrides.min_roi is not None:
        settings.default_min_roi = overrides.min_roi
    if overrides.min_profit is not None:
        settings.default_min_profit = overrides.min_profit
    return settings


def build_item_query(item: Dict[str, Any], settings: Settings) -> ItemQuery:
    name = item.get("market_hash_name") or item.get("name")
    if not name:
        raise ConfigError("Каждый предмет должен иметь поле market_hash_name.")
    min_price = float(item.get("min_price", 0.0) or 0.0)
    max_price_raw = item.get("max_price")
    max_price = float(max_price_raw) if max_price_raw not in (None, "", "none") else None
    min_roi = float(item.get("min_roi", settings.default_min_roi))
    min_profit = float(item.get("min_profit", settings.default_min_profit))
    markets = item.get("markets")
    watch_markets = {m.strip() for m in markets} if isinstance(markets, (list, set, tuple)) else None
    aliases = item.get("aliases") or {}
    return ItemQuery(
        market_hash_name=name,
        min_price=min_price,
        max_price=max_price,
        min_roi=min_roi,
        min_profit=min_profit,
        watch_markets=watch_markets,
        aliases=aliases,
    )


def parse_cli_item(expr: str, settings: Settings) -> ItemQuery:
    parts: Dict[str, str] = {}
    for chunk in expr.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ConfigError(f"Неверный формат '--item {expr}'. Используйте key=value;...")
        key, value = chunk.split("=", 1)
        parts[key.strip()] = value.strip()
    name = parts.pop("name", None)
    if not name:
        raise ConfigError("--item должен содержать как минимум name=...")
    min_price = float(parts.pop("min_price", 0.0) or 0.0)
    max_price_raw = parts.pop("max_price", None)
    max_price = float(max_price_raw) if max_price_raw not in (None, "", "none") else None
    min_roi = float(parts.pop("min_roi", settings.default_min_roi))
    min_profit = float(parts.pop("min_profit", settings.default_min_profit))
    watch_raw = parts.pop("markets", None)
    watch_markets = {m.strip() for m in watch_raw.split(",")} if watch_raw else None
    aliases: Dict[str, str] = {}
    for key in list(parts.keys()):
        if key.startswith("alias:"):
            market = key.split(":", 1)[1]
            aliases[market] = parts.pop(key)
    if parts:
        raise ConfigError(f"Неизвестные параметры у --item: {', '.join(parts.keys())}")
    return ItemQuery(
        market_hash_name=name,
        min_price=min_price,
        max_price=max_price,
        min_roi=min_roi,
        min_profit=min_profit,
        watch_markets=watch_markets,
        aliases=aliases,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Мониторинг выгодных предложений CS:GO.")
    parser.add_argument("-c", "--config", help="Путь к YAML конфигурации")
    parser.add_argument("--item", action="append", help="Добавить/переопределить предмет: name=...;min_price=...;min_roi=...;...")
    parser.add_argument("--refresh", type=int, help="Интервал обновления в секундах")
    parser.add_argument("--currency", help="Код валюты Steam (USD, EUR, RUB, ...)")
    parser.add_argument("--min-roi", type=float, help="Минимальный ROI по умолчанию")
    parser.add_argument("--min-profit", type=float, help="Минимальная прибыль по умолчанию")
    parser.add_argument("--log-level", default="INFO", help="Уровень логирования (DEBUG, INFO, ...)")
    return parser.parse_args()


def build_fetchers(settings: Settings, markets_cfg: Dict[str, Any]) -> List[MarketFetcher]:
    fetchers: List[MarketFetcher] = []
    registry = [
        (CSMoneyFetcher, "cs_money"),
        (LisSkinsFetcher, "lis_skins"),
        (CSGOMarketFetcher, "market_csgo"),
        (SkinportFetcher, "skinport"),
    ]
    for cls, key in registry:
        cfg = markets_cfg.get(key, {})
        if not cfg or not cfg.get("enabled", True):
            continue
        try:
            fetcher = cls(settings, cfg)
        except ConfigError as exc:
            logging.warning("Маркет %s пропущен: %s", key, exc)
            continue
        fetchers.append(fetcher)
    return fetchers


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    settings = build_settings(config, args)
    items_cfg = config.get("items") or []
    queries = [build_item_query(item, settings) for item in items_cfg]
    if args.item:
        for expr in args.item:
            queries.append(parse_cli_item(expr, settings))
    if not queries:
        raise ConfigError("Не указано ни одного предмета (config.yaml -> items или аргумент --item).")
    fetchers = build_fetchers(settings, config.get("markets") or {})
    if not fetchers:
        raise ConfigError("Нет активных маркетов (markets.*.enabled = true).")
    steam_client = SteamMarketClient(settings)
    monitor = DealMonitor(settings, fetchers, steam_client)
    try:
        monitor.run(queries)
    except KeyboardInterrupt:
        console.print("[yellow]Мониторинг остановлен пользователем[/yellow]")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as exc:
        console.print(f"[red]Ошибка конфигурации:[/red] {exc}")
        sys.exit(2)
