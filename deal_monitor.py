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

DEFAULT_SCORE_WEIGHTS: Dict[str, float] = {"roi": 0.6, "profit": 0.3, "discount": 0.1}


class ConfigError(RuntimeError):
    """Ошибки конфигурации."""


def resolve_steam_currency_id(code: str) -> int:
    try:
        return STEAM_CURRENCY_IDS[code.upper()]
    except KeyError as exc:
        raise ConfigError(f"Валюта {code!r} не поддерживается Steam API.") from exc


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
    deal_score_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SCORE_WEIGHTS)
    )
    max_displayed_deals: Optional[int] = None


WEAR_ALIASES: Dict[str, str] = {
    "fn": "Factory New",
    "factory new": "Factory New",
    "minimal wear": "Minimal Wear",
    "mw": "Minimal Wear",
    "field-tested": "Field-Tested",
    "ft": "Field-Tested",
    "well-worn": "Well-Worn",
    "ww": "Well-Worn",
    "battle-scarred": "Battle-Scarred",
    "bs": "Battle-Scarred",
}


def normalize_wear(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    return WEAR_ALIASES.get(cleaned, value.strip())


def normalize_weapon(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


@dataclass(frozen=True)
class ItemAttributes:
    original_name: str
    weapon: Optional[str]
    skin: Optional[str]
    wear: Optional[str]
    stattrak: bool
    souvenir: bool


def parse_market_hash_name(name: Optional[str]) -> ItemAttributes:
    source = (name or "").strip()
    stattrak = source.startswith("StatTrak™ ")
    if stattrak:
        source = source[len("StatTrak™ ") :]
    souvenir = source.startswith("Souvenir ")
    if souvenir:
        source = source[len("Souvenir ") :]
    source = source.lstrip("★ ").strip()
    weapon: Optional[str] = None
    skin: Optional[str] = None
    wear: Optional[str] = None
    if " | " in source:
        weapon, rest = source.split(" | ", 1)
        weapon = weapon.strip() or None
    else:
        rest = source
    rest = rest.strip()
    if rest.endswith(")") and "(" in rest:
        base, wear_part = rest.rsplit("(", 1)
        skin = base.strip() or None
        wear = normalize_wear(wear_part[:-1])
    else:
        skin = rest or None
    return ItemAttributes(name or "", weapon, skin, wear, stattrak, souvenir)


def to_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    result: List[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def to_string_set(value: Any) -> Set[str]:
    return set(to_string_list(value))


def parse_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigError(f"Не удалось интерпретировать логическое значение: {value!r}")


@dataclass
class ItemQuery:
    market_hash_name: str
    min_price: float = 0.0
    max_price: Optional[float] = None
    min_roi: float = 0.05
    min_profit: float = 1.0
    watch_markets: Optional[Set[str]] = None
    aliases: Dict[str, str] = field(default_factory=dict)
    search_term: Optional[str] = None
    steam_name: Optional[str] = None
    exact_name: bool = True
    dynamic: bool = False
    wear_filters: Set[str] = field(default_factory=set)
    weapon_filters: Set[str] = field(default_factory=set)
    stattrak: Optional[bool] = None
    souvenir: Optional[bool] = None
    include_keywords: Set[str] = field(default_factory=set)
    exclude_keywords: Set[str] = field(default_factory=set)

    def name_for(self, market: str) -> str:
        if market == "steam":
            if self.steam_name is not None:
                return self.steam_name
            if "steam" in self.aliases:
                return self.aliases["steam"]
        return self.aliases.get(market, self.market_hash_name)

    def search_for(self, market: str) -> str:
        if market in self.aliases:
            return self.aliases[market]
        if self.search_term and market != "steam":
            return self.search_term
        return self.market_hash_name

    def matches_search(self, candidate: str, market: str) -> bool:
        if self.exact_name:
            return candidate == self.search_for(market).lower()
        term = self.search_for(market).lower()
        return not term or term in candidate

    def resolve_steam_name(self, offer: "MarketOffer") -> Optional[str]:
        if self.dynamic:
            return offer.market_hash_name or self.name_for("steam")
        return self.name_for("steam")

    def matches_offer(self, offer: "MarketOffer") -> bool:
        attrs = parse_market_hash_name(offer.market_hash_name)
        if self.weapon_filters and (
            attrs.weapon is None or attrs.weapon.lower() not in self.weapon_filters
        ):
            return False
        if self.wear_filters and (
            attrs.wear is None or attrs.wear not in self.wear_filters
        ):
            return False
        if self.stattrak is not None and attrs.stattrak != self.stattrak:
            return False
        if self.souvenir is not None and attrs.souvenir != self.souvenir:
            return False
        name_lower = attrs.original_name.lower()
        if self.include_keywords and not all(k in name_lower for k in self.include_keywords):
            return False
        if any(k in name_lower for k in self.exclude_keywords):
            return False
        if self.exact_name and not self.dynamic:
            expected = self.name_for("steam").lower()
            return not expected or name_lower == expected
        return True


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
    discount: float
    score: float

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
        self.currency_id = resolve_steam_currency_id(settings.currency)
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
    supports_dynamic: bool = False

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
        if self.config.get("auto_cookies", True):
            self._bootstrap_cookies()

    def request(self, method: str, url: str, **kwargs) -> Response:
        timeout = kwargs.pop("timeout", self.settings.request_timeout)
        try:
            return self.scraper.request(method, url, timeout=timeout, **kwargs)
        except RequestException as exc:
            raise RuntimeError(f"{self.name}: сетевой сбой — {exc}") from exc

    def _bootstrap_cookies(self) -> None:
        targets = self.config.get("bootstrap_urls")
        if not targets:
            targets = [
                self.config.get("page_url"),
                self.config.get("api_url"),
                self.config.get("base_url"),
            ]
        for target in targets:
            if not target:
                continue
            try:
                tokens, user_agent = self.scraper.get_tokens(target)
            except Exception as exc:
                logging.debug("%s: auto-cookie update failed for %s: %s", self.name, target, exc)
                continue
            if tokens:
                self.scraper.cookies.update(tokens)
            if user_agent:
                self.scraper.headers["User-Agent"] = user_agent
            logging.debug("%s: получены Cloudflare cookie автоматически", self.name)
            return
        logging.debug("%s: не удалось автоматически получить Cloudflare cookie", self.name)


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
            "market_hash_name": query.search_for(self.name),
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
        target = query.search_for(self.name).lower()
        for entry in items:
            raw_name = entry.get("market_hash_name") or entry.get("market_name") or ""
            name = raw_name.lower()
            if query.exact_name and name and name != target:
                continue
            if not query.exact_name and target and target not in name:
                continue
            price = coerce_float(entry.get("min_price") or entry.get("price"))
            if price is None:
                continue
            adjusted = self.adjust_price(price)
            offer_id = str(entry.get("item_id") or entry.get("market_hash_name"))
            url = self.config.get("item_url_template", "https://skinport.com/item/{name}").format(
                name=urllib.parse.quote(query.search_for(self.name))
            )
            offers.append(
                MarketOffer(
                    self.name,
                    offer_id,
                    raw_name or query.market_hash_name,
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
                    entry.get("market_hash_name") or query.market_hash_name,
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


class SteamCommunityFetcher(HTTPFetcher):
    name = "steamcommunity.com"
    LISTING_URL = "https://steamcommunity.com/market/listings/730/{name}/render"

    def __init__(self, settings: Settings, config: Optional[Dict[str, Any]] = None):
        super().__init__(settings, config)
        self.currency_id = resolve_steam_currency_id(self.config.get("currency", settings.currency))

    def fetch_offers(self, query: ItemQuery) -> List[MarketOffer]:
        market_name = query.name_for(self.name)
        if not market_name:
            return []
        params = {
            "start": self.config.get("start", 0),
            "count": self.config.get("count", 20),
            "currency": self.currency_id,
            "format": "json",
        }
        url = self.LISTING_URL.format(name=urllib.parse.quote(market_name))
        response = self.session.get(url, params=params, timeout=self.settings.request_timeout)
        if response.status_code == 429:
            raise RuntimeError("steamcommunity.com: получен 429 (превышен лимит запросов)")
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("steamcommunity.com: некорректный JSON") from exc
        if not data.get("success", True):
            return []
        listinginfo = data.get("listinginfo") or {}
        offers: List[MarketOffer] = []
        for listing_id, info in listinginfo.items():
            base_cents = coerce_int(info.get("converted_price_per_unit") or info.get("converted_price"))
            fee_cents = coerce_int(info.get("converted_fee_per_unit") or info.get("converted_fee") or 0)
            if base_cents is None:
                continue
            total_cents = base_cents + (fee_cents or 0)
            price = self.adjust_price(total_cents / 100.0)
            asset = info.get("asset") or {}
            raw_name = (
                asset.get("market_hash_name")
                or asset.get("market_name")
                or asset.get("name")
                or query.market_hash_name
            )
            offers.append(
                MarketOffer(
                    self.name,
                    str(listing_id),
                    raw_name,
                    price,
                    self.config.get(
                        "item_url_template",
                        "https://steamcommunity.com/market/listings/730/{name}",
                    ).format(name=urllib.parse.quote(market_name)),
                    quantity=coerce_int(info.get("quantity")),
                    extra={
                        "original_amount": info.get("original_amount"),
                        "asset": info.get("asset"),
                    },
                    raw=info,
                )
            )
        return offers


class LisSkinsFetcher(CloudflareFetcher):
    name = "lis-skins.com"
    supports_dynamic = True

    def fetch_offers(self, query: ItemQuery) -> List[MarketOffer]:
        page_url = self.config.get("page_url", "https://lis-skins.com/market/csgo")
        params = {"search": query.search_for(self.name)}
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
        seen: Set[str] = set()
        for entry in items:
            raw_name = entry.get("market_hash_name") or entry.get("fullName") or entry.get("name") or ""
            name = raw_name.lower()
            if not query.matches_search(name, self.name):
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
            ).format(name=urllib.parse.quote(query.search_for(self.name)))
            offers.append(
                MarketOffer(
                    self.name,
                    offer_id,
                    raw_name or query.market_hash_name,
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
    supports_dynamic = True

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
            "search": query.search_for(self.name),
        }
        response = self.request("GET", url, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"cs.money API вернул {response.status_code}")
        data = response.json()
        items = data.get("items") or data.get("sellOrders") or data.get("data") or []
        return self._parse_items(items, query)

    def _fetch_via_page(self, query: ItemQuery) -> List[MarketOffer]:
        page_url = self.config.get("page_url", "https://cs.money/csgo/trade")
        params = {"search": query.search_for(self.name)}
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
        seen: Set[str] = set()
        for entry in items:
            raw_name = entry.get("market_hash_name") or entry.get("fullName") or entry.get("name") or ""
            name = raw_name.lower()
            if not query.matches_search(name, self.name):
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
                    raw_name or query.market_hash_name,
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
        self._dynamic_skip_notified: Set[str] = set()

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
        missing_quotes: Set[str] = set()
        for query in queries:
            base_steam_name: Optional[str] = None
            if not query.dynamic:
                base_steam_name = query.name_for("steam")
                if base_steam_name and base_steam_name not in quotes:
                    steam_quote = self.steam_client.fetch(base_steam_name)
                    if steam_quote is None:
                        if base_steam_name not in missing_quotes:
                            errors.append(f"Steam: нет цены для {base_steam_name}")
                            missing_quotes.add(base_steam_name)
                        continue
                    quotes[base_steam_name] = steam_quote
            for fetcher in self.fetchers:
                if query.watch_markets and fetcher.name not in query.watch_markets:
                    continue
                if query.dynamic and not getattr(fetcher, "supports_dynamic", False):
                    if fetcher.name not in self._dynamic_skip_notified:
                        errors.append(f"{fetcher.name}: динамические фильтры не поддерживаются, пропуск")
                        self._dynamic_skip_notified.add(fetcher.name)
                    continue
                try:
                    offers = fetcher.fetch_offers(query)
                except Exception as exc:
                    logging.exception("Ошибка в %s для %s", fetcher.name, query.market_hash_name)
                    errors.append(f"{fetcher.name}: {exc}")
                    continue
                for offer in offers:
                    if not query.matches_offer(offer):
                        continue
                    steam_name = query.resolve_steam_name(offer)
                    if not steam_name:
                        continue
                    steam_quote = quotes.get(steam_name)
                    if steam_quote is None:
                        steam_quote = self.steam_client.fetch(steam_name)
                        if steam_quote is None:
                            if steam_name not in missing_quotes:
                                errors.append(f"Steam: нет цены для {steam_name}")
                                missing_quotes.add(steam_name)
                            continue
                        quotes[steam_name] = steam_quote
                    deal = self.evaluate_offer(query, steam_quote, offer)
                    if deal:
                        key = f"{deal.market}:{deal.offer.offer_id}"
                        self.last_seen[key] = time.time()
                        deals.append(deal)
        deals.sort(key=lambda d: (d.score, d.roi, d.profit), reverse=True)
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
        discount = (target_price - offer.price) / target_price if target_price else 0.0
        weights = self.settings.deal_score_weights
        profit_norm = profit / target_price if target_price else 0.0
        score = (
            weights.get("roi", 0.0) * roi
            + weights.get("profit", 0.0) * profit_norm
            + weights.get("discount", 0.0) * discount
        )
        return Deal(query, offer, steam_quote, target_price, profit, roi, discount, score)

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
        table.add_column("Скидка", justify="right")
        table.add_column("Счёт", justify="right")
        table.add_column("Ссылка", overflow="fold")
        max_deals = self.settings.max_displayed_deals
        display_deals = deals[:max_deals] if max_deals else deals
        if not display_deals:
            table.add_row("—", "Нет выгодных предложений", "—", "—", "—", "—", "—", "—", "—")
        for deal in display_deals:
            table.add_row(
                deal.market,
                deal.offer.market_hash_name or deal.item.market_hash_name,
                format_money(deal.offer.price, self.settings.currency),
                format_money(deal.target_sell_price, self.settings.currency),
                format_money(deal.profit, self.settings.currency, signed=True),
                f"{deal.roi * 100:+.2f}%",
                f"{deal.discount * 100:+.2f}%",
                f"{deal.score:.3f}",
                deal.offer.url,
            )
        console.print(table)
        if max_deals and len(deals) > len(display_deals):
            console.print(
                Panel(
                    f"Показаны топ {len(display_deals)} из {len(deals)} предложений по счёту.",
                    style="cyan",
                )
            )
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
    weights_cfg = data.get("deal_score_weights") or {}
    score_weights = dict(DEFAULT_SCORE_WEIGHTS)
    for key, value in weights_cfg.items():
        try:
            score_weights[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"deal_score_weights.{key} должно быть числом") from exc
    max_deals_raw = data.get("max_displayed_deals")
    max_displayed_deals = (
        int(max_deals_raw)
        if max_deals_raw not in (None, "", "none", "null")
        else None
    )
    if max_displayed_deals is not None and max_displayed_deals <= 0:
        max_displayed_deals = None
    settings = Settings(
        refresh_interval=int(data.get("refresh_interval", 60)),
        currency=(data.get("currency") or "USD").upper(),
        steam_fee=float(data.get("steam_fee", 0.13)),
        default_min_roi=float(data.get("default_min_roi", 0.05)),
        default_min_profit=float(data.get("default_min_profit", 1.0)),
        request_timeout=float(data.get("request_timeout", 25.0)),
        verify_tls=bool(data.get("verify_tls", True)),
        proxies=data.get("proxies") or {},
        deal_score_weights=score_weights,
        max_displayed_deals=max_displayed_deals,
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
    filters_cfg = item.get("filters") or {}

    def get_filter_value(*keys: str) -> Any:
        for key in keys:
            if key in filters_cfg:
                return filters_cfg[key]
        for key in keys:
            if key in item:
                return item[key]
        return None

    base_name = item.get("market_hash_name") or item.get("name")
    search_term = item.get("search") or filters_cfg.get("search")
    min_price = float(item.get("min_price", 0.0) or 0.0)
    max_price_raw = item.get("max_price")
    max_price = float(max_price_raw) if max_price_raw not in (None, "", "none") else None
    min_roi = float(item.get("min_roi", settings.default_min_roi))
    min_profit = float(item.get("min_profit", settings.default_min_profit))
    markets = item.get("markets")
    watch_markets = {m.strip() for m in markets} if isinstance(markets, (list, set, tuple)) else None
    aliases = item.get("aliases") or {}
    wear_values = to_string_list(get_filter_value("wears", "wear", "quality", "qualities"))
    wear_filters = {normalize_wear(v) for v in wear_values if normalize_wear(v)}
    weapon_values_raw = to_string_list(get_filter_value("weapons", "weapon"))
    weapon_filters: Set[str] = set()
    for value in weapon_values_raw:
        normalized = normalize_weapon(value)
        if normalized:
            weapon_filters.add(normalized.lower())
    try:
        stattrak = parse_optional_bool(get_filter_value("stattrak", "stat_trak", "is_stattrak"))
        souvenir = parse_optional_bool(get_filter_value("souvenir", "souvenir_only"))
    except ConfigError as exc:
        raise ConfigError(f"{exc} для предмета {base_name or item}") from exc
    include_keywords = {kw.lower() for kw in to_string_list(get_filter_value("contains", "include", "keywords"))}
    exclude_keywords = {kw.lower() for kw in to_string_list(get_filter_value("exclude", "excludes", "without"))}
    if not search_term:
        search_term = base_name
    if not search_term and weapon_values_raw:
        search_term = weapon_values_raw[0]
    dynamic = bool(item.get("dynamic", False))
    if not base_name:
        dynamic = True
    if dynamic and search_term is None:
        raise ConfigError("Для динамического запроса требуется указать search или weapon/filters.")
    if base_name is None and not dynamic:
        raise ConfigError("Каждый предмет должен иметь market_hash_name или быть динамическим фильтром.")
    if search_term is None:
        search_term = base_name or ""
    label = item.get("label") or base_name
    if not label:
        label_parts: List[str] = []
        if stattrak is True:
            label_parts.append("StatTrak")
        elif stattrak is False:
            label_parts.append("Non-StatTrak")
        if souvenir:
            label_parts.append("Souvenir")
        if weapon_values_raw:
            label_parts.append(", ".join(weapon_values_raw))
        if wear_filters:
            label_parts.append("/".join(sorted(wear_filters)))
        if not label_parts and search_term:
            label_parts.append(str(search_term))
        label = " | ".join(label_parts) if label_parts else "Запрос"
    if "exact_name" in item:
        exact_name = bool(item.get("exact_name"))
    else:
        exact_name = not dynamic
    steam_name = item.get("steam_name")
    if steam_name is None and not dynamic:
        steam_name = aliases.get("steam", base_name)
    return ItemQuery(
        market_hash_name=label,
        min_price=min_price,
        max_price=max_price,
        min_roi=min_roi,
        min_profit=min_profit,
        watch_markets=watch_markets,
        aliases=aliases,
        search_term=search_term,
        steam_name=steam_name,
        exact_name=exact_name,
        dynamic=dynamic,
        wear_filters={w for w in wear_filters if w},
        weapon_filters={w for w in weapon_filters if w},
        stattrak=stattrak,
        souvenir=souvenir,
        include_keywords=include_keywords,
        exclude_keywords=exclude_keywords,
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
    def pop_list(*keys: str) -> List[str]:
        for key in keys:
            if key in parts:
                return to_string_list(parts.pop(key))
        return []

    def pop_bool(key: str) -> Optional[bool]:
        if key not in parts:
            return None
        value = parts.pop(key)
        try:
            return parse_optional_bool(value)
        except ConfigError as exc:
            raise ConfigError(f"--item {expr}: {exc}") from exc

    search_term = parts.pop("search", None)
    wear_filters = {normalize_wear(v) for v in pop_list("wears", "wear", "quality") if normalize_wear(v)}
    weapon_values = pop_list("weapons", "weapon")
    weapon_filters: Set[str] = set()
    for value in weapon_values:
        normalized = normalize_weapon(value)
        if normalized:
            weapon_filters.add(normalized.lower())
    stattrak = pop_bool("stattrak")
    souvenir = pop_bool("souvenir")
    include_keywords = {kw.lower() for kw in pop_list("contains", "include")}
    exclude_keywords = {kw.lower() for kw in pop_list("exclude", "excludes")}
    dynamic_flag = pop_bool("dynamic")
    exact_flag = pop_bool("exact_name")
    if parts:
        raise ConfigError(f"Неизвестные параметры у --item: {', '.join(parts.keys())}")
    dynamic = bool(dynamic_flag) if dynamic_flag is not None else False
    if name is None and not dynamic and not search_term:
        raise ConfigError("--item должен содержать name=... или быть динамическим (--item dynamic=true;search=...)")
    if search_term is None:
        search_term = name
    label = name or search_term or (", ".join(weapon_values) if weapon_values else "CLI Query")
    if exact_flag is None:
        exact_name = not dynamic
    else:
        exact_name = bool(exact_flag)
    steam_name = name if not dynamic else None
    return ItemQuery(
        market_hash_name=label,
        min_price=min_price,
        max_price=max_price,
        min_roi=min_roi,
        min_profit=min_profit,
        watch_markets=watch_markets,
        aliases=aliases,
        search_term=search_term,
        steam_name=steam_name,
        exact_name=exact_name,
        dynamic=dynamic,
        wear_filters=wear_filters,
        weapon_filters={w for w in weapon_filters if w},
        stattrak=stattrak,
        souvenir=souvenir,
        include_keywords=include_keywords,
        exclude_keywords=exclude_keywords,
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
        (SteamCommunityFetcher, "steam_community"),
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
