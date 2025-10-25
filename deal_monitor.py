#!/usr/bin/env python3
"""Мониторинг выгодных офферов CS:GO across cs.money, lis-skins.com, market.csgo.com, skinport.com и Steam."""
from __future__ import annotations

import argparse
import json
import html
import logging
import os
import sys
import time
import urllib.parse
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Callable

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
        self._browser_cfg = {
            "browser": self.config.get("browser", "firefox"),
            "platform": self.config.get("platform", "windows"),
            "mobile": bool(self.config.get("mobile", False)),
        }
        headers = {"User-Agent": self.config.get("user_agent", DEFAULT_USER_AGENT)}
        headers.update(self.config.get("headers") or {})
        self._base_headers = dict(headers)
        cookies = self.config.get("cookies") or {}
        self._base_cookies = dict(cookies)
        self._bootstrap_attempted = False
        self._recreate_scraper()
        if self.config.get("auto_cookies", True):
            self._bootstrap_cookies()

    def request(self, method: str, url: str, **kwargs) -> Response:
        timeout = kwargs.pop("timeout", self.settings.request_timeout)
        auto_cookies = self.config.get("auto_cookies", True)
        attempts = 2 if auto_cookies else 1
        for attempt in range(attempts):
            try:
                response = self.scraper.request(method, url, timeout=timeout, **kwargs)
            except RequestException as exc:
                if auto_cookies and attempt + 1 < attempts:
                    logging.debug("%s: повторная попытка после сетевой ошибки: %s", self.name, exc)
                    self._handle_retry()
                    continue
                raise RuntimeError(f"{self.name}: сетевой сбой — {exc}") from exc
            if response.status_code in {401, 403} and auto_cookies and attempt + 1 < attempts:
                logging.debug("%s: получен статус %s, пробуем обновить cookie", self.name, response.status_code)
                self._handle_retry()
                continue
            return response
        raise RuntimeError(f"{self.name}: не удалось выполнить запрос к {url}")

    def _handle_retry(self) -> None:
        self._recreate_scraper()
        refreshed = self._bootstrap_cookies(force=True)
        if not refreshed:
            logging.debug("%s: повторное получение Cloudflare cookie не удалось", self.name)

    def _recreate_scraper(self) -> None:
        self.scraper = cloudscraper.create_scraper(browser=self._browser_cfg)
        self.scraper.headers.update(self._base_headers)
        if self._base_cookies:
            self.scraper.cookies.update(self._base_cookies)

    def _bootstrap_cookies(self, force: bool = False) -> bool:
        if self._bootstrap_attempted and not force:
            return False
        self._bootstrap_attempted = True
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
            return True
        logging.debug("%s: не удалось автоматически получить Cloudflare cookie", self.name)
        return False


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


_SCRIPT_TAG_RE = re.compile(
    r"<script(?P<attrs>[^>]*)>(?P<content>.*?)</script>", re.S | re.IGNORECASE
)


def extract_embedded_json(html_source: str, variable_names: Sequence[str]) -> List[Any]:
    """Извлекает словари JSON из тегов <script> и присваиваний window.*."""

    results: List[Any] = []
    normalized_targets = {name.lower() for name in variable_names}

    for match in _SCRIPT_TAG_RE.finditer(html_source):
        attrs = match.group("attrs") or ""
        content = html.unescape(match.group("content") or "").strip()
        if not content:
            continue
        script_type = None
        script_id = None
        type_match = re.search(r'type="([^"]+)"', attrs, re.I)
        if type_match:
            script_type = type_match.group(1).lower()
        id_match = re.search(r'id="([^"]+)"', attrs, re.I)
        if id_match:
            script_id = id_match.group(1).lower()
        if script_type == "application/json" or (
            script_id and script_id in normalized_targets
        ):
            try:
                results.append(json.loads(content))
                continue
            except json.JSONDecodeError:
                pass
        for target in variable_names:
            if target.lower() not in content.lower():
                continue
            json_candidate = _extract_json_from_assignment(content, target)
            if json_candidate is not None:
                results.append(json_candidate)
                break

    # В некоторых случаях присваивание может находиться вне тегов <script> (например, минимизированный HTML).
    for target in variable_names:
        candidate = _extract_json_from_assignment(html_source, target)
        if candidate is not None:
            results.append(candidate)
    return results


def _extract_json_from_assignment(source: str, variable: str) -> Optional[Any]:
    assignment_pattern = re.compile(
        rf"(?:window\.)?{re.escape(variable)}\s*=\s*(\{{.*?\}});",
        re.S,
    )
    parse_pattern = re.compile(
        rf"(?:window\.)?{re.escape(variable)}\s*=\s*JSON\.parse\((\".*?\")\)",
        re.S,
    )

    assignment_match = assignment_pattern.search(source)
    if assignment_match:
        json_text = assignment_match.group(1)
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass
    parse_match = parse_pattern.search(source)
    if parse_match:
        try:
            encoded = json.loads(parse_match.group(1))
        except json.JSONDecodeError:
            return None
        try:
            return json.loads(encoded)
        except json.JSONDecodeError:
            return None
    return None


_COMMON_LIST_KEYS = {
    "items",
    "listings",
    "offers",
    "sellOrders",
    "lots",
    "results",
    "entries",
    "nodes",
    "data",
    "edges",
}

_DICT_PRICE_KEYS = ("amount", "value", "min", "max", "price")
_DIRECT_PRICE_KEYS = (
    "price",
    "price_usd",
    "priceUsd",
    "priceUSD",
    "priceFloat",
    "price_float",
    "usdPrice",
    "min_price",
    "max_price",
    "mean_price",
    "current_price",
    "median_price",
    "suggested_price",
    "sale_price",
    "lowest_price",
    "best_price",
)
_NESTED_PRICE_KEYS = ("pricing", "priceInfo", "price_info", "buyOrder")

_NAME_KEYS = (
    "market_hash_name",
    "marketHashName",
    "market_name",
    "marketName",
    "fullName",
    "full_name",
    "name",
    "title",
)

_ID_KEYS = (
    "id",
    "offer_id",
    "offerId",
    "listingid",
    "listingId",
    "assetid",
    "steamId",
    "hash",
    "slug",
    "id64",
    "item_id",
)

_QUANTITY_KEYS = ("quantity", "count", "amount", "available", "stock", "total")

_URL_KEYS = (
    "url",
    "link",
    "href",
    "web_url",
    "item_page",
    "itemPage",
    "page_url",
    "pageUrl",
)

_EXTRA_FLOAT_KEYS = ("float", "floatValue", "wear", "float_value")
_EXTRA_PATTERN_KEYS = ("phase", "pattern", "phaseName")
_EXTRA_STICKERS_KEYS = ("stickers", "decals")


def extract_offer_price(entry: Dict[str, Any]) -> Optional[float]:
    candidate: Any = None
    price_value = entry.get("price")
    if isinstance(price_value, dict):
        for key in _DICT_PRICE_KEYS:
            if price_value.get(key) is not None:
                candidate = price_value[key]
                break
    elif price_value is not None:
        candidate = price_value
    if candidate is None:
        for key in _DIRECT_PRICE_KEYS:
            if entry.get(key) is not None:
                candidate = entry[key]
                break
    if candidate is None:
        for nested_key in _NESTED_PRICE_KEYS:
            nested = entry.get(nested_key)
            if isinstance(nested, dict):
                nested_value = extract_offer_price(nested)
                if nested_value is not None:
                    candidate = nested_value
                    break
    if candidate is None and isinstance(entry.get("item"), dict):
        return extract_offer_price(entry["item"])
    if candidate is None:
        return None
    value = coerce_float(candidate)
    if value is None and isinstance(candidate, (int, float)):
        value = float(candidate)
    return value


def extract_offer_name(entry: Dict[str, Any]) -> Optional[str]:
    for key in _NAME_KEYS:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value
    item = entry.get("item")
    if isinstance(item, dict):
        return extract_offer_name(item)
    return None


def extract_offer_id(entry: Dict[str, Any], fallback: Optional[str] = None) -> str:
    for key in _ID_KEYS:
        value = entry.get(key)
        if value:
            return str(value)
    item = entry.get("item")
    if isinstance(item, dict):
        nested = extract_offer_id(item, fallback)
        if nested:
            return nested
    return str(fallback or "0")


def extract_offer_quantity(entry: Dict[str, Any]) -> Optional[int]:
    for key in _QUANTITY_KEYS:
        value = entry.get(key)
        if value is not None:
            quantity = coerce_int(value)
            if quantity is not None:
                return quantity
    item = entry.get("item")
    if isinstance(item, dict):
        return extract_offer_quantity(item)
    return None


def normalize_offer_url(
    candidate: Optional[str], base_url: Optional[str], default_url: str
) -> str:
    if not candidate or not isinstance(candidate, str):
        return default_url
    url = candidate.strip()
    if not url:
        return default_url
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if base_url:
        return urllib.parse.urljoin(base_url, url)
    return default_url


def extract_offer_url(
    entry: Dict[str, Any], base_url: Optional[str], default_url: str
) -> str:
    for key in _URL_KEYS:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_offer_url(value, base_url, default_url)
    item = entry.get("item")
    if isinstance(item, dict):
        return extract_offer_url(item, base_url, default_url)
    return default_url


def extract_offer_extra(entry: Dict[str, Any]) -> Dict[str, Any]:
    item = entry.get("item") if isinstance(entry.get("item"), dict) else None
    float_value = None
    for key in _EXTRA_FLOAT_KEYS:
        if entry.get(key) is not None:
            float_value = entry[key]
            break
    if float_value is None and item:
        for key in _EXTRA_FLOAT_KEYS:
            if item.get(key) is not None:
                float_value = item[key]
                break
    pattern = None
    for key in _EXTRA_PATTERN_KEYS:
        if entry.get(key) is not None:
            pattern = entry[key]
            break
    if pattern is None and item:
        for key in _EXTRA_PATTERN_KEYS:
            if item.get(key) is not None:
                pattern = item[key]
                break
    stickers = None
    for key in _EXTRA_STICKERS_KEYS:
        if entry.get(key) is not None:
            stickers = entry[key]
            break
    if stickers is None and item:
        for key in _EXTRA_STICKERS_KEYS:
            if item.get(key) is not None:
                stickers = item[key]
                break
    extra: Dict[str, Any] = {}
    if float_value is not None:
        extra["float"] = float_value
    if pattern is not None:
        extra["pattern"] = pattern
    if stickers is not None:
        extra["stickers"] = stickers
    return extra


def build_market_offers(
    market: str,
    entries: Iterable[Dict[str, Any]],
    query: ItemQuery,
    adjust_price: Callable[[float], float],
    default_url: str,
    base_url: Optional[str] = None,
    expected_name: Optional[str] = None,
) -> List[MarketOffer]:
    offers: List[MarketOffer] = []
    seen: Set[str] = set()
    expected_lower = expected_name.lower() if expected_name else None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_name = extract_offer_name(entry) or expected_name or query.market_hash_name
        if not raw_name:
            continue
        name_lower = raw_name.lower()
        if expected_lower and name_lower != expected_lower and expected_lower not in name_lower:
            continue
        if not query.matches_search(name_lower, market):
            continue
        price_value = extract_offer_price(entry)
        if price_value is None:
            continue
        adjusted = adjust_price(price_value)
        offer_id = extract_offer_id(entry, raw_name)
        if offer_id in seen:
            continue
        seen.add(offer_id)
        offer_url = extract_offer_url(entry, base_url, default_url)
        quantity = extract_offer_quantity(entry)
        extra = extract_offer_extra(entry)
        offers.append(
            MarketOffer(
                market,
                offer_id,
                raw_name or query.market_hash_name,
                adjusted,
                offer_url,
                quantity=quantity,
                extra=extra,
                raw=entry,
            )
        )
    return offers


class SkinportFetcher(HTTPFetcher):
    name = "skinport"
    SEARCH_URL_TEMPLATE = "https://skinport.com/market/csgo?search={name}"

    _SCRIPT_NAMES = (
        "__NUXT__",
        "__INITIAL_STATE__",
        "__NEXT_DATA__",
        "_post",
        "__APOLLO_STATE__",
        "__APOLLO_CACHE__",
    )

    def __init__(self, settings: Settings, config: Optional[Dict[str, Any]] = None):
        super().__init__(settings, config)
        self.base_url = self.config.get("base_url", "https://skinport.com")
        self.search_url_template = self.config.get(
            "search_url_template", self.SEARCH_URL_TEMPLATE
        )
        self.item_url_template = self.config.get(
            "item_url_template", self.search_url_template
        )

    def fetch_offers(self, query: ItemQuery) -> List[MarketOffer]:
        search_term = query.search_for(self.name)
        if not search_term:
            return []
        url = self.search_url_template.format(name=urllib.parse.quote(search_term))
        response = self.session.get(url, timeout=self.settings.request_timeout)
        response.raise_for_status()
        payloads = extract_embedded_json(response.text, self._SCRIPT_NAMES)
        entries: List[Dict[str, Any]] = []
        for payload in payloads:
            for candidate in iter_items_by_keys(payload, _COMMON_LIST_KEYS):
                if isinstance(candidate, list):
                    entries.extend(candidate)
        default_url = self.item_url_template.format(
            name=urllib.parse.quote(search_term)
        )
        offers = build_market_offers(
            self.name,
            entries,
            query,
            self.adjust_price,
            default_url,
            base_url=self.base_url,
            expected_name=query.name_for(self.name),
        )
        return offers

class CSGOMarketFetcher(HTTPFetcher):
    name = "market.csgo.com"
    ITEM_URL_TEMPLATE = "https://market.csgo.com/item/{name}"
    SEARCH_URL_TEMPLATE = "https://market.csgo.com/?search={name}"
    _SCRIPT_RE = re.compile(r"<script[^>]*>(.*?)</script>", re.S | re.IGNORECASE)

    def __init__(self, settings: Settings, config: Optional[Dict[str, Any]] = None):
        super().__init__(settings, config)
        self.config.setdefault("price_multiplier", 1.0)
        self.item_url_template = self.config.get("item_url_template", self.ITEM_URL_TEMPLATE)
        self.search_url_template = self.config.get(
            "search_url_template", self.SEARCH_URL_TEMPLATE
        )

    def fetch_offers(self, query: ItemQuery) -> List[MarketOffer]:
        target_name = query.name_for(self.name)
        if not target_name:
            return []
        url = self.item_url_template.format(name=urllib.parse.quote(target_name))
        response = self.session.get(url, timeout=self.settings.request_timeout)
        response.raise_for_status()
        offers = self._parse_offers_from_html(response.text, query, target_name)
        if offers:
            return offers
        # Попытка использовать страницу поиска как запасной вариант.
        search_url = self.search_url_template.format(
            name=urllib.parse.quote(query.search_for(self.name))
        )
        if search_url != url:
            fallback = self.session.get(search_url, timeout=self.settings.request_timeout)
            if fallback.status_code == 200:
                parsed = self._parse_offers_from_html(fallback.text, query, target_name)
                if parsed:
                    return parsed
        return []

    def _parse_offers_from_html(
        self, html: str, query: ItemQuery, expected_name: str
    ) -> List[MarketOffer]:
        try:
            payload = self._extract_json_payload(html)
        except RuntimeError:
            return []
        offer_nodes = self._collect_offer_nodes(payload)
        if not offer_nodes:
            return []
        default_url = self.item_url_template.format(
            name=urllib.parse.quote(expected_name)
        )
        base_url = self.config.get("base_url")
        return build_market_offers(
            self.name,
            offer_nodes,
            query,
            self.adjust_price,
            default_url,
            base_url=base_url,
            expected_name=expected_name,
        )

    def _extract_json_payload(self, html: str) -> Dict[str, Any]:
        for match in self._SCRIPT_RE.finditer(html):
            content = match.group(1).strip()
            if not content or not content.startswith("{"):
                continue
            if "apollo.cache.state" not in content and "apollo.store.state" not in content:
                continue
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                continue
        raise RuntimeError("market.csgo.com: не удалось извлечь JSON данные из HTML")

    def _collect_offer_nodes(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        nodes: List[Dict[str, Any]] = []
        for key in ("apollo.cache.state", "apollo.store.state"):
            source = payload.get(key)
            if not isinstance(source, dict):
                continue
            nodes.extend(self._dfs_collect_offer_nodes(source))
        return nodes

    def _dfs_collect_offer_nodes(self, value: Any) -> List[Dict[str, Any]]:
        found: List[Dict[str, Any]] = []
        stack: List[Any] = [value]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                edges = current.get("edges")
                if isinstance(edges, list):
                    for edge in edges:
                        if not isinstance(edge, dict):
                            continue
                        node = edge.get("node")
                        if isinstance(node, dict) and extract_offer_price(node) is not None:
                            found.append(node)
                stack.extend(current.values())
            elif isinstance(current, list):
                stack.extend(current)
        return found


class SteamCommunityFetcher(HTTPFetcher):
    name = "steamcommunity.com"
    LISTING_PAGE_URL = "https://steamcommunity.com/market/listings/730/{name}"
    _LISTING_INFO_RE = re.compile(r"var\s+g_rgListingInfo\s*=\s*(\{.*?\});", re.S)
    _ASSETS_RE = re.compile(r"g_rgAssets\s*=\s*(\{.*?\});", re.S)

    def __init__(self, settings: Settings, config: Optional[Dict[str, Any]] = None):
        super().__init__(settings, config)
        self.currency_id = resolve_steam_currency_id(
            self.config.get("currency", settings.currency)
        )
        self.item_url_template = self.config.get(
            "item_url_template",
            "https://steamcommunity.com/market/listings/730/{name}",
        )

    def fetch_offers(self, query: ItemQuery) -> List[MarketOffer]:
        market_name = query.name_for(self.name)
        if not market_name:
            return []
        params = {"currency": self.currency_id}
        url = self.LISTING_PAGE_URL.format(name=urllib.parse.quote(market_name))
        response = self.session.get(url, params=params, timeout=self.settings.request_timeout)
        if response.status_code == 429:
            raise RuntimeError("steamcommunity.com: получен 429 (превышен лимит запросов)")
        response.raise_for_status()
        html = response.text
        listing_info = self._extract_json_block(self._LISTING_INFO_RE, html)
        if not listing_info:
            return []
        assets = self._extract_json_block(self._ASSETS_RE, html)
        asset_lookup = self._build_asset_lookup(assets)
        default_url = self.item_url_template.format(
            name=urllib.parse.quote(market_name)
        )
        offers: List[MarketOffer] = []
        for listing_id, info in listing_info.items():
            if not isinstance(info, dict):
                continue
            base_cents = coerce_int(
                info.get("converted_price_per_unit") or info.get("converted_price")
            )
            if base_cents is None:
                continue
            fee_cents = coerce_int(
                info.get("converted_fee_per_unit") or info.get("converted_fee") or 0
            ) or 0
            total_cents = base_cents + fee_cents
            price = self.adjust_price(total_cents / 100.0)
            asset = self._resolve_asset(info.get("asset"), asset_lookup)
            name = (
                (asset or {}).get("market_hash_name")
                or (asset or {}).get("market_name")
                or (asset or {}).get("name")
                or query.market_hash_name
            )
            offers.append(
                MarketOffer(
                    self.name,
                    str(listing_id),
                    name,
                    price,
                    default_url,
                    quantity=coerce_int(info.get("quantity")),
                    extra={
                        "asset": asset,
                        "converted_price": base_cents,
                        "converted_fee": fee_cents,
                    },
                    raw=info,
                )
            )
        return offers

    def _extract_json_block(self, pattern: re.Pattern, html: str) -> Dict[str, Any]:
        match = pattern.search(html)
        if not match:
            return {}
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return {}

    def _build_asset_lookup(
        self, assets: Dict[str, Any]
    ) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
        lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        if not isinstance(assets, dict):
            return lookup
        for appid, contexts in assets.items():
            if not isinstance(contexts, dict):
                continue
            for contextid, items in contexts.items():
                if not isinstance(items, dict):
                    continue
                for assetid, data in items.items():
                    if not isinstance(data, dict):
                        continue
                    key = (
                        str(data.get("appid") or appid),
                        str(data.get("contextid") or contextid),
                        str(data.get("id") or assetid),
                    )
                    lookup[key] = data
        return lookup

    def _resolve_asset(
        self,
        asset_info: Any,
        lookup: Dict[Tuple[str, str, str], Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(asset_info, dict):
            return None
        appid = str(asset_info.get("appid") or asset_info.get("app_id") or "730")
        contextid = str(asset_info.get("contextid") or asset_info.get("context_id") or "2")
        assetid = str(asset_info.get("id") or asset_info.get("assetid") or "")
        if not assetid:
            return asset_info
        return lookup.get((appid, contextid, assetid), asset_info)

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
        payloads = extract_embedded_json(
            html,
            ("__NEXT_DATA__", "__NUXT__", "__APOLLO_STATE__", "_post"),
        )
        if not payloads:
            raise RuntimeError(
                "lis-skins.com: не найден блок данных (обновите headers/cookies)."
            )
        entries: List[Dict[str, Any]] = []
        for payload in payloads:
            for candidate in iter_items_by_keys(payload, _COMMON_LIST_KEYS):
                if isinstance(candidate, list):
                    entries.extend(candidate)
        default_url = self.config.get(
            "item_url_template",
            "https://lis-skins.com/market/csgo?search={name}",
        ).format(name=urllib.parse.quote(query.search_for(self.name)))
        base_url = self.config.get("base_url", "https://lis-skins.com")
        return build_market_offers(
            self.name,
            entries,
            query,
            self.adjust_price,
            default_url,
            base_url=base_url,
            expected_name=query.name_for(self.name),
        )

class CSMoneyFetcher(CloudflareFetcher):
    name = "cs.money"
    supports_dynamic = True

    def fetch_offers(self, query: ItemQuery) -> List[MarketOffer]:
        return self._fetch_via_page(query)

    def _fetch_via_page(self, query: ItemQuery) -> List[MarketOffer]:
        page_url = self.config.get("page_url", "https://cs.money/market/csgo")
        params = {"search": query.search_for(self.name)}
        response = self.request("GET", page_url, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"cs.money страница вернула {response.status_code}")
        html = response.text
        payloads = extract_embedded_json(
            html,
            ("__NEXT_DATA__", "__NUXT__", "__APOLLO_STATE__", "_post"),
        )
        if not payloads:
            raise RuntimeError(
                "cs.money: не удалось найти данные (нужны свежие cookie/headers)."
            )
        entries: List[Dict[str, Any]] = []
        for payload in payloads:
            for candidate in iter_items_by_keys(payload, _COMMON_LIST_KEYS):
                if isinstance(candidate, list):
                    entries.extend(candidate)
        default_url = self.config.get(
            "item_url_template",
            "https://cs.money/market/csgo?search={name}",
        ).format(name=urllib.parse.quote(query.name_for(self.name)))
        base_url = self.config.get("base_url", "https://cs.money")
        return build_market_offers(
            self.name,
            entries,
            query,
            self.adjust_price,
            default_url,
            base_url=base_url,
            expected_name=query.name_for(self.name),
        )


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
    raw_expr = expr
    expr = expr.strip()
    if not expr:
        raise ConfigError("--item не должен быть пустым")
    parts: Dict[str, str] = {}
    freeform_name: Optional[str] = None
    for chunk in expr.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            if freeform_name is None and not parts:
                freeform_name = chunk
                continue
            raise ConfigError(
                f"Неверный формат '--item {raw_expr}'. Используйте key=value;..."
            )
        key, value = chunk.split("=", 1)
        parts[key.strip()] = value.strip()
    name = parts.pop("name", None)
    if name is None and freeform_name:
        name = freeform_name
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
    parser.add_argument(
        "--item",
        action="append",
        help=(
            "Добавить/переопределить предмет: можно указать название или"
            " key=value;min_price=...;min_roi=..."
        ),
    )
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
