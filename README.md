# monitor-vulns

## CS:GO Deal Monitor

Этот репозиторий содержит скрипт `deal_monitor.py` для мониторинга выгодных предложений на маркетах cs.money, lis-skins.com, market.csgo.com и skinport.com с сопоставлением цен из Steam Community Market. Скрипт оценивает прибыль, ROI и потенциальный доход от перепродажи, обновляя данные каждые N секунд.

### Подготовка

1. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
2. Скопируйте `config.example.yaml` в `config.yaml` и настройте параметры:
   - Добавьте API-ключ `market.csgo.com` (или задайте переменную окружения `CSGO_MARKET_API_KEY`).
   - Укажите актуальные cookies/заголовки для сайтов, защищённых Cloudflare (`cs.money`, `lis-skins.com`).
   - Определите список интересующих скинов и минимальные требования по цене/прибыли.

### Запуск

```bash
python deal_monitor.py -c config.yaml
```

Дополнительно можно задавать предметы через CLI:

```bash
python deal_monitor.py -c config.yaml --item "name=AK-47 | Redline (Field-Tested);min_price=40;min_roi=0.06"
```

### Возможности

- Опрос маркетов каждые `refresh_interval` секунд.
- Настраиваемые фильтры по цене, ROI, прибыли и списку маркетов.
- Отображение результатов в консоли с использованием библиотеки `rich`.
- Корректировка цен с учётом комиссий Steam.

Подробнее о структуре конфигурации смотрите в `config.example.yaml`.
