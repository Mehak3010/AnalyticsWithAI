# Verification log

What was tested, how, and what was observed. Environment: Python 3.11, local run on the DataCo Smart Supply Chain data, monitoring window Jan 2015 to Sep 2017 (`dataset.end_date: 2017-09-30`).

| Claim | How it was tested | Result |
|---|---|---|
| Ratio KPIs are recomputed from components, never averaged | `pytest`: `test_ratio_is_recomputed_from_components_not_averaged` | PASS |
| Config-driven | Changed only `z_threshold` from 3.5 to 8.0 in a copy of the YAML; full-history detections fell from 922 to 592, no code change | PASS |
| Detects a real spike | Injected a 3x sales spike on 2016-06-15 into the real data (`check_injection.py`) | PASS: one `up` alert dated 2016-06-15 |
| Clean data stays quiet | Same script, no injection, as-of 2016-06-17 | PASS: no alerts |
| Persistence suppresses one-day noise | Injected a 1.35x one-day blip (2016-03-10) | PASS: no alert |
| Sustained shifts are caught, with no mirror alert | Injected 1.5x for 3 days (2016-09-05 to 09-07) | PASS: one `up` alert on 2016-09-07 |
| Cooldown deduplication | Ran `--as-of 2017-06-06` twice, then `--as-of 2017-07-07` | PASS: 1 alert, then 0 (suppressed), then 1 new alert |
| LLM summary with fallback | Set an invalid `GROQ_API_KEY` | PASS: 2 retries logged, no crash, rule-based summary in the digest |
| Severity scoring | `pytest`: severity increases with deviation | PASS (unit test) |
| Email digest renders | `--dry-run` writes `reports/last_email_digest.html` | PASS |
| Email delivered via SMTP | Real Mailtrap send | TODO |
| CI | GitHub Actions | Feature integration and variance-context test PRs were merged after their CI runs passed; rerun CI after future changes |
| Row count and date range | Counted rows in the tidy table after the `end_date` filter | PASS: 171,962 rows, 2015-01-01 to 2017-09-30, 5 markets |

## Implemented components not yet fully operationalized

- **Root-cause attribution:** `src/anomaly_agent/attribution.py` calculates additive segment contributions and ratio mix/rate effects; `cli.py` attaches the results to headline anomalies when enough history is available.
- **Numeric narration validation:** `src/anomaly_agent/narration/validator.py` checks numeric claims against the slim fact bundle. The narrator retries rejected output and then uses the deterministic template.
- **Forecasting and variance:** `src/anomaly_agent/forecasting.py` provides ETS and seasonal-naive forecasts. Forecast context is attached to eligible headline anomalies and displayed in the email digest. `src/anomaly_agent/variance.py` provides the volume-price-mix bridge, and `src/anomaly_agent/variance_context.py` attaches a reconciled bridge to eligible headline sales anomalies using net sales per unit as effective price. The email digest renders the variance context. Unit and integration tests cover the bridge and fail-open behaviour.
- **Evaluation harness:** `evaluation/` contains anomaly injection, detector replay, baseline methods and scoring for precision, recall, F1, detection delay and false-alarm rate. Run its tests with the full pytest suite; a published real-data comparison report remains to be generated.

## Known limits

- The last four months of the DataCo data (from early Oct 2017) degenerate: a constant 68 to 69 order lines a day, only two markets and one or two categories a day. They are excluded from the monitoring window.
- The dataset appears partly synthetic and has weak seasonality, so naturally occurring anomalies are few. Detection claims rest on injected anomalies.
- The injection check covers the sales metric on the headline series only.
