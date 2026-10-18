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
| CI | GitHub Actions run | TODO: add link to a green run |
| Row count and date range | Counted rows in the tidy table after the `end_date` filter | PASS: 171,962 rows, 2015-01-01 to 2017-09-30, 5 markets |

## Not implemented yet

- Root-cause attribution (segment contribution to the headline move). Drivers shown in alerts are recently abnormal segments in the same direction, not a measured contribution.
- Forecast-variance analysis (ETS, volume-price-mix bridge). The module exists but is not wired into the pipeline.
- Numeric validator that checks every number in an LLM summary against the data.
- Evaluation harness reporting precision, recall and detection delay.

## Known limits

- The last four months of the DataCo data (from early Oct 2017) degenerate: a constant 68 to 69 order lines a day, only two markets and one or two categories a day. They are excluded from the monitoring window.
- The dataset appears partly synthetic and has weak seasonality, so naturally occurring anomalies are few. Detection claims rest on injected anomalies.
- The injection check covers the sales metric on the headline series only.
