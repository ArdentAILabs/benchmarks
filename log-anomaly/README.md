# log-anomaly

`logsim` simulates the logs of a multi-tenant API: orgs, their API keys, and the
databases they connect. It labels every case that should alert and every case
that should not. One seeded run can be written to files or streamed live, so
several alerting systems can watch the same records, and `logsim score` grades
each one's pages against the same truth.

## Why matched pairs

A detector that never alerts has no false positives. A detector that alerts on
every spike catches everything. Neither is useful, and a benchmark full of
anomalies that one threshold can separate will not tell them apart.

So most cases here come in pairs. The case that must alert reproduces the raw
signal of a case that must stay quiet, drawn from the same distribution. Only
the entity's own history, or the metadata on its records, says which is which.

| Raw signal | Stays quiet when | Must alert when |
|---|---|---|
| a burst of ~1,000 requests/s for ~90 s from one key (before `rate_scale`) | the key bursts like this about twice a day (`natural_burst`) | the key is steady and never has (`burst_on_steady_key`) |
| a source fails health checks for 5–10 min | the org's provider does daily maintenance at that hour (`maintenance_window`) | the provider does no maintenance (`unplanned_outage`) |
| a maintenance window | it opens an hour later in UTC because the provider's zone left summer time | it runs 30–60 min instead of 5–10 (`maintenance_overrun`) |
| a key exports ~20 GB at night | the key does this every week (`scheduled_export`) | the key never has (`unscheduled_bulk_export`) |
| a handful of errors in ten minutes | a known error that clumps by chance (`chance_error_clump`) | an error message never seen before (`novel_error`) |
| an error message never seen before | a deploy renamed a known message and it arrives at the old rate (`renamed_error`) | it is new, and nothing went away (`novel_error`) |
| a dependency fails a key's requests with 503 for 20–60 s, and its clients retry | the clients back off and the retries die out (`retry_echo`) | each failure spawns more than one retry, until the admission limit and a circuit breaker (`retry_storm`) |
| a key's 5xx responses rise 4–8x for 30–90 min | its daily batch job multiplied its traffic, errors with it (`batch_surge`) | the share of requests failing rose and traffic did not (`error_rate_shift`), or a deploy broke one provider's orgs until a fix shipped (`bad_deploy`) |
| an org's numbers climb over days | its traffic grows and stays grown (`organic_growth`) | its latency climbs until the leaking process restarts (`latency_leak`) |

Chance clumps and renamed errors are found after the fact, from the emitted
lines. Every run of at least `novel_error_count` lines of one background
message within `novel_error_window_minutes` becomes one chance clump,
overlapping runs merged. A renamed error runs from its deploy to the new
text's `novel_error_count`-th line, when it stops being new. Both are
fleet-wide, so they name their keys and orgs in `detail` instead of one entity.

Three cases stand alone: `key_silence` (a busy key goes quiet for two weekday
hours, must alert), `provider_outage` (every org on one provider fails its
health checks at once, must alert, and should page once rather than once per
org) and `onboarding` (a new org appears and ramps up over days, must stay
quiet).

The plan keeps every must-alert case at least 30 minutes from any natural
quiet case that shares its signal and its key (or, for a case on whole orgs,
anything in them), so one page cannot be right about both.
`test_alert_cases_keep_clear_of_quiet_look_alikes` checks it.

## The world

A config describes a distribution over worlds; the seed draws one.

| Part | Behaviour |
|---|---|
| `steady` keys | Poisson requests around a daily curve in the org's timezone, lower on weekends |
| `bursty` keys | the same, plus bursts at about 1,000 requests/s before `rate_scale` (100/s in `dev.toml`) |
| `overdispersed` keys | Poisson with a gamma-distributed rate per minute (negative binomial counts) |
| providers | each org connects one database; some providers take it down for a few minutes daily, scheduled in local time or in UTC |
| health checks | every org's source is probed every 30 s; failures read the same whether planned or not |
| exports | every key exports small amounts now and then; some keys export ~20 GB weekly |
| server errors | 0.5% of requests fail with 500 |
| daily batches | some keys run a batch job at about the same local time every day, multiplying their traffic for 30–90 minutes |
| retries | now and then a dependency fails a busy key's requests for 20–60 s and its clients retry, as a self-exciting process simulated per second; these echoes die out |
| deploys | the fleet ships about three releases a weekday in US Pacific working hours, each logged by the `deployer` service |
| background errors | each error template arrives as a fleet-wide Poisson process |

Fixed orgs (`[[orgs.fixed]]`) mix key types, so an org's total hides what any
one key does.

## Scenarios

Injected cases fall only in the scored phase, after `warmup_days` of normal
traffic. Each kind picks a target that kind has not used yet while any remain,
so no detector is graded on what it learned from an earlier injection on the
same entity. Cases in one org are at least `min_gap_hours` apart. A schedule
that cannot be placed fails with the knob to change; it is never thinned.

| Kind | Expect | Eligible targets |
|---|---|---|
| `burst_on_steady_key` | alert | steady keys in orgs that also have a bursty key |
| `unplanned_outage` | alert | orgs whose provider has no maintenance |
| `maintenance_overrun` | alert | scored maintenance windows; replaces the window it extends |
| `novel_error` | alert | any key; each `errors.novel` template is used once |
| `unscheduled_bulk_export` | alert | keys without a weekly export |
| `key_silence` | alert | non-bursty keys at or above `silence_min_rate_per_second` |
| `retry_storm` | alert | keys at or above `retries.min_rate_per_second`; a cascade that dies out in its first seconds is drawn again |
| `error_rate_shift` | alert | keys at or above `error_shift_min_rate_per_second`; duration and multiplier come from `[batches]` |
| `provider_outage` | alert | providers with at least two eligible orgs; every eligible org on the provider fails |
| `bad_deploy` | alert | a scored natural deploy; every eligible org on one provider fails at `bad_deploy_error_share` until a fix deploy |
| `latency_leak` | alert | any eligible org |
| `renamed_error` | quiet | a scored natural deploy no other scenario claimed; each `errors.renames` entry is used once |
| `onboarding` | quiet | randomly drawn orgs, which then take part in no other scenario |
| `organic_growth` | quiet | any org not onboarding, which then takes part in no other scenario |

"Eligible" orgs exclude the onboarding and organic-growth orgs, provider
outages and bad deploys included, so each of those tests only the one thing it
was set aside for.

## Running

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest

# Write a run to files. --hours limits the logs; truth always covers the run.
uv run logsim generate --config configs/dev.toml --seed 1 \
  --start 2026-10-19T00:00:00Z --hours 24 --out out/dev

# Score a system's pages against a run, and produce the threshold baseline's pages.
uv run logsim baseline --run out/dev --out out/dev/baseline.jsonl
uv run logsim score --run out/dev --pages out/dev/baseline.jsonl --json out/dev/report.json

# Emit a run in real time. Truth, world, manifest and journal go to --private-dir.
uv run logsim stream --config configs/dev.toml --seed 1 \
  --start 2026-10-19T00:00:00Z --endpoint http://127.0.0.1:4318 \
  --private-dir private/dev
```

`stream` waits for `--start` if it is in the future. Started late or restarted,
it regenerates the same plan and resumes from the current moment; it does not
backfill. Run it under a supervisor that restarts it. Restarting with a
different config, seed, start or package version is refused.

`collector/otelcol.yaml` fans the stream out to Better Stack and Sazabi through
an OpenTelemetry Collector. It listens on `127.0.0.1:4318`, so on Linux run it
on the host network:

```sh
export BETTERSTACK_INGESTING_HOST=... BETTERSTACK_SOURCE_TOKEN=... SAZABI_INTAKE_URL=...
docker run --rm --network host -v "$PWD/collector:/cfg:ro" \
  -e BETTERSTACK_INGESTING_HOST -e BETTERSTACK_SOURCE_TOKEN -e SAZABI_INTAKE_URL \
  otel/opentelemetry-collector-contrib:0.120.0 --config=/cfg/otelcol.yaml
```

That config passes `validate` on that image. It has not delivered to either
vendor yet.

## Outputs

| File | Contents |
|---|---|
| `logs.jsonl` | one record per line: `time` (RFC 3339, ns), `time_unix_nano`, `severity`, `service`, `body`, `attributes` |
| `truth.jsonl` | one case per line: `id`, `kind`, `pair`, `expect` (`alert` or `quiet`), `scored`, `org.id`, `api_key.id`, `start`, `end`, `detail`; cases spanning several orgs have neither id and list their `orgs` (and, where they come from emitted lines, `keys`) in `detail` |
| `world.json` | every org and key with its archetype, rate, provider and schedule |
| `manifest.json` | seed, config digest, run window, package versions; `generate` adds record counts and the log digest |
| `journal.jsonl` | `stream` only: each start, minute-by-minute progress, and where it stopped |

Request, export and error records carry `org.id`, `api_key.id` and
`source.provider`; requests add OpenTelemetry HTTP attributes and
`duration_ms`, and retries are requests like any other. Health checks carry
`org.id` and `source.provider`. Deploys are `deployed release r0042` from
service `deployer`, carrying only `deploy.release`. Nothing in a record says
whether it was injected: injected and natural events use the same
record builders, and `test_injection_leaves_no_structural_trace` checks that a
run with scenarios produces the same services, severities, attribute keys and
request and health-check bodies as one without.

Over a live stream the journal's gaps, from a `stopped` entry's
`sent_through` to the next `emitting_from`, are time no detector saw.

## Scoring

A system under test is scored on the pages that reached a person, one JSON
object per line:

```json
{"time": "2026-10-27T03:41:10Z", "api_key.id": "key_0123456789", "org.id": "org_0123456789", "signal": "errors"}
```

`time` is required, RFC 3339 with an offset. Name the key, the org, both, or
neither, as the page did; other fields are ignored. An id the world does not
know is an error, never a silent miss. Turning a vendor's incidents or issues
into this file is the one step outside `logsim`.

`signal` says what the page saw: `requests`, `errors`, `latency`, `health` or
`exports`. With it, a page matches only cases that show that signal
(`logsim/kinds.py` lists them). Without it, a page about anything in an org
counts as finding a two-day latency leak there. Label every page in a file or
none; a mix is an error, and the report says which rule it used. Compare two
systems only under the same rule.

Every page is classified once. The rules live in `logsim/score.py`'s module
docstring; in short:

- A page detects a must-alert case when it names the case's entity and arrives
  between the case's start and 15 minutes (`--deadline-minutes`) after its
  end. Key-level cases need the key; naming only its org is not enough.
  Org-level cases (outages, overruns, leaks, onboarding, growth) accept the org
  or any of its keys. Cases spanning several orgs (provider outages, bad
  deploys, chance clumps, renamed errors) accept any member org or key, or a
  page naming nothing. `logsim/kinds.py` holds the table.
- Later pages on a detected case are duplicates, reported but not false. The
  report gives pages per detected case by kind: a provider outage paged once
  per org is a provider outage paged several times over.
- Any other page in the scored phase is false. One that lands on a must-stay-quiet
  case is charged to that kind; the rest are `unexplained` (named an entity)
  or `unattributed` (named none).
- Warm-up, stream gaps, and cases that ran past the last emitted record are
  excluded, not counted either way.

The report gives false pages per week by cause, precision, recall with a
95% Wilson interval, recall by kind, the share of each quiet kind that drew a
page, and minutes to detect. False pages per week only means something next
to recall: a system that never pages has none.

## Baseline

`logsim baseline` is the control every system has to beat: one threshold per
signal, learned in warm-up and then frozen (`logsim/baseline.py` lists them).
If it scored well, the pairs would be too easy.

On the full `dev.toml` run with seed 1 (two scored weeks), scored with
signals, it detected 34 of 37 must-alert cases, a 95% Wilson interval of 0.79
to 0.97, with a median of 0.9 minutes to detect. It also sent 1,104 false
pages, 552 a week, for a precision of 0.03:

| Cause | False pages | What happened |
|---|---|---|
| `unexplained` | 736 | 733 are request-spike pages on the two onboarded orgs' keys after their ramp: with no warm-up history, the frozen floor fires every 30 minutes for the rest of the run |
| `onboarding` | 127 | the same keys during their ramp |
| `chance_error_clump` | 120 | it paged on 122 of 126 chance clumps |
| `maintenance_window` | 74 | it paged on all 77 maintenance windows |
| `batch_surge` | 24 | it paged on 13 of 135 daily batches |
| `retry_echo` | 20 | it paged on 19 of 116 echoes |
| `renamed_error` | 3 | it paged on 2 of 3 renamed messages |

It missed 3 of the 4 `error_rate_shift` cases: a 5xx count frozen at its
warm-up maximum cannot tell a rising error share from a busy hour. It caught
both latency leaks, 14 and 21 hours in, about 40% of the way through each, then
paged on them every 30 minutes until the restart: 48.5 pages per detected
leak. It never paged on a natural burst (0 of 277), a scheduled export (0 of
14) or organic growth (0 of 2).

Without signals it scores almost the same (34 detected, 1,099 false), because
its false pages mostly land where no alert case is running. A system whose
noise falls inside a long case would gain detections it did not make.

A false page is charged to a quiet case by entity, time and signal, not by
what the page says: a request-spike page on a key in a chance clump would be
charged to the clump if it carried `errors`.

## Reproducibility

The same config, seed, start and lockfile produce the same bytes:

- Every random draw comes from a stream keyed by seed, purpose, entity and hour
  (`logsim.repro.rng`), so any hour regenerates alone.
  `test_an_hour_is_the_same_alone_or_in_sequence` checks it.
- Timezone rules come from the pinned `tzdata` package, not the host
  (`logsim.repro.zone`).
- `uv.lock` pins numpy and tzdata. `test_output_is_pinned` fails if a code
  change or an upgrade changes what a seed produces.

## Public and private

`configs/dev.toml` and any seed are public: tune against them freely. A scored
run uses a private config with the same schema (its own novel error messages,
counts and rates) and a private seed. Its truth and world files stay with the
operator until scoring is done. `.gitignore` excludes `out/`, `private/` and
`*.private.toml`.

## Volume

Measured with `configs/dev.toml` (`rate_scale = 0.1`) and seed 1 over the full
21 days: 21.7 million records, 1.03 million a day, 7.8 GB as JSONL. On an
Apple-silicon laptop that was also running the test suite, generating it took
3.1 minutes, the baseline 63 seconds, and scoring 0.3 seconds.

`traffic.rate_scale` multiplies every request rate, bursts included, and keeps
their ratios. Lower rates also mean sparser counts per minute, which changes
how noisy a key looks, so scale down knowingly.

## Not built yet

- **Vendor adapters.** Nothing yet turns Better Stack incidents or Sazabi
  issues into a pages file.
- **The agent arm.** A fresh agent with a fixed prompt and read-only queries
  against the same logs is planned as a comparison arm; neither the prompt nor
  its harness exists.
- **Delivery parity.** Nothing checks that every system received every record
  beyond the Collector's own exporter logs.
