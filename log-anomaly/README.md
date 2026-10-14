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

Chance clumps are found after the fact: every run of at least
`novel_error_count` lines of one background message within
`novel_error_window_minutes` becomes one quiet case, overlapping runs merged.
They are fleet-wide, so they name their keys and orgs in `detail` instead of
one entity.

Two cases stand alone: `key_silence` (a busy key goes quiet for two weekday
hours, must alert) and `onboarding` (a new org appears and ramps up over days,
must stay quiet).

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
| `onboarding` | quiet | randomly drawn orgs, which then take part in no other scenario |

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
| `truth.jsonl` | one case per line: `id`, `kind`, `pair`, `expect` (`alert` or `quiet`), `scored`, `org.id`, `api_key.id`, `start`, `end`, `detail`; fleet-wide cases have neither id and list `keys` and `orgs` in `detail` |
| `world.json` | every org and key with its archetype, rate, provider and schedule |
| `manifest.json` | seed, config digest, run window, package versions; `generate` adds record counts and the log digest |
| `journal.jsonl` | `stream` only: each start, minute-by-minute progress, and where it stopped |

Records carry `org.id`, `api_key.id` (except health checks) and
`source.provider`; requests add OpenTelemetry HTTP attributes. Nothing in a
record says whether it was injected: injected and natural events use the same
record builders, and `test_injection_leaves_no_structural_trace` checks that a
run with scenarios produces the same services, severities, attribute keys and
request and health-check bodies as one without.

Over a live stream the journal's gaps, from a `stopped` entry's
`sent_through` to the next `emitting_from`, are time no detector saw.

## Scoring

A system under test is scored on the pages that reached a person, one JSON
object per line:

```json
{"time": "2026-10-27T03:41:10Z", "api_key.id": "key_0123456789", "org.id": "org_0123456789"}
```

`time` is required, RFC 3339 with an offset. Name the key, the org, both, or
neither, as the page did; other fields are ignored. An id the world does not
know is an error, never a silent miss. Turning a vendor's incidents or issues
into this file is the one step outside `logsim`.

Every page is classified once. The rules live in `logsim/score.py`'s module
docstring; in short:

- A page detects a must-alert case when it names the case's entity and arrives
  between the case's start and 15 minutes (`--deadline-minutes`) after its
  end. Key-level cases need the key; naming only its org is not enough.
  Org-level cases (outages, overruns, onboarding) accept the org or any of its
  keys.
- Later pages on a detected case are duplicates, reported but not false.
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

On the full `dev.toml` run with seed 1 (two scored weeks), it detected all 23
must-alert cases, a 95% Wilson interval of 0.86 to 1.0, with a median of 0.84
minutes to detect. It also sent 1,058 false pages, 529 a week, for a
precision of 0.02:

| Cause | False pages | What happened |
|---|---|---|
| `unexplained` | 717 | 716 are request-spike pages on the two onboarded orgs' keys after their ramp: with no warm-up history, the frozen floor fires every 30 minutes for the rest of the run |
| `chance_error_clump` | 141 | it paged on 123 of 126 chance clumps |
| `onboarding` | 122 | the same keys during their ramp |
| `maintenance_window` | 78 | it paged on all 77 maintenance windows |

It never paged on a natural burst (0 of 277) or a scheduled export (0 of 14):
per-key history is enough for those pairs. The outage and error pairs and new
customers are where a threshold fails, and where a system has to show it can do
better.

A false page is charged to a quiet case by entity and time, not by what the
page says: 16 of the baseline's request-spike pages are charged to chance
clumps because the key they named was in one.

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
21 days: 19.2 million records, 0.92 million a day, 7.4 GB as JSONL. On an
Apple-silicon laptop, generating it took 2.6 minutes, the baseline 52 seconds,
and scoring 0.2 seconds.

`traffic.rate_scale` multiplies every request rate, bursts included, and keeps
their ratios. Lower rates also mean sparser counts per minute, which changes
how noisy a key looks, so scale down knowingly.

## Not built yet

- **Vendor adapters.** Nothing yet turns Better Stack incidents or Sazabi
  issues into a pages file.
- **The agent arm.** A fresh agent with a fixed prompt and read-only queries
  against the same logs is planned as a comparison arm; neither the prompt nor
  its harness exists.
- **Self-exciting bursts.** Retry storms, where one failure triggers more, are
  not modelled.
- **Delivery parity.** Nothing checks that every system received every record
  beyond the Collector's own exporter logs.
