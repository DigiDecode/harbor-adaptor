# slopon-harbor

A [Harbor](https://github.com/harbor-framework/harbor) agent adaptor that
runs the **SlopOn backend's agentic loop** inside each trial container.
The agent is `SlopOnAgent` (`slopon_harbor.agent:SlopOnAgent`); on a
Linux benchmark host it drives a shared SlopOn backend instance over
WebSocket RPC while an in-container SlopOn runner — uploaded fresh into
every trial — performs the actual tool work against the task's
filesystem, which is what Harbor's verifier scores.

```
harbor trial ──(WS JSON-RPC)──► SlopOn backend ◄──(WS runner protocol)── runner.js (inside the trial container)
```

## 1. Pre-provision the backend (once per benchmark host)

The backend must be listening on an address task containers can reach —
its default `127.0.0.1` bind is unreachable from containers and the
adaptor refuses to derive an endpoint from a loopback config.

1. Create `~/.slopon/config.json` **before the first start** (schema:
   `server.port` int, optional `server.listenIp` IP, optional
   `server.apiKey` ≥ 32 chars):

   ```json
   {
     "server": {
       "port": 4200,
       "listenIp": "0.0.0.0",
       "apiKey": "<random string, at least 32 characters>"
     }
   }
   ```

2. Start the backend from the prepared runtime (see §2):
   `node /abs/path/slopon-runtime/index.js`. The host-local runner the
   backend auto-spawns must start successfully (fail-fast) — that is
   expected and harmless for benchmarks; projects created by the adaptor
   are explicitly assigned to the per-trial container runners instead.

The adaptor reads its endpoint and API key from this file (or from
`SLOPON_BACKEND_URL` / `SLOPON_BACKEND_API_KEY` in the harbor process
environment — an explicit URL opts out of the loopback check).

**The benchmark instance is disposable by design**: provider rows are
reused by name and never re-read their credentials, so rotating LLM
credentials means starting from a fresh `~/.slopon` (fresh DB).

## 2. Prepare the runner runtime (once per benchmark host)

1. Download the combined customer bundle published under the literal
   `latest` tag (assets are overwritten in place on every release, so
   **record the asset's SHA-256 from the release-notes table** to make
   the benchmarked backend version traceable):

   ```
   https://github.com/DigiDecode/SlopOn.dev/releases/download/latest/slopon-latest-linux-x64.zip
   ```

2. Unzip. The layout is `slopon-latest-linux-x64/{README.md, frontend/,
   backend/}` — only `backend/` matters (the `frontend/` Flutter desktop
   app is irrelevant to benchmarks). The `backend/` subtree ships
   `index.js`, `runner.js`, `package.json`, `pnpm-workspace.yaml`,
   `README.md`, `migrations/`, `prompts/` — and **no `node_modules`**.

3. Run `npm install` inside
   `<unzip-dir>/slopon-latest-linux-x64/backend` (requires Node ≥ 20,
   git, and a C toolchain on the host — `better-sqlite3` has a native
   build). npm produces the flat `node_modules/` layout the runner
   needs.

4. That `backend/` directory (release tree **after** the install) is the
   value of `SLOPON_RUNNER_RUNTIME` and must contain `runner.js`,
   `prompts/`, `node_modules/` (the adaptor checks at startup).

Expect the runtime to be **hundreds of MB uncompressed**; it is uploaded
and extracted per trial, multiplied by concurrent trials. Ensure the
Docker VM disk and per-task storage budgets (hello-world:
`storage_mb = 10240`) have headroom. Host and containers share the same
Linux x64 machine, so the host-installed `node_modules` (including the
native `better-sqlite3` build and `node_modules/.bin` language servers)
is binary-compatible in-container.

## 3. Install the adaptor

Into harbor's Python ≥ 3.12 venv:

```
uv pip install -e /harbor-adaptor
python -c "from slopon_harbor.agent import SlopOnAgent; print('ok')"
```

## 4. Single trial (hello-world, stock image)

Run from the harbor repository root, with `SLOPON_LLM_API_KEY` exported
in the shell:

```
export SLOPON_LLM_API_KEY=<key>
harbor trials start -p examples/tasks/hello-world \
  -a slopon_harbor.agent:SlopOnAgent \
  -m <model_id> \
  --ae SLOPON_RUNNER_RUNTIME=/abs/path/slopon-runtime \
  --ae SLOPON_LLM_BASE_URL=<llm-base-url> \
  --agent-setup-timeout 600 \
  --agent-timeout 900
```

The stock `ubuntu:24.04` image ships no Node and no downloader; the
adaptor bootstraps curl via apt and then installs the pinned Node v22.x
`.tar.gz` from nodejs.org during setup (this is why the setup timeout is
raised). Expected artifacts after the trial:
`trial_dir/agent/history.json` (final chat transcript from
`chat.getHistory`) and `trial_dir/agent/runner.log`; a verifier reward is
produced as usual.

## 5. Job / concurrency

```
harbor job start -c examples/hello-world-slopon.yaml            # from the harbor repo root
harbor job start -c examples/hello-world-slopon.yaml --print-config
```

Start it from a shell with `SLOPON_LLM_API_KEY` exported. The example
sets `n_concurrent_trials: 2`; each trial gets its own backend project,
runner registration, bot, and chat (names derive from the trial's
`session_id` plus a per-run random suffix, so `--trial-name` reruns
cannot collide).

## 6. Networking matrix

| Task policy | What to do |
|---|---|
| `public` (hello-world default) | Nothing — no egress sidecar is installed; the runner reaches the backend at the bridge-gateway IP. |
| `allowlist` | The dialed backend address (the gateway IP the adaptor logs, or a covering CIDR such as `172.16.0.0/12`) must be in **both** `environment.extra_allowed_hosts` (the runner already connects during setup) **and** `agent.extra_allowed_hosts`; the LLM provider host must be in the agent list. If the image lacks a downloader, also allow `archive.ubuntu.com` + `nodejs.org` in the environment list for the setup-phase bootstrap. |
| `no-network` | Unsupported — the trial fails fast (neither the backend nor the LLM provider is reachable). |

Container egress never needs `github.com` / `objects.githubusercontent.com`
/ `registry.npmjs.org` — the runtime arrives as a host-built tarball.
Docker Desktop dev machines: set
`SLOPON_BACKEND_PUBLIC_URL=ws://host.docker.internal:<port>` and prepare
the runtime for the Docker VM's platform. The adaptor logs the discovered
gateway IP during setup so operators know exactly which address is
dialed.

## 7. Secrets & token hygiene

- `SLOPON_BACKEND_API_KEY` is derived from host `~/.slopon/config.json`
  (or the harbor process env). `SLOPON_LLM_API_KEY` is read from the
  harbor **process environment**. Both are **never** passed via
  `agent.env` / `--ae` — the adaptor refuses them there with a warning:
  harbor injects agent env into every container exec (setup and agent
  phase), which would leak the keys into the agent-under-test's sandbox.
  The in-container runner needs no LLM credentials from the environment
  at all — they travel to it inside the job spec.
- The runner's backend URL and one-time token travel via the exec `env`
  dict (`RUNNER_URL` / `RUNNER_TOKEN`), not argv, so `ps` shows nothing.
  Residual exposure is `/proc/<pid>/environ`, readable only by the same
  user/root inside that trial's own container — an accepted trade-off
  scoped to the container.
- Harbor redacts sensitive env values in trial locks; the runner token
  is never logged by the adaptor.

## 8. Ops notes

- **Row accumulation**: every trial adds a runner row, project (+5 seeded
  bots, 4 phases), source folder, benchmark bot, chat + messages, and
  config rows (compaction disabled globally once; per-tool approval
  overrides per project). SQLite rows accumulate by design; teardown is
  a separate follow-up task.
- **Credential rotation** = fresh benchmark instance (fresh `~/.slopon`).
- **Backend restart mid-trial** orphans in-flight jobs (known backend
  limitation): restart the benchmark job; the DB is disposable.
- **Never set provider `reasoningEffort` to `null`** (breaks
  `openai_compatible` providers; the adaptor never sends the field on
  create — the service defaults it to `'high'`).
- **The in-container runner needs a writable home directory**
  (`~/.slopon/<instanceId>/` is derived at startup). Works as root
  (hello-world's default agent user); tasks with a non-root agent user
  without a writable home fail at runner startup — visible in the
  `runner.log` tail attached to the runner-online timeout error.
- Backend concurrency is bounded per chat only (one active job per chat;
  no global cap) — keep harbor-side `n_concurrent_trials` sane.
- `SLOPON_RUNNER_ONLINE_TIMEOUT_SEC` (default 90 s) bounds the
  runner-online wait; the timeout error carries the `runner.log` tail.

## Configuration reference

| Variable | Required | Source | Meaning |
|---|---|---|---|
| `SLOPON_BACKEND_URL` | no | agent env / process env | Adaptor's own backend WS URL; defaults to `ws://<listenIp>:<port>` from `~/.slopon/config.json` (loopback rejected unless explicit). |
| `SLOPON_BACKEND_API_KEY` | no | **process env only** | Defaults to `server.apiKey` from `~/.slopon/config.json`. |
| `SLOPON_BACKEND_PUBLIC_URL` | no | agent env / process env | URL the container dials; default `ws://<gateway>:<port>` discovered in-container. |
| `SLOPON_RUNNER_RUNTIME` | **yes** | agent env / process env | Host path to the prepared runtime (`runner.js`, `prompts/`, `node_modules/`). |
| `SLOPON_CONTAINER_WORKDIR` | no | agent env / process env | Default `/app` (hello-world WORKDIR); becomes the project's source folder. |
| `SLOPON_LLM_TYPE` | no | agent env / process env | Provider type; default `openai_compatible`. |
| `SLOPON_LLM_BASE_URL` | **yes** | agent env / process env | LLM provider base URL (non-secret). |
| `SLOPON_LLM_API_KEY` | **yes** | **process env only** | LLM provider key. |
| `SLOPON_LLM_MODEL_ID` | **yes*** | agent env / process env | Falls back to harbor `-m/--model`. |
| `SLOPON_RUNNER_ONLINE_TIMEOUT_SEC` | no | agent env / process env | Default `90`. |
| `SLOPON_NODE_VERSION` | no | agent env / process env | Pinned Node; default `22.17.1`, major must be 22. |

## Development

```
uv pip install -e '.[dev]'
pytest
ruff check .
```

The test suite is fully hermetic (in-process fake WS backend, fake
clients and environments); it needs no running backend or Docker.
