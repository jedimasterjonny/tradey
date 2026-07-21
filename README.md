# tradey

A personal portfolio-rebalancing helper. Point it at a
[Portfolio Performance](https://www.portfolio-performance.info/) export and an
amount of new cash, and it suggests how to split that cash across a small number
of holdings so your portfolio moves closer to its target asset allocation.

Under the hood it:

- reads a Portfolio Performance `.portfolio` export (a binary protobuf file),
- values every holding in your base currency, fetching live exchange rates from
  [frankfurter.dev](https://frankfurter.dev/) (ECB data) for any holding priced
  in a non-base currency,
- reads your target weights from a taxonomy in the file (default
  `Asset Allocation`), and
- runs a SciPy optimizer to find which `N` holdings to top up, and by how much,
  to minimise the squared deviation from those targets.

It only ever *suggests* an allocation. It never places trades or touches your
Portfolio Performance file.

## Requirements

- Python 3.14 (see `.python-version`)
- [uv](https://docs.astral.sh/uv/)

## Install

```sh
uv sync
```

This installs the development tooling (tests, pyright) too; add `--no-dev` for
just the runtime dependencies.

## Usage

```sh
uv run tradey <file.portfolio> <additional_investment> [options]
```

Positional arguments:

| Argument                | Meaning                                            |
| ----------------------- | -------------------------------------------------- |
| `filepath`              | Path to the `.portfolio` file.                     |
| `additional_investment` | Amount of new cash to invest (must be `> 0`).      |

Options:

| Flag                       | Default            | Meaning                                                       |
| -------------------------- | ------------------ | ------------------------------------------------------------ |
| `-n`, `--n-assets N`       | `2`                | Number of holdings to spread the investment across (`>= 1`). |
| `-x`, `--exclude NAME ...` | none               | Holding names to exclude from the optimization.              |
| `-t`, `--taxonomy NAME`    | `Asset Allocation` | Taxonomy in the file to use for target weights.              |

Example:

```sh
uv run tradey ~/portfolio.portfolio 5000 -n 2 -x "Cash"
```

Example output:

```text
--- Optimal Allocation Found ---
Invest in: ['Global Bonds', 'Cash']
Optimal allocation: £4,597.30 to 'Global Bonds' (91.95%)
Optimal allocation: £402.70 to 'Cash' (8.05%)

SSE: 1.28e-03

--- Portfolio Weights ---
+--------------+--------+--------------+--------------+
| Investment   | Target | Old Deviance | New Deviance |
+--------------+--------+--------------+--------------+
| UK Equity    |  25.0% |        21.5% |         2.5% |
| US Equity    |  40.0% |        15.7% |        -2.3% |
| Global Bonds |  30.0% |       -37.0% |         1.0% |
| Cash         |   5.0% |       -11.1% |         0.2% |
+--------------+--------+--------------+--------------+
```

`Old Deviance` / `New Deviance` are each holding's percentage over- (`+`) or
under- (`-`) its target weight, before and after the suggested investment. `SSE`
is the sum of squared deviations the optimizer minimised.

## Input format

The `filepath` argument is a Portfolio Performance `.portfolio` export, produced
by the desktop app (File → Save As → the protobuf/binary format). It is a ZIP
containing a `data.portfolio` member: the ASCII signature `PPPBV1` followed by a
serialized `PClient` protobuf message.

This file contains your personal financial data, so **it is not committed** —
`*.portfolio` is listed in `.gitignore`. Keep your own export outside the repo.

## How targets work

tradey reads a two-level taxonomy from the file: top-level categories (e.g.
*Equity*, *Bonds*) each split into sub-categories (e.g. *UK Equity*,
*US Equity*). Each holding is valued in your base currency and rolled up to the
sub-category it is assigned to. A sub-category's target weight is its weight
within its parent category multiplied by the category's weight in the whole
portfolio.

Real taxonomies often do not add up to exactly 100% (unclassified cash, partial
weights, and so on). tradey normalises the derived targets so they sum to 100%,
and prints a warning to stderr when the raw weights are more than one percentage
point away from 100% before rescaling. Optimization is then over the normalised
targets.

## Development

Common tasks are collected in the [`justfile`](justfile) (install
[just](https://github.com/casey/just), then run `just <recipe>`), or run the
commands directly:

```sh
uv run pytest          # run the test suite
uvx ruff check .       # lint
uvx ruff format .      # format
uv run pyright         # type-check
```

Install the git hooks (ruff, yamllint, whitespace/EOF fixers) once with:

```sh
uvx pre-commit install
```

CI (`.github/workflows/test.yml`) runs `ruff check`, `ruff format --check`,
`pyright`, and `pytest` on every push to `main` and every pull request.

### Regenerating the protobuf stubs

The schema lives in [`proto/client.proto`](proto/client.proto) (vendored from
the upstream Portfolio Performance application). The committed
`proto/client_pb2.py` and `proto/client_pb2.pyi` were generated with protobuf
gencode **7.34.1** (`protoc` / `libprotoc` 34.1); use a matching `protoc` so the
runtime-version check baked into `client_pb2.py` keeps passing:

```sh
just proto
# or, equivalently:
cd proto && protoc --python_out=. --pyi_out=. client.proto
```

Note that `proto/client_pb2.pyi` is a hand-simplified stub covering only the
fields tradey actually reads; a full regeneration overwrites it with the
complete auto-generated interface. Regenerate deliberately.

## Disclaimer

This is a personal tool, not financial advice. It suggests allocations from the
data you give it; it makes no judgement about whether those holdings or targets
are appropriate for you. Check its output before acting on it.
