# tradey task runner. Run `just` for the default checks, or `just <recipe>`.

# Lint, type-check and test (mirrors CI).
default: lint typecheck test

# Run the test suite.
test:
    uv run pytest -v

# Lint with ruff.
lint:
    uvx ruff check .

# Auto-format with ruff.
fmt:
    uvx ruff format .

# Type-check with pyright.
typecheck:
    uv run pyright

# Regenerate the protobuf stubs from proto/client.proto (needs protoc, see the
# comment at the top of that file). Overwrites the hand-simplified .pyi.
proto:
    cd proto && protoc --python_out=. --pyi_out=. client.proto

# Run tradey, e.g. `just run portfolio.portfolio 5000 -n 3`.
run *ARGS:
    uv run tradey {{ARGS}}
