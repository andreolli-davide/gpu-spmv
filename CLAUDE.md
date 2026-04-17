# SpMV GPU Optimization Project

## Documentation Rule
After implementing any new source file or significant function, update the corresponding
section in docs/ and regenerate documentation with `make docs`. Every function/type
declared in a header must have a Doxygen-compatible comment block.

## Project Structure
- `src/common/` — shared utilities (sparse matrix, Matrix Market parser, timers)
- `src/cpu/` — CPU baseline SpMV implementation
- `src/gpu/` — GPU SpMV kernels (Phase 2)
- `examples/` — usage examples
- `tests/` — correctness tests
- `docs/` — Sphinx + Breathe HTML documentation (build with `make docs`)