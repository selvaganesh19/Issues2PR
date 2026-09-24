# mathlib (sample repo)

A deliberately tiny package used as a fixture for the Issue2PR agent demo.

## Known bug

`mathlib.add(a, b)` currently returns `a - b` instead of `a + b`. The test in
`tests/test_add.py` asserts `add(2, 3) == 5`, so the suite FAILS until the bug
is fixed. An agent (or a human) is expected to change the `-` to `+`.
