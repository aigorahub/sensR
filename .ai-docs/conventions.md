# Coding Conventions

## Naming Patterns

- **Functions**: `snake_case` (e.g., `psy_fun`, `discrim_power`).
  - *Mapping from R*: R's `camelCase` or `dot.separated` is converted to `snake_case`. Example: `discrimSS` -> `discrim_sample_size`.
- **Classes**: `PascalCase` (e.g., `BetaBinomialResult`).
- **Variables**: `snake_case`.
  - `pc`: Proportion Correct.
  - `pd`: Proportion Discriminators.
  - `d_prime`: d' (Sensitivity).

## Type Definitions

- **Type Hints**: Mandatory for all public functions.
- **ArrayLike**: Used for inputs to accept lists, tuples, or NumPy arrays.
- **Enums**: Prefer `Protocol` and `Statistic` enums over raw strings, though strings are parsed for convenience.

## R Parity

- **Numerical**: Results must match R within strict tolerances (`1e-6` to `1e-10` depending on the metric).
- **Defaults**: Default arguments (e.g., `conf_level=0.95`, `test="difference"`) should match `sensR` defaults unless there is a compelling Pythonic reason not to.

## Error Handling

- Input validation happens early (checking probability bounds [0,1], non-negative counts).
- Raise `ValueError` for invalid inputs.
- Warnings (`warnings.warn`) used for optimization convergence issues or edge cases (e.g., boundary parameters).
