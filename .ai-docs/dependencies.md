# Module Dependencies

## External Dependencies

- **`numpy`**: Fundamental array operations.
- **`scipy`**: Optimization (`optimize`), statistics (`stats`), and integration (`integrate`).
- **`pandas`**: Optional, for data frame inputs/outputs (referenced in `pyproject.toml`).
- **`numba`**: Performance optimization for heavy likelihood loops (referenced in `pyproject.toml`, possibly future phase).

## Internal Dependency Graph

1.  **`senspy.core`** (Leaf)
    - Defines types and enums used by everyone.
    - No internal dependencies.

2.  **`senspy.links`**
    - Depends on `senspy.core`.
    - Contains the math; used by `discrim`, `power`, `utils`.

3.  **`senspy.utils`**
    - Depends on `senspy.core` and `senspy.links` (for rescaling).
    - Provides stats helpers used by models.

4.  **`senspy.discrim` / `senspy.power` / `senspy.betabin`** (High Level)
    - Depend on `senspy.links` for math.
    - Depend on `senspy.utils` for p-values and critical values.
    - Depend on `senspy.core` for result objects.

## Critical Paths

- **`senspy.links.psychometric`**: This is the kernel. If this breaks, all estimations (`discrim`, `power`, models) break. Tests here are critical.
