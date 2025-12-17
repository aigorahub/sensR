# System Architecture

## High-Level Design

`sensPy` is a library designed to be a drop-in Python replacement for the R package `sensR`. It focuses on statistical inference for sensory discrimination data.

### Core Design Pattern: Protocol-Driven Dispatch

The central abstraction is the **Protocol** (defined in `senspy.core.types`).
1.  **Input**: User selects a method (e.g., `"triangle"`, `"twoafc"`).
2.  **Dispatch**: The system resolves this string to a `Protocol` enum.
3.  **Link**: `senspy.links.get_link(protocol)` retrieves a `Link` object containing specific mathematical functions (`linkinv`, `linkfun`, `mu_eta`) for that protocol.
4.  **Compute**: Statistical functions (`discrim`, `power`) use these generic link methods to perform calculations without knowing the specifics of the protocol.

### Data Flow

1.  **Raw Data**: Counts of correct/incorrect responses or confusion matrices.
2.  **Estimation**: `discrim()` or model classes (e.g., `betabin()`) estimate parameters (d-prime, pc).
3.  **Optimization**: `scipy.optimize` (often `brentq` or `minimize`) is used to invert link functions or maximize likelihoods.
4.  **Result Objects**: Functions return rich data classes (e.g., `DiscrimResult`) containing point estimates, standard errors, and test statistics.

### Key Abstractions

- **`Link`**: A dataclass encapsulating the psychometric function for a specific test type.
- **`StatsModel`**: (Implicit) Pattern for complex models (`BetaBinomial`, `TwoAC`) providing `coefficients`, `vcov`, and `log_likelihood`.

### R Parity

The architecture is strictly constrained by the need to replicate R results. Where `sensR` uses approximations (e.g., for Hexad or 2-out-of-5), `sensPy` implements the same approximations rather than "purer" math, to ensure validation succeeds.
