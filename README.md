# sensPy

A Python port of the R package `sensR`. This project is in early development.

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.
To install the development version:

```bash
poetry install
```

## Usage

```python
from senspy import psyfun, BetaBinomial

pc = psyfun(1.5)
model = BetaBinomial(alpha=2.0, beta=2.0, n=10)
print(model.pmf(5))
```

## Contributing

Contributions are welcome! Please submit pull requests via GitHub. See
`CONTRIBUTING.md` for guidelines.

## License

This project is licensed under the terms of the GNU General Public License
version 2.0 or later. See [LICENSE](LICENSE) for details.
