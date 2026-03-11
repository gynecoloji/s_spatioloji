# s_spatioloji

![PyPI version](https://img.shields.io/pypi/v/s_spatioloji.svg)

scalable spatioloji

* Created by **[gynecoloji](https://github.com/gynecoloji)**
  * PyPI: https://pypi.org/user/gynecoloji/
* PyPI package: https://pypi.org/project/s_spatioloji/
* Free software: MIT License

## Features

* TODO

## Documentation

Documentation is built with [Zensical](https://zensical.org/) and deployed to GitHub Pages.

* **Live site:** https://gynecoloji.github.io/s_spatioloji/
* **Preview locally:** `just docs-serve` (serves at http://localhost:8000)
* **Build:** `just docs-build`

API documentation is auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

Docs deploy automatically on push to `main` via GitHub Actions. To enable this, go to your repo's Settings > Pages and set the source to **GitHub Actions**.

## Development

To set up for local development:

```bash
# Clone your fork
git clone git@github.com:your_username/s_spatioloji.git
cd s_spatioloji

# Install in editable mode with live updates
uv tool install --editable .
```

This installs the CLI globally but with live updates - any changes you make to the source code are immediately available when you run `s_spatioloji`.

Run tests:

```bash
uv run pytest
```

Run quality checks (format, lint, type check, test):

```bash
just qa
```

## Author

s_spatioloji was created in 2026 by gynecoloji.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
