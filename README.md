# oic-toolkit

A consolidated Python package for image informatics and bioimage analysis workflows, maintained by the VAI Optical Imaging Core.

## Getting Started

### Installation

You can install the library directly either from PyPi or from this repository.

```bash
pip install oic-toolkit
pip install "oic-toolkit @ git+https://github.com/vaioic/oic-toolkit.git@main"
```

If you need the latest bleeding-edge version (which likely contains bugs and other incomplete code)

```bash
pip install "oic-toolkit @ git+https://github.com/vaioic/oic-toolkit.git@dev"
```

### Issues

If you encounter any issues with running the code or have any questions, please create an [Issue](https://github.com/vaioic/template-analysis-uv/issues) or send an email to opticalimaging@vai.org. If you are reporting a bug, please include any error messages to aid with troubleshooting.

### License

This project is licensed under the GPLv3 license. For more details, see [LICENSE](LICENSE).

### Citing & Acknowledgements

This repository is publicly available for open-source use, but it is developed and maintained by the Optical Imaging Core at the Van Andel Institute. If code from this repository contributed to data used in a publication, abstract, or presentation, please cite and acknowledge our work based on your affiliation:

#### For External Users
Please cite this repository and acknowledge the author(s) in your publication's materials, methods, or acknowledgements section:
> "Image analysis pipelines were adapted from open-source tools developed by the Optical Imaging Core at the Van Andel Institute (GitHub:[oic-toolkit](https://github.com/vaioic/oic-toolkit))."

If you require custom adjustments or advanced analysis support, please contact us at opticalimaging@vai.org.

#### For Internal Users & Close Collaborators
If you are an internal researcher or an external collaborator working directly with our staff, please include our Research Resource Identifier (RRID) in your materials and methods section:
> "Image analysis and data processing were performed in collaboration with the Optical Imaging Core at the Van Andel Institute (RRID:SCR_021968)."

Please review the Acknowledgement and Authorship Guidelines on [VAI's Core Technology and Services website](https://vanandelinstitute.sharepoint.com/sites/Cores/SitePages/Acknowledgements-and-Authorship.aspx)

#### Contributors
<a href="https://github.com/vaioic/oic-toolkit/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=vaioic/oic-toolkit" />
</a>


## Development

### Setup

This project uses [``uv``](https://docs.astral.sh/uv/) to manage the development environment.

If you are modifying the toolkit while simultaneously running analysis scripts in another project, install it in **editable mode**.

1. Clone the repository
   ```bash
   git clone https://github.com/vaioic/oic-toolkit.git
   cd oic-toolkit
   ```

2. Sync the environment
   ```bash
   uv sync
   ```

3. Link this toolbox in your analysis project
   ```bash
   uv add --editable "path/to/oic-toolkit"
   ```

Note: You should change this to the published version when you are done.

### Code style and testing

This project also uses ``ruff`` for ultra-fast linting and code formatting, and ``pytest`` for unit tests.

```bash
# Run linting checks
uv run ruff check

# Auto-format codebase
uv run ruff format

# Run test suite
uv run pytest
```

## Changelog

### v0.1.0 (2027-07-02)
* Initial release with common segmentation and registration functions.