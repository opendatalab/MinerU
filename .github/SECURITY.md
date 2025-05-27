# Security and Disclosure Information Policy for the MinerU Project

The MinerU team and community take security bugs seriously. We appreciate your efforts to responsibly disclose your findings, and will make every effort to acknowledge your contributions.

## Supported Versions

The latest versions of MinerU are supported.

### Security

- Use of [HTTPS](https://en.wikipedia.org/wiki/HTTPS) for network communication
- Use of secure protocols for network communication (through the use of HTTPS)
- Up-to-date support for TLS/SSL (through the use of [OpenSSL](https://www.openssl.org/))
- Performance of TLS certificate verification by default before sending HTTP headers with private information (through the use of OpenSSL and HTTPS)
- Distribution of the software via cryptographically signed releases (on the [PyPI](https://pypi.org/)package repositories)
- Use of [GitHub](https://github.com/) Issues for vulnerability reporting and tracking

### Analysis

- Use of [Ruff](https://docs.astral.sh/ruff/), [Mypy](https://mypy.readthedocs.io/) and [Pytest](https://docs.pytest.org/en/7.2.x/) for Python code linting (static and dynamic analysers) on pull requests and builds
- Use of GitHub Issues for bug reporting and tracking

## Reporting a Vulnerability

If you think you've identified a security issue in an MinerU project repository, please DO NOT report the issue publicly via the GitHub issue tracker, etc.

Instead, send an email with as many details as possible to [moe@myhloli.com](mailto:deepsearch-core@zurich.ibm.com). This is a private mailing list for the maintainers team.

Please do not create a public issue.

### Security Vulnerability Response

Each report is acknowledged and analyzed by the core maintainers within 30 working days.

Any vulnerability information shared with core maintainers stays within the MinerU project and will not be disseminated to other projects unless it is necessary to get the issue fixed.

After the initial reply to your report, the security team will keep you informed of the progress towards a fix and full announcement, and may ask for additional information or guidance.

## Security Alerts

We will send announcements of security vulnerabilities and steps to remediate on the [MinerU announcements](https://github.com/opendatalab/MinerU/discussions/categories/announcements).
