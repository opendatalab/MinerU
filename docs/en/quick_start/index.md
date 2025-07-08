# Quick Start

If you encounter any installation issues, please first consult the [FAQ](../FAQ/index.md).


If the parsing results are not as expected, refer to the [Known Issues](../known_issues.md).


There are three different ways to experience MinerU:

- [Online Demo](online_demo.md)
- [Local Deployment](local_deployment.md)


> [!WARNING]
> **Pre-installation Notice—Hardware and Software Environment Support**
>
> To ensure the stability and reliability of the project, we only optimize and test for specific hardware and software environments during development. This ensures that users deploying and running the project on recommended system configurations will get the best performance with the fewest compatibility issues.
>
> By focusing resources on the mainline environment, our team can more efficiently resolve potential bugs and develop new features.
>
> In non-mainline environments, due to the diversity of hardware and software configurations, as well as third-party dependency compatibility issues, we cannot guarantee 100% project availability. Therefore, for users who wish to use this project in non-recommended environments, we suggest carefully reading the documentation and FAQ first. Most issues already have corresponding solutions in the FAQ. We also encourage community feedback to help us gradually expand support.

<table>
    <tr>
        <td>Parsing Backend</td>
        <td>pipeline</td>
        <td>vlm-transformers</td>
        <td>vlm-sglang</td>
    </tr>
    <tr>
        <td>Operating System</td>
        <td>windows/linux/mac</td>
        <td>windows/linux</td>
        <td>windows(wsl2)/linux</td>
    </tr>
    <tr>
        <td>CPU Inference Support</td>
        <td>✅</td>
        <td colspan="2">❌</td>
    </tr>
    <tr>
        <td>GPU Requirements</td>
        <td>Turing architecture or later, 6GB+ VRAM or Apple Silicon</td>
        <td colspan="2">Ampere architecture or later, 8GB+ VRAM</td>
    </tr>
    <tr>
        <td>Memory Requirements</td>
        <td colspan="3">Minimum 16GB+, 32GB+ recommended</td>
    </tr>
    <tr>
        <td>Disk Space Requirements</td>
        <td colspan="3">20GB+, SSD recommended</td>
    </tr>
    <tr>
        <td>Python Version</td>
        <td colspan="3">3.10-3.13</td>
    </tr>
</table>