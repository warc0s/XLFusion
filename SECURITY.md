# Security Policy

## Supported Versions

XLFusion is an early-stage public OSS project maintained by a primary maintainer. Security fixes are targeted at the current release line on `main`.

| Version | Supported |
| --- | --- |
| 2.4.x | Yes |
| Earlier versions | No |

## Reporting a Vulnerability

Please report security issues privately through GitHub Security Advisories:

https://github.com/warc0s/XLFusion/security/advisories/new

If advisories are unavailable, contact the maintainer through the repository owner profile and avoid posting exploit details in a public issue.

Include:

- affected XLFusion version or commit
- Python version and operating system
- command, GUI flow, or batch YAML involved
- minimal reproduction steps
- impact and expected severity
- sanitized logs or traceback

Do not attach proprietary checkpoints, LoRAs, generated images, local config files with secrets, or large binary artifacts. Use generic model names, tensor-shape notes, or partial hashes instead.

## Security-Relevant Areas

The main risk areas are:

- path handling for workspace inputs, outputs, presets, and metadata recovery
- YAML and configuration parsing
- metadata recovery from local folders
- loading external `.safetensors` files
- GUI file selection and user-selected paths
- generated artifacts such as merged checkpoints, metadata folders, logs, and recovered batch YAML

XLFusion does not intentionally execute code from model files. Still, users should treat external checkpoints and LoRAs as untrusted data, keep them outside version control, and only load files from sources they are willing to trust.

## Disclosure Expectations

Please give the maintainer reasonable time to reproduce and fix the issue before public disclosure. Security fixes should include a regression test when a synthetic fixture can cover the behavior without large files.
