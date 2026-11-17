# macOS no-admin portable release (Apple Silicon)

The `latest` GitHub release provides a **separate no-admin portable kit**:

- `SmartAI-macOS-arm64-NoAdmin-Portable-Kit.zip`
- `SmartAI-macOS-arm64-NoAdmin-Portable-Kit.zip.sha256`

Unzip the kit to a user-owned folder, then double-click its
`Install-and-Launch-SmartAI-NoAdmin.command` or run it in Terminal with
`/bin/bash "<kit-folder>/Install-and-Launch-SmartAI-NoAdmin.command"`.
The helper uses the system `curl`, `osascript`, `shasum`, and `ditto` to fetch,
verify, and assemble the **existing** split Mac release ZIP. GitHub limits a
single asset to 2 GiB, so this option shares the standard archive's signed-off
model parts rather than doubling the ~18 GB model-bearing release storage.

The assembled archive contains a fully populated `SmartAI.app`, a bundled
Python 3.11 runtime with dependencies already installed, and **both** model
snapshots in `Contents/Resources/app/preloaded_models/`:

1. `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` (Bonsai 2, Qwen3.8-based)
2. `dealignai/Bonsai-2-27B-CRACK-Ternary-JANG` (CRACK, Qwen3.8-based)

The app extracts into `~/Applications/SmartAI-Portable/<release-sha>/SmartAI.app`
and starts as the ordinary user. No `/Applications` write permission, `sudo`,
Homebrew, Xcode, system Python or additional dependency installation is needed.
The first kit run downloads the shared release parts from GitHub; **once the
complete app has been assembled, neither model is downloaded at app launch**,
and subsequent launches work offline. The parts are SHA-256 checked before the
ZIP is assembled, the assembled ZIP is checked again, and both model directories
and the embedded Python readiness marker are validated before moving the app
into place. Existing versioned installs are left untouched.

**Storage:** Plan for at least 55 GB free space while downloading, assembling,
and extracting this large archive. On a Mac with limited unified memory,
loading both 27B models at once may not be possible; they remain available as
separate model choices.

**macOS security:** The app is not notarized. If macOS displays a security
warning, verify the source and use its ordinary user-level Open controls.
The portable workflow does not disable Gatekeeper or require admin privileges.

**CI proof:** `tests/unit/test_macos_noadmin_portable.py` checks the exact two
models and immutable revisions, ZIP script permissions, manifest integrity,
and an offline, synthetic, model-free install on an Apple Silicon GitHub runner.
The release workflow only publishes the kit after the existing full CI matrix,
relocated application smoke tests, and the verified full model-bundled ZIP pass.
Actual pretrained-model inference is not executed in the release builder.
