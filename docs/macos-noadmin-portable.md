# macOS no-admin portable release (Apple Silicon)

The `latest` GitHub release provides a **separate no-admin portable kit**:

- `SmartAI-macOS-arm64-NoAdmin-Portable-Kit.zip`
- `SmartAI-macOS-arm64-NoAdmin-Portable-Kit.zip.sha256`

Unzip the kit to a user-owned folder, then double-click its
`Install-and-Launch-SmartAI-NoAdmin.command` or run it in Terminal with
`/bin/bash "<kit-folder>/Install-and-Launch-SmartAI-NoAdmin.command"`.
The helper uses the system `curl`, `osascript`, `shasum`, and `ditto` to fetch,
verify, and assemble the **existing** split Mac release ZIP. GitHub limits a
single asset to 2 GiB, so this option shares the standard archive's verified
parts rather than uploading a duplicate application archive.

The assembled archive contains a fully populated `SmartAI.app`, a bundled
Python 3.11 runtime with dependencies already installed. **It contains no
pretrained model weights.** The default tabs are Bonsai 2.5 and Bonsai 2.5 CRACK;
choose Download in the app to fetch the selected pinned model and its MTP head.
See [model sources](bonsai25-models.md).

The app extracts into `~/Applications/SmartAI-Portable/<release-sha>/SmartAI.app`
and starts as the ordinary user. No `/Applications` write permission, `sudo`,
Homebrew, Xcode, system Python or additional dependency installation is needed.
The first kit run downloads the shared release parts from GitHub. The app does
not download a model merely because it starts. After you download a model in
Studio, that model can be loaded offline. The parts are SHA-256 checked before the
ZIP is assembled, the assembled ZIP is checked again, and the explicit empty
model manifest and embedded Python readiness marker are validated before moving the app
into place. Existing versioned installs are left untouched.

**Storage:** Keep room for the archive parts, assembled ZIP, extracted app, and
any previous installation you retain. Model downloads need additional space;
only download the presets you intend to use. Studio switches between resident
models rather than loading both 27B presets together.

**macOS security:** The app is not notarized. If macOS displays a security
warning, verify the source and use its ordinary user-level Open controls.
The portable workflow does not disable Gatekeeper or require admin privileges.

**CI proof:** The packaging tests check ZIP script permissions, manifest integrity,
explicit model-free installation, and an offline synthetic install on an Apple
Silicon GitHub runner. The release workflow publishes the kit only after the
focused regression/native gates, relocated application smoke tests, and the
verified model-free ZIP pass. The full regression matrix remains a manual option.
Actual pretrained-model inference is not executed in the release builder.
