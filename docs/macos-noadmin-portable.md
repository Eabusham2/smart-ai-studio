# Mac installation without administrator access (Apple Silicon)

Download `Install-SmartAI-macOS-Full.command` from the current release. Quit an
existing SmartAI session, then double-click the installer or run:

```sh
/bin/bash Install-SmartAI-macOS-Full.command
```

The installer downloads the same `SmartAI-macOS-arm64-portable.zip` offered as a
direct download. It checks the archive against its embedded SHA-256, extracts the
complete app and runtime, and opens it. Fresh installations use
`~/Applications/SmartAI.app` without an administrator password. Existing copies
in the user Applications folder, `/Applications`, or the older versioned portable
folder are detected and updated in place so their saved-data location stays the
same. If several copies exist, choose the one to update. No separate kit,
reassembler, model archive, native-runtime ZIP, Homebrew, Xcode, system Python or
extra dependency installation is needed.

The existing app is retained as a dated backup. Chat and learning data in their
normal data directories are preserved. Repeating the installer for the same
release reuses the installed app and does not restart an active session. Future
updates can use the app's **↑ Update** button.

## Portable and offline use

For a portable copy, download `SmartAI-macOS-arm64-portable.zip`, extract it into
a user-owned folder, and open `SmartAI.app`. The app includes Python and installed
dependencies. Download the selected model inside Studio before using it offline;
no pretrained model weights are bundled or fetched just by opening the app.

The command installer retains verified downloads after an interrupted attempt.
Set `SMARTAI_FULL_OFFLINE=1` to require its cached archive and forbid downloads.
On success it removes only its own archive/cache staging, preserving the previous
app backup and unrelated files. A previously downloaded command is pinned to its
release: if a rolling release has changed, download the current command again.

## Another installation folder

Use an absolute destination ending in `/SmartAI.app`:

```sh
SMARTAI_FULL_TARGET="$HOME/My Apps/SmartAI.app" /bin/bash Install-SmartAI-macOS-Full.command
```

Choosing or updating `/Applications/SmartAI.app` may require administrator
credentials if that directory is not writable by your account. A fresh install
in the default user Applications folder avoids this requirement. An explicit
`SMARTAI_FULL_TARGET` selects that location instead of detecting existing copies.

Keep room for the downloaded ZIP, extracted app, retained backup and your chosen
models. The app is unsigned; use macOS's ordinary Open controls if it asks for
approval. The installer does not disable system security settings.

CI checks the canonical archive, relocated first launch and Setup's private
dependency installation. Mac-specific synthetic installation tests also exercise
offline installation, upgrade, owned-cache cleanup and rollback after a failed
move. These packaging checks do not load a pretrained model.
