"""
Multi-Platform Native Application & Installer Builder for Smart AI.
Generates:
1. macOS: Native SmartAI.app Application Bundle & SmartAI-macOS-arm64.dmg Disk Image Installer
2. Windows: SmartAI-Windows package with SmartAI.bat, SmartAI.vbs (silent windowed launcher)
3. Linux/Unix: SmartAI.AppDir structure, SmartAI.desktop launcher, AppRun, and SmartAI-Linux-x86_64.tar.gz
"""

import os
import platform
import shutil
import stat
import subprocess
import sys


def create_macos_bundle(dist_dir: str, app_name: str):
    """Creates a native macOS .app bundle and compiles a .dmg disk image installer."""
    print(f"[*] Assembling macOS {app_name}.app Application Bundle...")
    app_bundle = os.path.join(dist_dir, f"{app_name}.app")
    contents_dir = os.path.join(app_bundle, "Contents")
    macos_dir = os.path.join(contents_dir, "MacOS")
    resources_dir = os.path.join(contents_dir, "Resources")
    app_code_dir = os.path.join(resources_dir, "app")

    # 1. Compile native macOS Mach-O executable applet via osacompile if available
    if shutil.which("osacompile"):
        applescript = f"""
on run
    set myPath to POSIX path of (path to me)
    set resDir to myPath & "Contents/Resources/app"
    set logFile to "/tmp/smartai_launch.log"
    set shScript to "export PATH=\\"/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/Library/Frameworks/Python.framework/Versions/3.12/bin:/Library/Frameworks/Python.framework/Versions/3.11/bin:/Library/Frameworks/Python.framework/Versions/3.10/bin:$HOME/.pyenv/shims:$HOME/.local/bin:$PATH\\"; cd " & quoted form of resDir & "; PYTHON_CMD=\\"\\"; for C in \\"$(which python3 2>/dev/null)\\" /opt/homebrew/bin/python3 /usr/local/bin/python3 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 \\"$HOME/.pyenv/shims/python3\\"; do if [ -n \\"$C\\" ] && [ -x \\"$C\\" ]; then if \\"$C\\" -c \\"import tkinter\\" 2>/dev/null; then PYTHON_CMD=\\"$C\\"; break; fi; fi; done; if [ -z \\"$PYTHON_CMD\\" ]; then PYTHON_CMD=\\"python3\\"; fi; \\"$PYTHON_CMD\\" app_gui.py > " & quoted form of logFile & " 2>&1 &"
    try
        do shell script shScript
    on error errMsg
        display dialog ("Could not launch Smart AI Studio: " & errMsg) buttons {{"OK"}} default button "OK" with icon stop
    end try
end run
"""
        if os.path.exists(app_bundle):
            shutil.rmtree(app_bundle, ignore_errors=True)
        subprocess.run(["osacompile", "-o", app_bundle, "-e", applescript], check=True)

    os.makedirs(macos_dir, exist_ok=True)
    os.makedirs(app_code_dir, exist_ok=True)

    # 2. Copy application source trees & icons
    for item in ["app_gui.py", "main.py", "config", "core", "memory", "consolidation", "app_icon.png", "AppIcon.icns"]:
        src = os.path.abspath(item)
        dst = os.path.join(app_code_dir, item)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        elif os.path.isfile(src):
            shutil.copy2(src, dst)

    if os.path.exists("AppIcon.icns"):
        shutil.copy2("AppIcon.icns", os.path.join(resources_dir, "AppIcon.icns"))
        shutil.copy2("AppIcon.icns", os.path.join(resources_dir, "applet.icns"))

    # 3. Write Info.plist
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>English</string>
    <key>CFBundleDisplayName</key>
    <string>Smart AI</string>
    <key>CFBundleExecutable</key>
    <string>applet</string>
    <key>CFBundleIconFile</key>
    <string>applet</string>
    <key>CFBundleIdentifier</key>
    <string>com.smartai.studio</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>{app_name}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>aplt</string>
    <key>CFBundleShortVersionString</key>
    <string>2.0.0</string>
    <key>CFBundleVersion</key>
    <string>2</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.developer-tools</string>
    <key>NSHumanReadableCopyright</key>
    <string>Copyright © 2026 Smart AI Studio Contributors. All rights reserved.</string>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
</dict>
</plist>
"""
    with open(os.path.join(contents_dir, "Info.plist"), "w") as f:
        f.write(plist_content)

    # 4. Clear macOS quarantine attributes
    if platform.system() == "Darwin":
        try:
            subprocess.run(["xattr", "-cr", app_bundle], capture_output=True)
        except Exception:
            pass

    print(f"[✓] macOS Application Bundle created at: {app_bundle}")

    # 4. Generate .dmg disk image installer
    if shutil.which("hdiutil"):
        dmg_path = os.path.join(dist_dir, f"{app_name}-macOS-arm64.dmg")
        if os.path.exists(dmg_path):
            os.remove(dmg_path)

        dmg_temp = os.path.join(dist_dir, "dmg_staging")
        if os.path.exists(dmg_temp):
            shutil.rmtree(dmg_temp, ignore_errors=True)
        os.makedirs(dmg_temp, exist_ok=True)
        shutil.copytree(app_bundle, os.path.join(dmg_temp, f"{app_name}.app"))
        
        # Copy quick launchers and helper scripts
        for helper in ["Launch_SmartAI_macOS.command", "Fix_Mac_Permissions.command", "README.md"]:
            helper_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), helper)
            if os.path.exists(helper_path):
                dest = os.path.join(dmg_temp, helper)
                shutil.copy2(helper_path, dest)
                if helper.endswith(".command"):
                    os.chmod(dest, 0o755)

        if os.path.exists(dmg_path):
            try:
                os.remove(dmg_path)
            except Exception:
                pass

        print(f"[*] Compiling DMG disk image installer: {dmg_path}...")
        cmd = [
            "hdiutil", "create",
            "-volname", f"{app_name}",
            "-srcfolder", dmg_temp,
            "-ov",
            "-format", "UDZO",
            dmg_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)

        # Create portable macOS zip archive from staging
        zip_path = os.path.join(dist_dir, f"{app_name}-macOS-arm64.zip")
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except Exception:
                pass
        shutil.make_archive(os.path.join(dist_dir, f"{app_name}-macOS-arm64"), 'zip', dmg_temp)
        shutil.rmtree(dmg_temp, ignore_errors=True)
        if os.path.exists(zip_path):
            zsize = os.path.getsize(zip_path) / (1024 * 1024)
            print(f"[✓] SUCCESS: macOS Portable Zip Bundle generated: {zip_path} ({zsize:.2f} MB)")


def create_windows_bundle(dist_dir: str, app_name: str):
    """Creates standalone Windows application package with .exe / .vbs launcher."""
    print(f"[*] Assembling Windows {app_name} standalone package...")
    win_dir = os.path.join(dist_dir, f"{app_name}-Windows")
    os.makedirs(win_dir, exist_ok=True)

    # Copy application source trees
    for item in ["app_gui.py", "main.py", "config", "core", "memory", "consolidation", "app_icon.png", "requirements.txt", "pyproject.toml"]:
        src = os.path.abspath(item)
        dst = os.path.join(win_dir, item)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        elif os.path.isfile(src):
            shutil.copy2(src, dst)

    # 1. Create Windows Batch Launcher (SmartAI.bat)
    bat_content = f"""@echo off
cd /d "%~dp0"
echo ======================================================================
echo   Launching {app_name}
echo ======================================================================
python app_gui.py %*
if %ERRORLEVEL% NEQ 0 (
    python3 app_gui.py %*
)
"""
    with open(os.path.join(win_dir, f"{app_name}.bat"), "w") as f:
        f.write(bat_content)

    # 2. Create Windowed Silent VBS Launcher (SmartAI.vbs)
    vbs_content = f"""Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "python app_gui.py", 0, False
"""
    with open(os.path.join(win_dir, f"{app_name}.vbs"), "w") as f:
        f.write(vbs_content)

    # 3. Create Windows Setup script
    setup_bat = f"""@echo off
echo Installing dependencies for {app_name}...
pip install -r requirements.txt
echo Setup Complete! Launch by double-clicking {app_name}.vbs or {app_name}.bat
pause
"""
    with open(os.path.join(win_dir, "Install_Dependencies.bat"), "w") as f:
        f.write(setup_bat)

    # Zip Windows package
    shutil.make_archive(win_dir, "zip", root_dir=dist_dir, base_dir=f"{app_name}-Windows")
    print(f"[✓] SUCCESS: Windows standalone package generated at: {win_dir}.zip")


def create_linux_bundle(dist_dir: str, app_name: str):
    """Creates Linux AppDir, desktop shortcut, AppRun, and tar.gz package."""
    print(f"[*] Assembling Linux {app_name} AppDir & standalone package...")
    linux_dir = os.path.join(dist_dir, f"{app_name}-Linux")
    appdir = os.path.join(linux_dir, f"{app_name}.AppDir")
    usr_bin = os.path.join(appdir, "usr", "bin")
    os.makedirs(usr_bin, exist_ok=True)

    # Copy application code into AppDir
    for item in ["app_gui.py", "main.py", "config", "core", "memory", "consolidation", "app_icon.png", "requirements.txt", "pyproject.toml"]:
        src = os.path.abspath(item)
        dst = os.path.join(usr_bin, item)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        elif os.path.isfile(src):
            shutil.copy2(src, dst)

    # 1. Create AppRun script
    apprun_content = f"""#!/bin/sh
HERE="$(dirname "$(readlink -f "${{0}}")")"
exec python3 "$HERE/usr/bin/app_gui.py" "$@"
"""
    apprun_path = os.path.join(appdir, "AppRun")
    with open(apprun_path, "w") as f:
        f.write(apprun_content)
    os.chmod(apprun_path, os.stat(apprun_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # 2. Create .desktop file
    desktop_content = f"""[Desktop Entry]
Name={app_name}
Exec=AppRun
Icon=smartai
Type=Application
Categories=Development;Science;ArtificialIntelligence;
Comment=Smart AI 27B Autonomous Reasoning System
Terminal=false
"""
    with open(os.path.join(appdir, f"{app_name}.desktop"), "w") as f:
        f.write(desktop_content)

    # 3. Create root launcher in linux_dir
    launcher_sh = f"""#!/usr/bin/env bash
DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR/{app_name}.AppDir"
./AppRun "$@"
"""
    launcher_path = os.path.join(linux_dir, f"{app_name}.sh")
    with open(launcher_path, "w") as f:
        f.write(launcher_sh)
    os.chmod(launcher_path, os.stat(launcher_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # 4. Create Linux/Unix Dependency Setup script
    install_sh = f"""#!/usr/bin/env bash
echo "Installing Python dependencies for {app_name} on Unix/Linux..."
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
echo "Dependencies installed successfully!"
echo "Launch with: ./{app_name}.sh"
"""
    install_path = os.path.join(linux_dir, "install_dependencies.sh")
    with open(install_path, "w") as f:
        f.write(install_sh)
    os.chmod(install_path, os.stat(install_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # Create Linux x86_64 tar.gz archive
    tar_path = shutil.make_archive(
        base_name=os.path.join(dist_dir, f"{app_name}-Linux-x86_64"),
        format="gztar",
        root_dir=dist_dir,
        base_dir=f"{app_name}-Linux"
    )
    print(f"[✓] SUCCESS: Linux standalone package generated: {tar_path}")

    # Create Generic Unix / POSIX tar.gz archive (BSD / Solaris / Generic Unix)
    unix_tar_path = shutil.make_archive(
        base_name=os.path.join(dist_dir, f"{app_name}-Unix-POSIX"),
        format="gztar",
        root_dir=dist_dir,
        base_dir=f"{app_name}-Linux"
    )
    print(f"[✓] SUCCESS: Unix/POSIX standalone package generated: {unix_tar_path}")


def main():
    dist_dir = os.path.abspath("dist")
    app_name = "SmartAI"

    if os.path.exists(dist_dir):
        shutil.rmtree(dist_dir)
    os.makedirs(dist_dir, exist_ok=True)

    print("=" * 75)
    print("  SMART AI: MULTI-PLATFORM PACKAGING (DMG, EXE/ZIP, APPIMAGE/TAR)")
    print("=" * 75 + "\n")

    # 1. macOS .app & .dmg
    create_macos_bundle(dist_dir, app_name)

    # 2. Windows package
    create_windows_bundle(dist_dir, app_name)

    # 3. Linux package
    create_linux_bundle(dist_dir, app_name)

    print("\n" + "=" * 75)
    print("  ALL SMART AI PACKAGES GENERATED SUCCESSFULLY")
    print(f"  Target directory: {dist_dir}")
    for item in sorted(os.listdir(dist_dir)):
        p = os.path.join(dist_dir, item)
        size_str = f"{os.path.getsize(p)/(1024*1024):.2f} MB" if os.path.isfile(p) else "Directory"
        print(f"  ► {item:<35} [{size_str}]")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
