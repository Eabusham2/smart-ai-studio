#!/usr/bin/env python3
"""
Generates high-resolution AppIcon.icns and PNG assets for Smart AI Studio.
Creates a glowing neural spark / quantum brain icon using pure Python + zlib,
then packages it via macOS iconutil.
"""

import math
import os
import shutil
import struct
import subprocess
import zlib


def create_png_image(width: int, height: int) -> bytes:
    """Generates a raw RGBA PNG image with a futuristic glowing AI logo."""
    # Generate pixel matrix
    pixels = bytearray()
    cx, cy = width / 2.0, height / 2.0
    radius = width * 0.42

    for y in range(height):
        # Filter type byte (0 = None) for each scanline in PNG
        pixels.append(0)
        for x in range(width):
            dx = x - cx
            dy = y - cy
            dist = math.sqrt(dx * dx + dy * dy)
            angle = math.atan2(dy, dx)

            # Background: Rounded obsidian shield / circle
            # Check rounded square bounds
            corner_r = width * 0.22
            box_half = width * 0.44
            in_box = (
                abs(dx) <= box_half and abs(dy) <= box_half
                and not (
                    abs(dx) > (box_half - corner_r)
                    and abs(dy) > (box_half - corner_r)
                    and math.hypot(abs(dx) - (box_half - corner_r), abs(dy) - (box_half - corner_r)) > corner_r
                )
            )

            if in_box:
                # Radial gradient background: #0D1424 to #06080F
                norm_d = min(1.0, dist / (width * 0.55))
                bg_r = int(13 * (1 - norm_d) + 6 * norm_d)
                bg_g = int(20 * (1 - norm_d) + 8 * norm_d)
                bg_b = int(36 * (1 - norm_d) + 15 * norm_d)

                # Glowing center neural node / star
                # 4-point star formula: r = r0 / (cos(2*theta)^2 + sin(2*theta)^2)
                star_factor = math.pow(abs(math.cos(angle * 2)), 3) + math.pow(abs(math.sin(angle * 2)), 3)
                spark_dist = dist / (width * 0.38)
                is_core = dist < (width * 0.16)
                is_spark = (dist < width * 0.32) and (spark_dist < (0.35 + 0.65 * star_factor))

                # Ring of synapses
                ring_dist = abs(dist - width * 0.28)
                is_ring = ring_dist < (width * 0.02)

                if is_core:
                    # Pure Electric Cyan / White Core
                    core_f = 1.0 - (dist / (width * 0.16))
                    r = int(bg_r * (1 - core_f) + 255 * core_f)
                    g = int(bg_g * (1 - core_f) + 255 * core_f)
                    b = int(bg_b * (1 - core_f) + 255 * core_f)
                elif is_spark:
                    # Emerald to Cyan Glow
                    glow_f = max(0.0, 1.0 - (dist / (width * 0.32)))
                    r = int(bg_r * (1 - glow_f) + 0 * glow_f)
                    g = int(bg_g * (1 - glow_f) + 210 * glow_f)
                    b = int(bg_b * (1 - glow_f) + 255 * glow_f)
                elif is_ring:
                    # Orbital Synapse Ring (Emerald)
                    ring_f = 1.0 - (ring_dist / (width * 0.02))
                    r = int(bg_r * (1 - ring_f) + 16 * ring_f)
                    g = int(bg_g * (1 - ring_f) + 185 * ring_f)
                    b = int(bg_b * (1 - ring_f) + 129 * ring_f)
                else:
                    # Cyan border accent
                    border_dist = min(
                        box_half - abs(dx),
                        box_half - abs(dy)
                    )
                    if border_dist < (width * 0.025):
                        b_f = 1.0 - (border_dist / (width * 0.025))
                        r = int(bg_r * (1 - b_f) + 0 * b_f)
                        g = int(bg_g * (1 - b_f) + 210 * b_f)
                        b = int(bg_b * (1 - b_f) + 255 * b_f)
                    else:
                        r, g, b = bg_r, bg_g, bg_b

                pixels.extend([r, g, b, 255])
            else:
                # Transparent outside rounded box
                pixels.extend([0, 0, 0, 0])

    # Construct PNG binary structure
    # 1. Signature
    png = b"\x89PNG\r\n\x1a\n"

    # 2. IHDR Chunk
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    png += struct.pack(">I", 13) + b"IHDR" + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF)

    # 3. IDAT Chunk (compressed image data)
    compressed = zlib.compress(bytes(pixels), level=9)
    png += struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF)

    # 4. IEND Chunk
    png += struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    return png


def generate_iconset():
    print("[*] Generating high-resolution Smart AI application icons...")
    iconset_dir = "SmartAI.iconset"
    if os.path.exists(iconset_dir):
        shutil.rmtree(iconset_dir)
    os.makedirs(iconset_dir, exist_ok=True)

    sizes = [
        (16, "icon_16x16.png"),
        (32, "icon_16x16@2x.png"),
        (32, "icon_32x32.png"),
        (64, "icon_32x32@2x.png"),
        (128, "icon_128x128.png"),
        (256, "icon_128x128@2x.png"),
        (256, "icon_256x256.png"),
        (512, "icon_256x256@2x.png"),
        (512, "icon_512x512.png"),
        (1024, "icon_512x512@2x.png"),
    ]

    for sz, filename in sizes:
        png_data = create_png_image(sz, sz)
        filepath = os.path.join(iconset_dir, filename)
        with open(filepath, "wb") as f:
            f.write(png_data)

    # Save primary 512x512 app_icon.png in workspace
    master_png = create_png_image(512, 512)
    with open("app_icon.png", "wb") as f:
        f.write(master_png)
    print("[✓] Generated master app_icon.png (512x512)")

    # Build macOS AppIcon.icns
    if shutil.which("iconutil"):
        cmd = ["iconutil", "-c", "icns", iconset_dir, "-o", "AppIcon.icns"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists("AppIcon.icns"):
            size_kb = os.path.getsize("AppIcon.icns") / 1024
            print(f"[✓] Generated native macOS AppIcon.icns ({size_kb:.1f} KB)")
        else:
            print(f"[!] iconutil warning: {res.stderr}")

    shutil.rmtree(iconset_dir, ignore_errors=True)


if __name__ == "__main__":
    generate_iconset()
