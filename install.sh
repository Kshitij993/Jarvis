#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo " Robotics Project - Linux Installer"
echo "============================================================"
echo ""

# ── Helper: install Python via the system package manager ────
install_python() {
    if command -v apt-get &>/dev/null; then
        echo "[INFO] Installing python3 via apt ..."
        sudo apt-get update -qq
        sudo apt-get install -y python3 python3-pip python3-venv
    elif command -v dnf &>/dev/null; then
        echo "[INFO] Installing python3 via dnf ..."
        sudo dnf install -y python3 python3-pip
    elif command -v pacman &>/dev/null; then
        echo "[INFO] Installing python3 via pacman ..."
        sudo pacman -Sy --noconfirm python python-pip
    else
        echo "[ERROR] No supported package manager found (apt / dnf / pacman)."
        echo "        Install Python 3.8+ manually and re-run this script."
        exit 1
    fi
}

# ── Check Python ──────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "[WARN] python3 not found. Attempting automatic installation ..."
    install_python
    if ! command -v python3 &>/dev/null; then
        echo "[ERROR] python3 installation failed. Please install it manually."
        exit 1
    fi
    echo "[OK] Python installed successfully."
fi

PY_VER=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "[OK] Found Python $PY_VER"

# ── Require Python 3.8+ ───────────────────────────────────────
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 8 ]; }; then
    echo "[ERROR] Python 3.8 or higher is required (found $PY_VER)."
    exit 1
fi

# ── Install system dependencies (PortAudio for PyAudio) ───────
echo ""
echo "[INFO] Checking for system-level audio dependencies ..."
if command -v apt-get &>/dev/null; then
    echo "[INFO] Detected apt — installing portaudio19-dev ..."
    sudo apt-get update -qq
    sudo apt-get install -y portaudio19-dev python3-dev build-essential
elif command -v dnf &>/dev/null; then
    echo "[INFO] Detected dnf — installing portaudio-devel ..."
    sudo dnf install -y portaudio-devel python3-devel gcc
elif command -v pacman &>/dev/null; then
    echo "[INFO] Detected pacman — installing portaudio ..."
    sudo pacman -Sy --noconfirm portaudio
else
    echo "[WARN] Unknown package manager. Make sure PortAudio dev headers are installed"
    echo "       before PyAudio can build correctly."
fi

# ── Create virtual environment ────────────────────────────────
if [ ! -d "venv" ]; then
    echo ""
    echo "[INFO] Creating virtual environment in ./venv ..."
    python3 -m venv venv
    echo "[OK] Virtual environment created."
else
    echo "[INFO] Virtual environment already exists, skipping creation."
fi

# ── Activate venv ─────────────────────────────────────────────
# shellcheck source=/dev/null
source venv/bin/activate

# ── Upgrade pip ───────────────────────────────────────────────
echo ""
echo "[INFO] Upgrading pip ..."
pip install --upgrade pip --quiet
echo "[OK] pip upgraded."

# ── Install project requirements ──────────────────────────────
echo ""
echo "[INFO] Installing project requirements from requirements.txt ..."
pip install -r requirements.txt

echo ""
echo "============================================================"
echo " Installation complete!"
echo ""
echo " To activate the environment in a new terminal run:"
echo "     source venv/bin/activate"
echo "============================================================"
