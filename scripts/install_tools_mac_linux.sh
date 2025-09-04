#!/usr/bin/env bash
set -euo pipefail

# Interactive tool checker/installer for macOS and Linux
# It checks: Python, pip, Docker, kubectl, kind (optional).
# For system package installation, we provide guided suggestions and attempt installs
# on common package managers when available.

confirm() {
  read -r -p "$1 [y/N]: " ans || true
  case "${ans:-}" in
    [yY][eE][sS]|[yY]) return 0;;
    *) return 1;;
  esac
}

have() { command -v "$1" >/dev/null 2>&1; }

OS="unknown"
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)   OS=linux;;
    Darwin*)  OS=mac;;
    *)        OS=other;;
esac

echo "Detected OS: $OS"

pkg_install_linux() {
  # Try apt, dnf, yum, zypper, pacman
  if have apt-get; then sudo apt-get update && sudo apt-get install -y "$@"; return; fi
  if have dnf; then sudo dnf install -y "$@"; return; fi
  if have yum; then sudo yum install -y "$@"; return; fi
  if have zypper; then sudo zypper install -y "$@"; return; fi
  if have pacman; then sudo pacman -S --noconfirm "$@"; return; fi
  echo "No known package manager detected. Please install manually: $*"
}

# Python
if have python3; then echo "OK: python3 $(python3 --version)"; else
  echo "Missing: python3";
  if [ "$OS" = mac ] && have brew; then
    if confirm "Install Python via Homebrew?"; then brew install python; fi
  elif [ "$OS" = linux ]; then
    if confirm "Attempt to install python3 via package manager?"; then pkg_install_linux python3 python3-venv python3-pip || true; fi
  fi
fi

# pip
if have pip3; then echo "OK: pip3 $(pip3 --version)"; else
  echo "Missing: pip3";
fi

# Docker
if have docker; then echo "OK: docker $(docker --version)"; else
  echo "Missing: docker"
  if [ "$OS" = mac ] && have brew; then
    echo "You can install Docker Desktop: https://docs.docker.com/desktop/install/mac/"
    if confirm "Open Docker Desktop download page in browser?"; then open "https://docs.docker.com/desktop/install/mac/" || true; fi
  elif [ "$OS" = linux ]; then
    echo "Install Docker Engine: https://docs.docker.com/engine/install/"
  fi
fi

# kubectl
if have kubectl; then echo "OK: kubectl $(kubectl version --client --short 2>/dev/null || echo client-only)"; else
  echo "Missing: kubectl"
  if [ "$OS" = mac ] && have brew; then
    if confirm "Install kubectl via Homebrew?"; then brew install kubectl; fi
  elif [ "$OS" = linux ]; then
    if confirm "Attempt to install kubectl via package manager?"; then pkg_install_linux kubectl || true; fi
    echo "Official instructions: https://kubernetes.io/docs/tasks/tools/"
  fi
fi

# kind (optional)
if have kind; then echo "OK: kind $(kind version)"; else
  echo "Optional: kind not found. If you want local Kubernetes in Docker, install kind."
  if [ "$OS" = mac ] && have brew; then
    if confirm "Install kind via Homebrew?"; then brew install kind; fi
  elif [ "$OS" = linux ]; then
    echo "Install kind: https://kind.sigs.k8s.io/docs/user/quick-start/#installation"
  fi
fi

echo "\nNext suggestions:"
echo "- Run scripts/setup_python.sh to prepare the Python venv."
echo "- If you installed Docker, restart your shell and start Docker Desktop/daemon."
echo "- For local k8s with kind: kind create cluster && scripts/k8s_quickstart.sh"
