#!/usr/bin/env bash
# Local Mendix Agent - user-level setup for Linux (counterpart of setup.ps1). No root, nothing
# installed system-wide, PATH is not modified. Everything lives in this folder (.venv, data/).
#
#   ./setup.sh                            create .venv and install dependencies from PyPI
#   ./setup.sh --wheelhouse ./wheelhouse  offline install from a folder of wheels
#   ./setup.sh --index-url https://nexus.company/pypi/simple   use a corporate PyPI mirror
#   ./setup.sh --python /usr/bin/python3.12                    use a specific interpreter
#   ./setup.sh --no-models                only the Python environment, download nothing else
#
# Unless --no-models is given, setup then downloads the llama.cpp runtime and one chat, vision and
# embedding model picked for this machine (about 6-9 GB, into data/). Roles that already have a
# model are skipped, so running setup again downloads nothing twice.
# Proxy: export HTTPS_PROXY before running if your network requires it.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

red() { printf '\033[31m%s\033[0m\n' "$*" >&2; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }

wheelhouse="" index_url="" python="" no_models=0
while [ $# -gt 0 ]; do
  case "$1" in
    --wheelhouse) wheelhouse="${2:?--wheelhouse needs a folder}"; shift 2 ;;
    --index-url)  index_url="${2:?--index-url needs a URL}"; shift 2 ;;
    --python)     python="${2:?--python needs a path}"; shift 2 ;;
    --no-models)  no_models=1; shift ;;
    -h|--help)    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) red "unknown option: $1 (see ./setup.sh --help)"; exit 2 ;;
  esac
done

find_python() {
  local c
  for c in ${python:+"$python"} python3.12 python3.13 python3.11 python3 python; do
    if command -v "$c" >/dev/null 2>&1 &&
       "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
      command -v "$c"; return 0
    fi
    [ -n "$python" ] && return 1   # an explicit --python is not silently replaced
  done
  return 1
}

if ! py="$(find_python)"; then
  red "Python 3.11+ was not found${python:+ at $python}."
  echo "Install it with your package manager (e.g. sudo apt install python3 python3-venv),"
  echo "or without root via https://github.com/astral-sh/uv or pyenv, then pass --python <path>."
  exit 1
fi
echo "Using Python: $py ($("$py" -c 'import platform; print(platform.python_version())'))"

venv_py=".venv/bin/python"
if [ ! -x "$venv_py" ]; then
  echo "Creating virtual environment in .venv ..."
  if "$py" -c 'import ensurepip' 2>/dev/null; then
    "$py" -m venv .venv || { red "venv creation failed"; exit 1; }
  else
    # Debian/Ubuntu ship ensurepip in the separate python3-venv package; work without it (no root)
    "$py" -m venv --without-pip .venv || { red "venv creation failed"; exit 1; }
  fi
fi

if ! "$venv_py" -c 'import pip' 2>/dev/null; then
  if "$venv_py" -c 'import ensurepip' 2>/dev/null; then
    echo "pip is missing in .venv, bootstrapping it (ensurepip) ..."
    "$venv_py" -m ensurepip --upgrade
  elif [ -z "$wheelhouse" ]; then
    echo "pip is missing in .venv, bootstrapping it (get-pip.py from bootstrap.pypa.io) ..."
    tmp="$(mktemp)"
    "$venv_py" -c 'import sys, urllib.request; urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", sys.argv[1])' "$tmp" &&
      "$venv_py" "$tmp" --disable-pip-version-check ${index_url:+--index-url "$index_url"}
    rm -f "$tmp"
  fi
  if ! "$venv_py" -c 'import pip' 2>/dev/null; then
    red "could not bootstrap pip in .venv."
    ver="$("$py" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    echo "Install the venv package (sudo apt install python${ver}-venv), delete .venv and run setup again."
    exit 1
  fi
fi

pip_args=(-m pip install --disable-pip-version-check -r requirements.txt)
if [ -n "$wheelhouse" ]; then
  [ -d "$wheelhouse" ] || { red "wheelhouse folder not found: $wheelhouse"; exit 1; }
  pip_args+=(--no-index --find-links "$(cd "$wheelhouse" && pwd)")
  echo "Installing dependencies offline from $wheelhouse ..."
elif [ -n "$index_url" ]; then
  pip_args+=(--index-url "$index_url")
  echo "Installing dependencies from $index_url ..."
else
  echo "Installing dependencies from PyPI ..."
fi
if ! "$venv_py" "${pip_args[@]}"; then
  echo
  red "Dependency installation failed. Typical causes:"
  echo " - pypi.org is blocked: use --index-url <corporate mirror> or --wheelhouse <folder>"
  echo "   (build the folder elsewhere:  pip download -r requirements.txt -d wheelhouse)"
  echo " - proxy required: export HTTPS_PROXY=http://proxy.company:8080"
  echo " - TLS inspection: export PIP_CERT=/path/to/company-root-ca.pem"
  exit 1
fi

if [ "$no_models" -eq 0 ]; then
  echo
  echo "Downloading the llama.cpp runtime and models for this machine (skip with --no-models) ..."
  if ! "$venv_py" main.py --yes --no-mcp -c "/download auto"; then
    yellow "Model download did not finish. Run it again later:  ./run.sh --yes --no-mcp -c \"/download auto\""
  fi
fi

echo
green "Setup complete."
echo "Start:        ./run.sh             (or .venv/bin/python main.py)"
echo "Diagnostics:  ./run.sh -c /doctor"
echo "First steps inside the app:  /models   /download <model-id>   /mendix connect"
