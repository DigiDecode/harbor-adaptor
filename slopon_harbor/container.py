"""In-container provisioning: pure command/URL factories and parsers.

All commands are executed later via ``BaseEnvironment.exec``. Credentials
(the backend URL and the runner token) never appear in command strings —
they travel through the exec ``env`` dict so they stay out of process
argv (``ps``).
"""

from __future__ import annotations

import tarfile
from pathlib import Path

PINNED_NODE_MAJOR = 22
NODE_DIST_BASE = "https://nodejs.org/dist"

NODE_BIN_MARKER = "/tmp/slopon-runner/node-bin"
RUNTIME_TAR_DEST = "/tmp/slopon-runner/runtime.tar.gz"
RUNTIME_ROOT = "/tmp/slopon-runner/runtime"
RUNNER_LOG_PATH = "/logs/agent/runner.log"

# Deliberately terminal, actionable message when Node cannot be installed.
NODE_BOOTSTRAP_HINT = (
    "preinstall Node >= 20 in the task image, or allow egress to "
    "archive.ubuntu.com and nodejs.org during setup"
)


def parse_node_version(output: str) -> int | None:
    """Return the major version from ``node -v`` output, or None.

    Accepts ``v22.17.1`` / ``22.17.1`` / ``v22``; returns None for absent
    or unparseable output.
    """
    text = (output or "").strip()
    if text.startswith("v"):
        text = text[1:]
    head = text.split(".", maxsplit=1)[0]
    if head.isdigit():
        return int(head)
    return None


def validate_pinned_node_version(version: str) -> str:
    """Normalize and validate the pinned Node version string.

    Returns the normalized ``vX.Y.Z`` form. The major must equal
    ``PINNED_NODE_MAJOR`` — the backend only verifies Node LTS releases up
    to v22 (Node 24 is explicitly unverified).
    """
    text = (version or "").strip()
    if text.startswith("v"):
        text = text[1:]
    major = parse_node_version(f"v{text}")
    if major is None or "." not in text:
        raise ValueError(
            f"invalid SLOPON_NODE_VERSION {version!r}: expected e.g. '22.17.1'"
        )
    if major != PINNED_NODE_MAJOR:
        raise ValueError(
            f"SLOPON_NODE_VERSION {version!r} must pin Node v{PINNED_NODE_MAJOR}.x "
            "(the newest LTS verified by the SlopOn backend)"
        )
    return f"v{text}"


def node_download_url(arch: str, node_version: str) -> str:
    """Exact nodejs.org static-tarball URL for the pinned version."""
    node_arch = {"x86_64": "x64", "aarch64": "arm64", "x64": "x64", "arm64": "arm64"}[
        arch
    ]
    version = validate_pinned_node_version(node_version)
    return f"{NODE_DIST_BASE}/{version}/node-{version}-linux-{node_arch}.tar.gz"


def node_provision_commands(arch: str, node_version: str) -> list[str]:
    """One bash script ensuring a Node >= 22 runtime exists in-container.

    Uses the system node when its major is >= 22; otherwise downloads the
    pinned static ``.tar.gz`` (gzip/tar ship in base Ubuntu images while
    xz does not) using the first available downloader
    (curl -> wget -> python3), bootstrapping curl via apt-get when the
    image ships no downloader at all. Prints the resolved node binary path
    and records it in a marker file for later steps.
    """
    url = node_download_url(arch, node_version)
    script = f"""set -e
HINT='{NODE_BOOTSTRAP_HINT}'
NODE_BIN=""
if command -v node >/dev/null 2>&1; then
  MAJ=$(node -v 2>/dev/null | sed 's/^v//' | cut -d. -f1)
  case "$MAJ" in ''|*[!0-9]*) MAJ=0;; esac
  if [ "$MAJ" -ge {PINNED_NODE_MAJOR} ]; then NODE_BIN=$(command -v node); fi
fi
if [ -z "$NODE_BIN" ]; then
  DIST='{url}'
  TARBALL='/tmp/slopon-runner/node.tar.gz'
  mkdir -p /tmp/slopon-runner
  DL=""
  if command -v curl >/dev/null 2>&1; then DL="curl";
  elif command -v wget >/dev/null 2>&1; then DL="wget";
  elif command -v python3 >/dev/null 2>&1; then DL="python3";
  fi
  if [ -z "$DL" ]; then
    if ! command -v apt-get >/dev/null 2>&1; then
      echo "ERROR: no downloader (curl/wget/python3) and no apt-get" \
        "available to bootstrap one; $HINT" >&2
      exit 1
    fi
    apt-get update && apt-get install -y curl ca-certificates || {{
      echo "ERROR: apt-get failed to bootstrap curl; $HINT" >&2
      exit 1
    }}
    DL="curl"
  fi
  if [ "$DL" = "python3" ]; then
    python3 - "$DIST" "$TARBALL" <<'PYDL'
import sys, urllib.request
urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PYDL
  elif [ "$DL" = "curl" ]; then
    curl -fsSL "$DIST" -o "$TARBALL"
  else
    wget -qO "$TARBALL" "$DIST"
  fi
  mkdir -p /tmp/slopon-runner/node
  tar -xzf "$TARBALL" -C /tmp/slopon-runner/node --strip-components=1
  NODE_BIN=/tmp/slopon-runner/node/bin/node
fi
printf '%s' "$NODE_BIN" > {NODE_BIN_MARKER}
echo "$NODE_BIN\""""
    return [script]


def runtime_unpack_command() -> str:
    """Extract the uploaded runtime tarball into the runner root."""
    return f"mkdir -p {RUNTIME_ROOT} && tar -xzf {RUNTIME_TAR_DEST} -C {RUNTIME_ROOT}"


def _awk_proc_route() -> str:
    # Pure-awk hex math (index-based) so it also runs under mawk, which
    # lacks strtonum and treats "0x.." strings as 0.
    return (
        "awk 'function hx(c) { return index(\"0123456789abcdef\", tolower(c)) - 1 }\n"
        "$2 == \"00000000\" {\n"
        "  printf \"%d.%d.%d.%d\\n\", "
        "hx(substr($3,7,1))*16+hx(substr($3,8,1)), "
        "hx(substr($3,5,1))*16+hx(substr($3,6,1)), "
        "hx(substr($3,3,1))*16+hx(substr($3,4,1)), "
        "hx(substr($3,1,1))*16+hx(substr($3,2,1))\n"
        "  found=1; exit\n"
        "}\n"
        "END { if (!found) exit 1 }' /proc/net/route"
    )


def _awk_ip_route() -> str:
    return (
        "ip route show default 2>/dev/null | "
        "awk '{for(i=1;i<=NF;i++) if($i==\"via\"){print $(i+1); exit}}'"
    )


def gateway_discovery_command() -> str:
    """Print the container's default-route gateway IP.

    Parses the little-endian hex gateway column of ``/proc/net/route``
    with awk, falling back to ``ip route show default``. Prints an empty
    line when neither source yields a gateway.
    """
    return "\n".join(
        [
            f"GW=$( {_awk_proc_route()} 2>/dev/null ) || GW=\"\"",
            "if [ -z \"$GW\" ] && command -v ip >/dev/null 2>&1; then",
            f"  GW=$( {_awk_ip_route()} )",
            "fi",
            "echo \"$GW\"",
        ]
    )


def runner_start_command(node_bin: str) -> str:
    """Start the runner in the background, logging to the mounted agent dir.

    The backend URL and runner token are deliberately absent: they are
    passed via the exec ``env`` dict (``RUNNER_URL`` / ``RUNNER_TOKEN``)
    and inherited by the nohup child, keeping them out of ``ps`` output.
    """
    return f"nohup {node_bin} {RUNTIME_ROOT}/runner.js >> {RUNNER_LOG_PATH} 2>&1 &"


def runner_start_env(backend_public_url: str, runner_token: str) -> dict[str, str]:
    """Exec-env dict carrying the runner credentials (never argv)."""
    return {"RUNNER_URL": backend_public_url, "RUNNER_TOKEN": runner_token}


def parse_gateway_route(proc_net_route: str) -> str | None:
    """Pure-Python mirror of the in-container gateway parser."""
    for line in (proc_net_route or "").splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[1] == "00000000":
            hex_gateway = fields[2].lower()
            if len(hex_gateway) != 8:
                continue
            try:
                octets = [int(hex_gateway[i : i + 2], 16) for i in range(6, -1, -2)]
            except ValueError:
                continue
            return ".".join(str(o) for o in octets)
    return None


def build_runtime_tar(runtime_dir: Path, dest: Path) -> Path:
    """Tar exactly the runner runtime into ``dest`` and return the path.

    Includes ``runner.js``, ``package.json``, ``prompts/`` and
    ``node_modules/`` (recursive, nothing else — ``index.js`` and
    ``migrations/`` stay behind). ``tarfile``'s default non-dereferencing
    adds preserve symlinks and every other file type byte-exact; a broken
    runtime otherwise only surfaces in-container as ERR_MODULE_NOT_FOUND,
    far from the cause.
    """
    members = ["runner.js", "package.json", "prompts", "node_modules"]
    missing = [name for name in members if not (runtime_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"runner runtime {runtime_dir} is missing {', '.join(missing)}"
        )
    # Level 1: the tree is hundreds of MB of already-minified JS — gzip's
    # extra levels cost far more CPU time than the bytes they save on a
    # localhost docker upload.
    with tarfile.open(dest, "w:gz", compresslevel=1) as tar:
        for name in members:
            tar.add(runtime_dir / name, arcname=name, recursive=True)
    return dest
