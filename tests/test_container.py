"""Tests for the in-container command factories and parsers."""

from __future__ import annotations

import re
import subprocess
import tarfile
from pathlib import Path

import pytest

from slopon_harbor.container import (
    RUNTIME_ROOT,
    RUNTIME_TAR_DEST,
    build_runtime_tar,
    gateway_discovery_command,
    node_download_url,
    node_provision_commands,
    parse_gateway_route,
    parse_node_version,
    runner_start_command,
    runner_start_env,
    runtime_unpack_command,
    validate_pinned_node_version,
)


class TestNodeVersionParsing:
    @pytest.mark.parametrize(
        ("output", "expected"),
        [
            ("v22.17.1\n", 22),
            ("v18.20.4\n", 18),
            ("v20.11.0", 20),
            ("22.17.1\n", 22),
            ("v10.24.1\n", 10),
            ("", None),
            ("garbage\n", None),
            ("v\n", None),
        ],
    )
    def test_parse_node_version(self, output, expected):
        assert parse_node_version(output) == expected

    def test_pinned_major_assertion(self):
        assert validate_pinned_node_version("22.17.1") == "v22.17.1"
        assert validate_pinned_node_version("v22.9.0") == "v22.9.0"
        with pytest.raises(ValueError, match="v22"):
            validate_pinned_node_version("24.1.0")
        with pytest.raises(ValueError, match="invalid"):
            validate_pinned_node_version("nan")


class TestNodeDownloadUrl:
    def test_exact_url_x86_64(self):
        assert node_download_url("x86_64", "22.17.1") == (
            "https://nodejs.org/dist/v22.17.1/node-v22.17.1-linux-x64.tar.gz"
        )

    def test_arch_mapping_aarch64(self):
        assert node_download_url("aarch64", "22.17.1") == (
            "https://nodejs.org/dist/v22.17.1/node-v22.17.1-linux-arm64.tar.gz"
        )

    def test_already_mapped_arches_pass_through(self):
        assert node_download_url("x64", "v22.17.1").endswith("linux-x64.tar.gz")
        assert node_download_url("arm64", "22.17.1").endswith("linux-arm64.tar.gz")

    def test_tarball_not_xz(self):
        assert node_download_url("x86_64", "22.17.1").endswith(".tar.gz")


class TestNodeProvisionScript:
    def script(self) -> str:
        return node_provision_commands("x86_64", "22.17.1")[0]

    def test_returns_single_command_list(self):
        commands = node_provision_commands("x86_64", "22.17.1")
        assert isinstance(commands, list) and len(commands) == 1

    def test_pinned_url_embedded(self):
        script = self.script()
        assert "https://nodejs.org/dist/v22.17.1/node-v22.17.1-linux-x64.tar.gz" in script
        assert ".tar.gz" in script and ".tar.xz" not in script

    def test_downloader_chain_order(self):
        script = self.script()
        curl_pos = script.index('DL="curl"')
        wget_pos = script.index('DL="wget"')
        python_pos = script.index('DL="python3"')
        assert curl_pos < wget_pos < python_pos

    def test_apt_bootstrap_branch(self):
        script = self.script()
        apt_pos = script.index(
            "apt-get update && apt-get install -y curl ca-certificates"
        )
        # The apt bootstrap happens only when no downloader was found...
        assert script.index('if [ -z "$DL" ]') < apt_pos
        # ...and installs curl BEFORE the download runs with it.
        assert apt_pos < script.index('curl -fsSL "$DIST"')

    def test_failure_messages_mention_remedy(self):
        script = self.script()
        assert "no downloader (curl/wget/python3)" in script
        assert "apt-get failed to bootstrap curl" in script
        assert "preinstall Node >= 20" in script

    def test_skip_when_system_node_recent_enough(self):
        script = self.script()
        assert 'if [ "$MAJ" -ge 22 ]; then NODE_BIN=$(command -v node); fi' in script

    def test_bash_syntax_valid(self):
        result = subprocess.run(
            ["bash", "-n"], input=self.script(), capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr


class TestGatewayParsing:
    ROUTE_TABLE = (
        "Iface\tDestination\tGateway\t\tFlags\tRefCnt\tUse\tMetric\tMask\t\t"
        "MTU\tWindow\tIRTT\n"
        "eth0\t00000000\t0101FEA9\t0003\t0\t0\t0\t00000000\t0\t0\t0\n"
        "eth0\t0101FEA9\t00000000\t0001\t0\t0\t1000\tFFFFFFFF\t0\t0\t0\n"
    )

    def test_hex_gateway_little_endian(self):
        assert parse_gateway_route(self.ROUTE_TABLE) == "169.254.1.1"

    def test_no_default_route(self):
        table = "Iface\tDestination\tGateway\tFlags\n" "eth0\t0101FEA9\t00000000\t0001\n"
        assert parse_gateway_route(table) is None

    def test_empty(self):
        assert parse_gateway_route("") is None

    def test_generated_command_is_valid_bash_and_awk(self):
        command = gateway_discovery_command()
        # Whole command parses as bash.
        result = subprocess.run(
            ["bash", "-n"], input=command, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        # The embedded awk program runs against a sample route table and
        # yields the gateway. Extract it (awk '... ' /proc/net/route) and
        # point it at a temp file instead.
        match = re.search(r"awk '((?:[^']|'\\'')*)' /proc/net/route", command)
        assert match, "awk program not found in gateway command"
        awk_program = match.group(1)
        result = subprocess.run(
            ["awk", awk_program],
            input=self.ROUTE_TABLE,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "169.254.1.1"


class TestCommandsCarryNoSecrets:
    def test_unpack_command(self):
        cmd = runtime_unpack_command()
        assert (
            cmd
            == f"mkdir -p {RUNTIME_ROOT} && tar -xzf {RUNTIME_TAR_DEST} -C {RUNTIME_ROOT}"
        )

    def test_start_command_has_no_url_or_token(self):
        cmd = runner_start_command("/tmp/slopon-runner/node/bin/node")
        assert "nohup /tmp/slopon-runner/node/bin/node" in cmd
        assert "/logs/agent/runner.log" in cmd
        assert cmd.rstrip().endswith("&")
        assert "ws://" not in cmd
        assert "http" not in cmd
        assert "TOKEN" not in cmd.upper()

    def test_credentials_live_in_env_dict(self):
        env = runner_start_env("ws://172.17.0.1:4200", "secret-token")
        assert env == {
            "RUNNER_URL": "ws://172.17.0.1:4200",
            "RUNNER_TOKEN": "secret-token",
        }


def make_runtime_tree(base: Path) -> Path:
    runtime = base / "runtime"
    (runtime / "prompts" / "deep").mkdir(parents=True)
    (runtime / "prompts" / "deep" / "system.md").write_text("sys", encoding="utf-8")
    (runtime / "node_modules" / "ws").mkdir(parents=True)
    (runtime / "node_modules" / "ws" / "index.js").write_text("ws", encoding="utf-8")
    (runtime / "runner.js").write_text("runner", encoding="utf-8")
    (runtime / "package.json").write_text("{}", encoding="utf-8")
    # Excluded decoys:
    (runtime / "index.js").write_text("server", encoding="utf-8")
    (runtime / "migrations").mkdir()
    (runtime / "migrations" / "001.sql").write_text("--", encoding="utf-8")
    return runtime


class TestBuildRuntimeTar:
    def test_member_selection_and_symlink_preservation(self, tmp_path):
        runtime = make_runtime_tree(tmp_path)
        # A symlink inside node_modules (type preservation): non-derefercing
        # add must store the LINK, not its target content.
        link = runtime / "node_modules" / ".bin" / "ws-cli"
        link.parent.mkdir(parents=True)
        link.symlink_to("../ws/index.js")

        dest = tmp_path / "runtime.tar.gz"
        result = build_runtime_tar(runtime, dest)
        assert result == dest and dest.is_file()

        extract = tmp_path / "extracted"
        with tarfile.open(dest) as tar:
            names = tar.getnames()
            tar.extractall(extract, filter="tar")

        # Included members.
        assert "runner.js" in names
        assert "package.json" in names
        assert "prompts/deep/system.md" in names
        assert "node_modules/ws/index.js" in names
        assert "node_modules/.bin/ws-cli" in names
        # Excluded members.
        assert "index.js" not in names
        assert not any(n.startswith("migrations") for n in names)

        extracted_link = extract / "node_modules" / ".bin" / "ws-cli"
        assert extracted_link.is_symlink()
        import os

        assert os.readlink(extracted_link) == "../ws/index.js"
        # The real file round-trips byte-exact.
        assert (
            extract / "node_modules" / "ws" / "index.js"
        ).read_text(encoding="utf-8") == "ws"
        # The symlink target relationship resolves post-extract.
        assert (extract / "node_modules" / ".bin" / "ws-cli").resolve() == (
            extract / "node_modules" / "ws" / "index.js"
        )

    def test_missing_members_raise(self, tmp_path):
        runtime = tmp_path / "empty-runtime"
        runtime.mkdir()
        (runtime / "runner.js").write_text("x", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="package.json"):
            build_runtime_tar(runtime, tmp_path / "out.tar.gz")
