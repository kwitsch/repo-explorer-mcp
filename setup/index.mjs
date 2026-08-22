#!/usr/bin/env node
// Zero-dependency ESM installer for repo-explorer-mcp.
// Node builtins only + global fetch (Node >= 18).

import os from "node:os";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { spawnSync } from "node:child_process";
import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";

const OWNER = "kwitsch";
const REPO = "repo-explorer-mcp";
const BINARY_BASE = "repo-explorer-mcp";

// ---- pure helpers (exported for tests) ----

export function parseArgs(argv) {
  const out = { yes: false, force: false, version: null, help: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--yes" || a === "-y") out.yes = true;
    else if (a === "--force") out.force = true;
    else if (a === "--help" || a === "-h") out.help = true;
    else if (a === "--version") {
      out.version = argv[++i] ?? null;
    } else if (a.startsWith("--version=")) {
      out.version = a.slice("--version=".length);
    }
  }
  return out;
}

export function detectPlatform(platform, arch) {
  if (platform === "linux" && arch === "x64") {
    return {
      target: "x86_64-unknown-linux-gnu",
      ext: "tar.gz",
      osKind: "linux",
    };
  }
  if (platform === "win32" && arch === "x64") {
    return { target: "x86_64-pc-windows-msvc", ext: "zip", osKind: "win32" };
  }
  throw new Error(
    `Unsupported platform/arch: ${platform}/${arch}. ` +
      `Supported: linux/x64 (x86_64-unknown-linux-gnu), win32/x64 (x86_64-pc-windows-msvc).`,
  );
}

export function buildArchiveName({ binaryBaseName, version, target, ext }) {
  return `${binaryBaseName}-${version}-${target}.${ext}`;
}

export function buildDownloadUrls({ owner, repo, version, archiveName }) {
  const base = `https://github.com/${owner}/${repo}/releases/download/v${version}/`;
  const archiveUrl = base + archiveName;
  return { archiveUrl, sha256Url: archiveUrl + ".sha256" };
}

export function parseSha256Line(text) {
  return text.trim().split(/\s+/)[0];
}

export function hexEqual(a, b) {
  return a.trim().toLowerCase() === b.trim().toLowerCase();
}

export function binaryName(osKind) {
  return osKind === "win32" ? `${BINARY_BASE}.exe` : BINARY_BASE;
}

export function installDir(osKind, env, homedir) {
  if (osKind === "win32") {
    const local = env.LOCALAPPDATA;
    if (!local)
      throw new Error(
        "LOCALAPPDATA is not set; cannot determine the Windows install directory.",
      );
    return path.join(local, "repo-explorer-mcp");
  }
  return path.join(homedir, ".local", "bin");
}

export function isDirOnPath(dir, pathEnv, delimiter) {
  if (!pathEnv) return false;
  const norm = (p) => path.normalize(p).replace(/[\\/]+$/, "");
  const want = norm(dir);
  return pathEnv.split(delimiter).some((p) => p && norm(p) === want);
}

// ---- impure helpers (not unit-tested; exercised via smoke test) ----

function onPathTool(bin, osKind) {
  const finder = osKind === "win32" ? "where" : "which";
  const r = spawnSync(finder, [bin], { encoding: "utf8" });
  return r.status === 0;
}

function probeVersion(bin, args = ["--version"]) {
  const r = spawnSync(bin, args, { encoding: "utf8", timeout: 5000 });
  if (r.status === 0 && typeof r.stdout === "string") return r.stdout.trim();
  return null;
}

async function prompt(question, { yes }) {
  if (yes) return true;
  if (!input.isTTY) {
    throw new Error(
      `Confirmation required for: ${question}\nstdin is not a TTY; re-run with --yes to proceed non-interactively.`,
    );
  }
  const rl = createInterface({ input, output });
  try {
    const ans = (await rl.question(`${question} [y/N] `)).trim().toLowerCase();
    return ans === "y" || ans === "yes";
  } finally {
    rl.close();
  }
}

async function resolveVersion(pinned) {
  if (pinned) return pinned.replace(/^v/, "");
  const url = `https://api.github.com/repos/${OWNER}/${REPO}/releases/latest`;
  const res = await fetch(url, {
    headers: {
      "User-Agent": "repo-explorer-mcp-setup",
      Accept: "application/vnd.github+json",
    },
  });
  if (!res.ok) {
    throw new Error(
      `Failed to resolve latest release from ${url} (HTTP ${res.status}). Pin a version with --version <x.y.z>.`,
    );
  }
  const json = await res.json();
  if (!json.tag_name)
    throw new Error(`GitHub API response from ${url} had no tag_name.`);
  return String(json.tag_name).replace(/^v/, "");
}

async function downloadTo(url, destFile) {
  const res = await fetch(url, {
    headers: { "User-Agent": "repo-explorer-mcp-setup" },
  });
  if (!res.ok) throw new Error(`Download failed: ${url} (HTTP ${res.status}).`);
  const buf = Buffer.from(await res.arrayBuffer());
  fs.writeFileSync(destFile, buf);
  return buf;
}

function sha256Hex(buf) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}

function extractArchive(osKind, archiveFile, destDir) {
  if (osKind === "win32") {
    const r = spawnSync(
      "powershell",
      [
        "-NoProfile",
        "-Command",
        `Expand-Archive -LiteralPath '${archiveFile}' -DestinationPath '${destDir}' -Force`,
      ],
      { stdio: "inherit" },
    );
    if (r.status !== 0)
      throw new Error(
        "Expand-Archive failed while extracting the release zip.",
      );
  } else {
    const r = spawnSync("tar", ["-xzf", archiveFile, "-C", destDir], {
      stdio: "inherit",
    });
    if (r.status !== 0)
      throw new Error("tar failed while extracting the release archive.");
  }
}

function detectLinuxPkgManager() {
  for (const pm of ["apt-get", "dnf", "pacman"]) {
    if (onPathTool(pm, "linux")) return pm;
  }
  return null;
}

function installCommandFor(pm) {
  switch (pm) {
    case "apt-get":
      return ["sudo", ["apt-get", "install", "-y", "ripgrep"]];
    case "dnf":
      return ["sudo", ["dnf", "install", "-y", "ripgrep"]];
    case "pacman":
      return ["sudo", ["pacman", "-S", "--noconfirm", "ripgrep"]];
    default:
      return null;
  }
}

async function ensureRipgrep(plat, opts) {
  if (onPathTool("rg", plat.osKind)) {
    return probeVersion("rg") ?? "present";
  }
  if (plat.osKind === "linux") {
    const pm = detectLinuxPkgManager();
    if (pm) {
      const [cmd, args] = installCommandFor(pm);
      console.log(
        `ripgrep is missing. Proposed install: ${cmd} ${args.join(" ")}`,
      );
      if (await prompt("Install ripgrep now?", opts)) {
        const r = spawnSync(cmd, args, { stdio: "inherit" });
        if (r.status !== 0)
          console.warn(
            "ripgrep install failed; continuing (search still works via rtk if present).",
          );
      }
      return onPathTool("rg", plat.osKind)
        ? (probeVersion("rg") ?? "present")
        : "missing";
    }
    console.warn(
      "ripgrep is missing and no supported package manager (apt-get/dnf/pacman) was found. Install ripgrep manually: https://github.com/BurntSushi/ripgrep#installation",
    );
    return "missing";
  }
  // win32
  if (onPathTool("winget", plat.osKind)) {
    console.log(
      "ripgrep is missing. Proposed install: winget install --id BurntSushi.ripgrep.MSVC",
    );
    if (await prompt("Install ripgrep now?", opts)) {
      const r = spawnSync(
        "winget",
        [
          "install",
          "--id",
          "BurntSushi.ripgrep.MSVC",
          "--accept-source-agreements",
          "--accept-package-agreements",
        ],
        { stdio: "inherit" },
      );
      if (r.status !== 0) console.warn("ripgrep install failed; continuing.");
    }
    return onPathTool("rg", plat.osKind)
      ? (probeVersion("rg") ?? "present")
      : "missing";
  }
  console.warn(
    "ripgrep is missing and winget was not found. Install ripgrep manually: https://github.com/BurntSushi/ripgrep#installation",
  );
  return "missing";
}

function reportReportOnlyDep(name, plat, pointer) {
  if (onPathTool(name, plat.osKind)) {
    const v = probeVersion(name) ?? probeVersion(name, ["--help"]) ?? "present";
    return v.split("\n")[0];
  }
  console.warn(
    `${name} is missing. This installer does not auto-install it. ${pointer}`,
  );
  return "missing";
}

function mcpSnippet(osKind, binPath) {
  const cmd = binPath.replace(/\\/g, "\\\\");
  return JSON.stringify(
    {
      mcpServers: {
        "repo-explorer": { command: cmd, args: [], env: {} },
      },
    },
    null,
    2,
  );
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.help) {
    console.log(
      [
        "repo-explorer-mcp-setup — install the repo-explorer-mcp MCP server.",
        "",
        "Usage: npx github:kwitsch/repo-explorer-mcp [options]",
        "",
        "Options:",
        "  -y, --yes             Non-interactive; auto-approve installs.",
        "  --force               Reinstall even if an up-to-date binary is present.",
        "  --version <x.y.z>     Pin the release to install (default: latest).",
        "  -h, --help            Show this help.",
      ].join("\n"),
    );
    return;
  }

  const plat = detectPlatform(process.platform, process.arch);
  console.log(
    `Detected platform: ${process.platform}/${process.arch} -> ${plat.target}`,
  );

  // Dependency checks.
  console.log("\nDependency status:");
  const rgVer = await ensureRipgrep(plat, opts);
  console.log(`  ripgrep: ${rgVer}`);
  const rtkVer = reportReportOnlyDep(
    "rtk",
    plat,
    "Install rtk from its upstream distribution; it is optional (search falls back to plain ripgrep).",
  );
  console.log(`  rtk: ${rtkVer}`);
  const cmVer = reportReportOnlyDep(
    "codebase-memory-mcp",
    plat,
    "Install codebase-memory-mcp from its upstream distribution; it is required at runtime.",
  );
  console.log(`  codebase-memory-mcp: ${cmVer}`);

  // Resolve version + install dir.
  const version = await resolveVersion(opts.version);
  const dir = installDir(plat.osKind, process.env, os.homedir());
  fs.mkdirSync(dir, { recursive: true });
  const binName = binaryName(plat.osKind);
  const binPath = path.join(dir, binName);

  // Idempotency check.
  if (!opts.force && fs.existsSync(binPath)) {
    const installed = probeVersion(binPath); // e.g. "repo-explorer-mcp 0.1.0"
    const installedVer = installed ? installed.split(/\s+/).pop() : null;
    if (installedVer && hexEqual(installedVer, version)) {
      console.log(
        `\nrepo-explorer-mcp ${version} already installed at ${binPath} — already up to date.`,
      );
      printSummary(plat, binPath, version, { rgVer, rtkVer, cmVer });
      return;
    }
  }

  // Download + verify + extract.
  const archiveName = buildArchiveName({
    binaryBaseName: BINARY_BASE,
    version,
    target: plat.target,
    ext: plat.ext,
  });
  const { archiveUrl, sha256Url } = buildDownloadUrls({
    owner: OWNER,
    repo: REPO,
    version,
    archiveName,
  });
  console.log(`\nDownloading ${archiveUrl}`);
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "repo-explorer-mcp-"));
  const archiveFile = path.join(tmp, archiveName);
  try {
    const buf = await downloadTo(archiveUrl, archiveFile);
    const sumRes = await fetch(sha256Url, {
      headers: { "User-Agent": "repo-explorer-mcp-setup" },
    });
    if (!sumRes.ok)
      throw new Error(
        `Checksum download failed: ${sha256Url} (HTTP ${sumRes.status}).`,
      );
    const expected = parseSha256Line(await sumRes.text());
    const actual = sha256Hex(buf);
    if (!hexEqual(actual, expected)) {
      fs.rmSync(archiveFile, { force: true });
      throw new Error(
        `Checksum mismatch for ${archiveName}: download corrupted or tampered (expected ${expected}, got ${actual}).`,
      );
    }
    console.log("Checksum verified.");
    extractArchive(plat.osKind, archiveFile, tmp);
    const extracted = path.join(tmp, binName);
    fs.copyFileSync(extracted, binPath);
    if (plat.osKind !== "win32") fs.chmodSync(binPath, 0o755);
    console.log(`Installed ${binName} -> ${binPath}`);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }

  // PATH reporting (detect, never modify).
  if (!isDirOnPath(dir, process.env.PATH, path.delimiter)) {
    if (plat.osKind === "win32") {
      console.warn(
        `\nWARNING: ${dir} is not on your PATH. Add it via System Properties > Environment Variables, or: setx PATH "%PATH%;${dir}"`,
      );
    } else {
      console.warn(
        `\nWARNING: ${dir} is not on your PATH. Add this to ~/.bashrc or ~/.profile:\n  export PATH="$PATH:${dir}"`,
      );
    }
  }

  printSummary(plat, binPath, version, { rgVer, rtkVer, cmVer });
}

function printSummary(plat, binPath, version, deps) {
  console.log("\n=== Summary ===");
  console.log(`repo-explorer-mcp ${version} -> ${binPath}`);
  console.log(
    `ripgrep: ${deps.rgVer}   rtk: ${deps.rtkVer}   codebase-memory-mcp: ${deps.cmVer}`,
  );
  console.log(
    '\nAdd this to your .mcp.json (adjust args, e.g. add "--config <path>" if your repo-explorer.toml is not at the launch cwd):',
  );
  console.log(mcpSnippet(plat.osKind, binPath));
}

// Run main only when executed directly (not when imported by tests).
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((e) => {
    console.error(`repo-explorer-mcp-setup: ${e.message}`);
    process.exit(1);
  });
}
