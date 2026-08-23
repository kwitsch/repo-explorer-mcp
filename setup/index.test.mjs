import { test } from "node:test";
import assert from "node:assert/strict";
import { sep } from "node:path";
import {
  parseArgs,
  detectPlatform,
  buildArchiveName,
  buildDownloadUrls,
  parseSha256Line,
  hexEqual,
  isDirOnPath,
  binaryName,
  installDir,
  mcpSnippet,
  resolveVersion,
  parseInstalledVersion,
} from "./index.mjs";

test("parseArgs reads all flags", () => {
  assert.deepEqual(parseArgs(["--yes", "--force", "--version", "1.2.3"]), {
    yes: true,
    force: true,
    version: "1.2.3",
    help: false,
  });
  assert.deepEqual(parseArgs(["-y"]), {
    yes: true,
    force: false,
    version: null,
    help: false,
  });
  assert.deepEqual(parseArgs(["--help"]), {
    yes: false,
    force: false,
    version: null,
    help: true,
  });
  assert.deepEqual(parseArgs([]), {
    yes: false,
    force: false,
    version: null,
    help: false,
  });
});

test("parseArgs does not swallow a following flag as --version's value", () => {
  assert.deepEqual(parseArgs(["--version", "--yes"]), {
    yes: true,
    force: false,
    version: null,
    help: false,
  });
  assert.deepEqual(parseArgs(["--version"]), {
    yes: false,
    force: false,
    version: null,
    help: false,
  });
});

test("detectPlatform maps supported combos", () => {
  assert.deepEqual(detectPlatform("linux", "x64"), {
    target: "x86_64-unknown-linux-gnu",
    ext: "tar.gz",
    osKind: "linux",
  });
  assert.deepEqual(detectPlatform("win32", "x64"), {
    target: "x86_64-pc-windows-msvc",
    ext: "zip",
    osKind: "win32",
  });
});

test("detectPlatform throws on unsupported combos", () => {
  assert.throws(
    () => detectPlatform("darwin", "arm64"),
    /Unsupported platform/,
  );
  assert.throws(() => detectPlatform("linux", "arm64"), /Unsupported platform/);
});

test("buildArchiveName follows the frozen contract", () => {
  assert.equal(
    buildArchiveName({
      binaryBaseName: "repo-explorer-mcp",
      version: "0.1.0",
      target: "x86_64-unknown-linux-gnu",
      ext: "tar.gz",
    }),
    "repo-explorer-mcp-0.1.0-x86_64-unknown-linux-gnu.tar.gz",
  );
  assert.equal(
    buildArchiveName({
      binaryBaseName: "repo-explorer-mcp",
      version: "2.5.9",
      target: "x86_64-pc-windows-msvc",
      ext: "zip",
    }),
    "repo-explorer-mcp-2.5.9-x86_64-pc-windows-msvc.zip",
  );
});

test("buildDownloadUrls builds archive + sha256 URLs", () => {
  const { archiveUrl, sha256Url } = buildDownloadUrls({
    owner: "kwitsch",
    repo: "repo-explorer-mcp",
    version: "0.1.0",
    archiveName: "repo-explorer-mcp-0.1.0-x86_64-unknown-linux-gnu.tar.gz",
  });
  assert.equal(
    archiveUrl,
    "https://github.com/kwitsch/repo-explorer-mcp/releases/download/v0.1.0/repo-explorer-mcp-0.1.0-x86_64-unknown-linux-gnu.tar.gz",
  );
  assert.equal(sha256Url, archiveUrl + ".sha256");
});

test("parseSha256Line takes the first whitespace token", () => {
  assert.equal(
    parseSha256Line("abc123  repo-explorer-mcp-0.1.0.tar.gz\n"),
    "abc123",
  );
  assert.equal(parseSha256Line("DEADBEEF *file.zip"), "DEADBEEF");
});

test("hexEqual is case-insensitive and trims", () => {
  assert.ok(hexEqual("ABCDEF", "abcdef"));
  assert.ok(hexEqual(" abc \n", "ABC"));
  assert.ok(!hexEqual("abc", "abd"));
});

test("binaryName is OS-specific", () => {
  assert.equal(binaryName("linux"), "repo-explorer-mcp");
  assert.equal(binaryName("win32"), "repo-explorer-mcp.exe");
});

test("installDir is OS-specific", () => {
  assert.equal(
    installDir("linux", {}, "/home/u"),
    ["", "home", "u", ".local", "bin"].join(sep),
  );
});

test("isDirOnPath normalizes membership", () => {
  const sep = ":";
  assert.ok(
    isDirOnPath("/home/u/.local/bin", "/usr/bin:/home/u/.local/bin", sep),
  );
  assert.ok(isDirOnPath("/home/u/.local/bin/", "/home/u/.local/bin", sep));
  assert.ok(!isDirOnPath("/opt/bin", "/usr/bin:/home/u/.local/bin", sep));
});

test("isDirOnPath is case-insensitive only when asked (Windows)", () => {
  const dir = "C:\\Users\\me\\AppData\\Local\\repo-explorer-mcp";
  const pathEnv =
    "C:\\Windows;c:\\users\\me\\appdata\\local\\repo-explorer-mcp";
  assert.ok(!isDirOnPath(dir, pathEnv, ";"));
  assert.ok(isDirOnPath(dir, pathEnv, ";", true));
});

test("mcpSnippet round-trips a path without double-escaping backslashes", () => {
  const winPath =
    "C:\\Users\\test\\AppData\\Local\\repo-explorer-mcp\\repo-explorer-mcp.exe";
  const parsed = JSON.parse(mcpSnippet(winPath));
  assert.equal(parsed.mcpServers["repo-explorer"].command, winPath);

  const posixPath = "/home/user/.local/bin/repo-explorer-mcp";
  const parsedPosix = JSON.parse(mcpSnippet(posixPath));
  assert.equal(parsedPosix.mcpServers["repo-explorer"].command, posixPath);
});

test("parseInstalledVersion finds the semver regardless of surrounding text", () => {
  assert.equal(parseInstalledVersion("repo-explorer-mcp 0.1.0"), "0.1.0");
  // Survives a suffix, which a last-token split would not.
  assert.equal(
    parseInstalledVersion("repo-explorer-mcp 0.1.0 (abc1234)"),
    "0.1.0",
  );
  assert.equal(
    parseInstalledVersion("repo-explorer-mcp 1.2.3-rc.1"),
    "1.2.3-rc.1",
  );
  assert.equal(parseInstalledVersion(null), null);
  assert.equal(parseInstalledVersion("no version here"), null);
});

test("resolveVersion rejects an explicitly empty pinned version", async () => {
  await assert.rejects(() => resolveVersion(""), /non-empty value/);
});

test("resolveVersion strips a leading v from a pinned version", async () => {
  assert.equal(await resolveVersion("v1.2.3"), "1.2.3");
  assert.equal(await resolveVersion("1.2.3"), "1.2.3");
});
