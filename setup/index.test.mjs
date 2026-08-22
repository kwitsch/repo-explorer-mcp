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
