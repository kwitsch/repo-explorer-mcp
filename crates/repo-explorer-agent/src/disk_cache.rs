//! The on-disk L2 behind the in-memory query cache: one JSON file per cached
//! result, under `<cache dir>/results/v<SCHEMA_VERSION>/`.
//!
//! No database, and no locking. Every entry is independent, idempotent and
//! recomputable, so concurrent MCP processes need nothing more than an atomic
//! replace: the file is written to a unique temp name and `fs::rename`d into
//! place (atomic on POSIX, `MoveFileEx(REPLACE_EXISTING)` on Windows). Two
//! writers racing on one key both write a correct answer for that key, and a
//! reader either sees the whole old value or the whole new one. Upgrade
//! trigger: swap this module for sqlite when `cache stats` must report
//! cross-session aggregates that the metrics stream cannot derive, or when
//! entries stop being independent.
//!
//! [`SCHEMA_VERSION`] is the single versioning point for the stored shape —
//! it appears both as the directory segment and as the `v` field, so a bump
//! makes every old entry unreachable and the sweep deletes the old directory.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use repo_explorer_core::domain::ExplorationOutcome;
use repo_explorer_core::fingerprint::RepoFingerprint;
use serde::{Deserialize, Serialize};

use crate::cache::QueryEntry;

/// Version of the on-disk entry shape. Bump this — and nothing else — when the
/// stored value changes; every entry under an older version becomes
/// unreachable by construction and is deleted by the next sweep.
pub const SCHEMA_VERSION: u32 = 2;

/// FNV-1a over the cache key, rendered as 16 hex chars, purely to get a legal
/// file name out of a key that contains a repository path and free-text query.
/// The *full* key is stored inside the entry and compared on read, so a
/// collision is a miss, never a wrong answer — which leaves determinism across
/// releases as the only requirement, and rules out `std`'s `DefaultHasher`
/// (documented as unstable across Rust versions: a toolchain bump would
/// silently cold-start every user's cache).
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn file_stem(key: &str) -> String {
    format!("{:016x}", fnv1a64(key.as_bytes()))
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Store-time identity of one file a cached answer references.
///
/// ponytail: `(len, mtime)` rather than a content hash — the same primitive
/// `GitStateProbe` already trusts for untracked files, at one `stat` per
/// referenced file. Hash the contents instead if a same-size, same-mtime
/// rewrite ever bites in practice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct FileDep {
    pub path: String,
    pub len: u64,
    pub mtime_ns: u64,
}

/// Stamp one referenced path, resolved against the repository root (an
/// absolute `rel` wins over the root, which is what `Path::join` already
/// does). `None` when the file is gone or unreadable — which a caller must
/// treat as "changed".
///
/// Follows symlinks deliberately: the snippet in the answer was read through
/// the link (`dispatch::read_file_canonical`), so the link's own inode — whose
/// `len` and `mtime` never move when the target is edited — would stamp a file
/// the answer does not actually come from. A dangling link then reads as
/// `None`, i.e. "changed", which is the safe direction.
pub(crate) fn file_dep(repo_root: &Path, rel: &str) -> Option<FileDep> {
    let meta = fs::metadata(repo_root.join(rel)).ok()?;
    let mtime_ns = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    Some(FileDep {
        path: rel.to_string(),
        len: meta.len(),
        mtime_ns,
    })
}

/// One cached result as it lives on disk. The single place the persisted shape
/// is defined, and therefore the single place M-3 replaces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct StoredEntry {
    pub v: u32,
    /// The full `ResultCache::query_key`, so a file-name hash collision reads
    /// back as a miss rather than as somebody else's answer.
    pub key: String,
    pub head_sha: String,
    pub dirty_hash: String,
    pub created_at: u64,
    pub llm_turns: u32,
    pub tokens: u64,
    pub deps: Vec<FileDep>,
    pub result: ExplorationOutcome,
}

impl StoredEntry {
    pub(crate) fn from_entry(key: &str, entry: &QueryEntry) -> Self {
        Self {
            v: SCHEMA_VERSION,
            key: key.to_string(),
            head_sha: entry.fingerprint.head_sha.clone(),
            dirty_hash: entry.fingerprint.dirty_hash.clone(),
            created_at: now_unix(),
            llm_turns: entry.llm_turns,
            tokens: entry.tokens,
            deps: entry.deps.clone(),
            result: entry.result.clone(),
        }
    }

    pub(crate) fn into_entry(self) -> QueryEntry {
        QueryEntry {
            fingerprint: RepoFingerprint {
                head_sha: self.head_sha,
                dirty_hash: self.dirty_hash,
            },
            result: self.result,
            llm_turns: self.llm_turns,
            tokens: self.tokens,
            deps: self.deps,
        }
    }
}

/// Serial for temp file names, so concurrent writers inside one process never
/// share a scratch path (the pid alone is not enough).
static WRITE_SEQ: AtomicU64 = AtomicU64::new(0);

pub(crate) struct DiskCache {
    /// `<dir>/results/v<SCHEMA_VERSION>`.
    root: PathBuf,
    max_bytes: u64,
    swept: AtomicBool,
}

impl DiskCache {
    /// `None` — i.e. L1-only for this process — when the layer is switched off
    /// (empty `dir`, zero budget) or the directory is unusable. Never an
    /// error: the cache must never fail a query.
    pub(crate) fn open(dir: &str, max_bytes: u64) -> Option<Self> {
        if dir.is_empty() || max_bytes == 0 {
            return None;
        }
        let root = Path::new(dir)
            .join("results")
            .join(format!("v{SCHEMA_VERSION}"));
        if let Err(e) = create_dir(&root) {
            tracing::warn!(
                dir = %root.display(),
                error = %e,
                "persistent result cache disabled: directory is not usable"
            );
            return None;
        }
        Some(Self {
            root,
            max_bytes,
            swept: AtomicBool::new(false),
        })
    }

    fn entry_path(&self, key: &str) -> PathBuf {
        self.root.join(format!("{}.json", file_stem(key)))
    }

    pub(crate) fn get(&self, key: &str) -> Option<StoredEntry> {
        let path = self.entry_path(key);
        let bytes = fs::read(&path).ok()?;
        match serde_json::from_slice::<StoredEntry>(&bytes) {
            Ok(entry) if entry.v == SCHEMA_VERSION && entry.key == key => {
                touch(&path);
                Some(entry)
            }
            // A hash collision or a hand-edited version field: a miss, and the
            // file stays — it is somebody else's valid-looking entry.
            Ok(_) => None,
            // Truncated or garbage: self-healing, so one bad write cannot
            // poison a key for good.
            Err(_) => {
                let _ = fs::remove_file(&path);
                None
            }
        }
    }

    pub(crate) fn put(&self, key: &str, entry: &StoredEntry) {
        // A write failure is logged and dropped, never retried and never
        // latched off: the store is write-through, so the next completed
        // exploration (seconds away, on a blocking thread) simply tries
        // again — which is what makes a transient ENOSPC or a Windows
        // sharing violation recoverable within one long-lived session.
        if let Err(e) = self.write_entry(key, entry) {
            tracing::warn!(
                dir = %self.root.display(),
                error = %e,
                "persistent result cache write failed; the result stays in memory only"
            );
            return;
        }
        if !self.swept.swap(true, Ordering::Relaxed) {
            // Once per process, riding the first successful write rather than
            // startup: the run's result already exists by then, and `put` runs
            // on a blocking thread (see `ResultCache::put_query_l2`).
            //
            // It is *awaited*, not detached, so it lands on the tail of that
            // one run — measured at ~60 ms over a full 256 MiB store, paid
            // once, on a run that has already spent seconds and several LLM
            // turns. A detached task would instead risk being killed
            // mid-`remove_file` by process exit, leaving the budget
            // unenforced. A cache *hit* never reaches this code at all.
            //
            // ponytail: a process that never completes an uncached query never
            // sweeps. Move this to a timer if a long-lived server is ever seen
            // overrunning the budget mid-session.
            sweep(&self.root, self.max_bytes);
        }
    }

    fn write_entry(&self, key: &str, entry: &StoredEntry) -> io::Result<()> {
        let json = serde_json::to_vec(entry).map_err(io::Error::other)?;
        let stem = file_stem(key);
        let tmp = self.root.join(format!(
            "{stem}.tmp.{}.{}",
            std::process::id(),
            WRITE_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let write = (|| -> io::Result<()> {
            use std::io::Write as _;
            let mut opts = fs::OpenOptions::new();
            opts.write(true).create(true).truncate(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt as _;
                opts.mode(0o600);
            }
            let mut f = opts.open(&tmp)?;
            f.write_all(&json)
        })();
        if let Err(e) = write {
            let _ = fs::remove_file(&tmp);
            return Err(e);
        }
        if let Err(e) = fs::rename(&tmp, self.root.join(format!("{stem}.json"))) {
            let _ = fs::remove_file(&tmp);
            return Err(e);
        }
        Ok(())
    }
}

/// 0700 on unix; on Windows the tree lives under the per-user
/// `%LOCALAPPDATA%`, whose ACL the OS already enforces (ACL code would mean a
/// new winapi-family dependency to restate a default).
fn create_dir(path: &Path) -> io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt as _;
        builder.mode(0o700);
    }
    builder.create(path)
}

/// mtime *is* `last_hit_at`, so LRU order is the directory listing and a hit
/// costs no rewrite of a multi-kilobyte entry. A failure is ignored: eviction
/// precision is not correctness.
fn touch(path: &Path) {
    let _ = fs::OpenOptions::new()
        .write(true)
        .open(path)
        .and_then(|f| f.set_times(fs::FileTimes::new().set_modified(SystemTime::now())));
}

/// Delete superseded schema directories, then evict oldest-by-mtime until the
/// store is back under budget.
///
/// "Superseded" means strictly older: a `v<N>` sibling with `N > SCHEMA_VERSION`
/// belongs to a newer binary sharing this cache directory (a mixed-version
/// window — a locally built binary next to an installed one), and wiping it
/// would leave both sides permanently cold. An unparseable name is left alone
/// for the same reason.
fn sweep(root: &Path, max_bytes: u64) {
    if let Some(parent) = root.parent()
        && let Ok(siblings) = fs::read_dir(parent)
    {
        for sibling in siblings.flatten() {
            let name = sibling.file_name();
            let older = name
                .to_str()
                .and_then(|n| n.strip_prefix('v'))
                .and_then(|n| n.parse::<u32>().ok())
                .is_some_and(|v| v < SCHEMA_VERSION);
            if older {
                let _ = fs::remove_dir_all(sibling.path());
            }
        }
    }
    let Ok(read_dir) = fs::read_dir(root) else {
        return;
    };
    let mut total = 0u64;
    let mut files: Vec<(SystemTime, u64, PathBuf)> = Vec::new();
    for entry in read_dir.flatten() {
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        total += meta.len();
        files.push((
            meta.modified().unwrap_or(UNIX_EPOCH),
            meta.len(),
            entry.path(),
        ));
    }
    if total <= max_bytes {
        return;
    }
    files.sort_by_key(|(mtime, _, _)| *mtime);
    for (_, len, path) in files {
        if total <= max_bytes {
            break;
        }
        if fs::remove_file(&path).is_ok() {
            total = total.saturating_sub(len);
        }
    }
}

/// What `<binary> cache stats` reports. Built from `read_dir` + `metadata`
/// only, so it never reads a multi-hundred-megabyte store into memory.
///
/// Deliberately no hit counts or saved turns/tokens: those are per-run facts
/// that already ride the metrics line and the eval scorer, and keeping them
/// here would force a rewrite of every entry on every read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CacheStats {
    pub dir: String,
    pub schema_version: u32,
    pub entries: u64,
    pub bytes: u64,
    pub oldest_unix: Option<u64>,
    pub newest_unix: Option<u64>,
}

/// `dir` is the configured cache directory (the parent of `results/`). A
/// directory that does not exist yet reports an empty store, not an error.
pub fn stats(dir: &Path) -> io::Result<CacheStats> {
    let root = dir.join("results").join(format!("v{SCHEMA_VERSION}"));
    let mut out = CacheStats {
        dir: dir.display().to_string(),
        schema_version: SCHEMA_VERSION,
        entries: 0,
        bytes: 0,
        oldest_unix: None,
        newest_unix: None,
    };
    let read_dir = match fs::read_dir(&root) {
        Ok(r) => r,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(e),
    };
    // Tolerant iteration, like `sweep` and `clear`: another process's `put`
    // renames a temp file over an entry and its own sweep unlinks entries
    // while this listing runs, so a vanished name is normal concurrency, not
    // a reason to fail a read-only inspection.
    for entry in read_dir.flatten() {
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() || entry.path().extension().is_none_or(|e| e != "json") {
            continue;
        }
        out.entries += 1;
        out.bytes += meta.len();
        let secs = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        out.oldest_unix = Some(out.oldest_unix.map_or(secs, |o: u64| o.min(secs)));
        out.newest_unix = Some(out.newest_unix.map_or(secs, |n: u64| n.max(secs)));
    }
    Ok(out)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CacheCleared {
    pub dir: String,
    pub removed_entries: u64,
    pub removed_bytes: u64,
}

/// Remove every persisted result (all schema versions), leaving the configured
/// directory itself in place. Non-interactive, like `--install`/`--uninstall`.
pub fn clear(dir: &Path) -> io::Result<CacheCleared> {
    let results = dir.join("results");
    let mut out = CacheCleared {
        dir: dir.display().to_string(),
        removed_entries: 0,
        removed_bytes: 0,
    };
    let versions = match fs::read_dir(&results) {
        Ok(r) => r,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(e),
    };
    for version in versions.flatten() {
        let Ok(entries) = fs::read_dir(version.path()) else {
            continue;
        };
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata()
                && meta.is_file()
            {
                out.removed_entries += 1;
                out.removed_bytes += meta.len();
            }
        }
    }
    fs::remove_dir_all(&results)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use repo_explorer_core::domain::{ExplorationFinding, FileLocation};
    use repo_explorer_core::domain::{ExplorationResult, StageExit};

    fn temp_dir(test: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("disk_cache_{test}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn outcome(path: &str) -> ExplorationOutcome {
        ExplorationOutcome {
            result: result(path),
            retrieval_confidence: 92,
            stage_exit: StageExit::Verify,
            symbols: vec![(
                FileLocation {
                    path: PathBuf::from(path),
                    line_start: 1,
                    line_end: 4,
                },
                "a::main".to_string(),
            )],
        }
    }

    fn result(path: &str) -> ExplorationResult {
        ExplorationResult {
            findings: vec![ExplorationFinding {
                location: FileLocation {
                    path: PathBuf::from(path),
                    line_start: 1,
                    line_end: 4,
                },
                snippet: Some("fn main() {}".to_string()),
                note: None,
            }],
            summary: "a summary".to_string(),
        }
    }

    fn entry(key: &str) -> StoredEntry {
        StoredEntry::from_entry(
            key,
            &QueryEntry {
                fingerprint: RepoFingerprint {
                    head_sha: "aaa".to_string(),
                    dirty_hash: "ddd".to_string(),
                },
                result: outcome("src/a.rs"),
                llm_turns: 3,
                tokens: 18_420,
                deps: vec![FileDep {
                    path: "src/a.rs".to_string(),
                    len: 12,
                    mtime_ns: 7,
                }],
            },
        )
    }

    #[test]
    fn entry_roundtrips_result() {
        let dir = temp_dir("roundtrip");
        let cache = DiskCache::open(&dir.display().to_string(), 1 << 20).unwrap();
        cache.put("k", &entry("k"));
        let got = cache.get("k").expect("entry must come back");
        assert_eq!(got.result, outcome("src/a.rs"));
        assert_eq!(got.head_sha, "aaa");
        assert_eq!(got.dirty_hash, "ddd");
        assert_eq!(got.llm_turns, 3);
        assert_eq!(got.tokens, 18_420);
        assert_eq!(got.deps.len(), 1);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn get_is_a_miss_when_the_stored_key_differs() {
        // A file-name hash collision must read back as a miss, never as
        // somebody else's answer — this is why 64 bits of FNV is enough.
        let dir = temp_dir("key_mismatch");
        let cache = DiskCache::open(&dir.display().to_string(), 1 << 20).unwrap();
        let mut stored = entry("key-b");
        stored.key = "key-b".to_string();
        let path = cache.entry_path("key-a");
        fs::write(&path, serde_json::to_vec(&stored).unwrap()).unwrap();
        assert!(cache.get("key-a").is_none());
        assert!(
            path.exists(),
            "a foreign but valid entry must not be deleted"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn key_hash_is_stable() {
        // Cross-release determinism: changing this constant cold-starts every
        // user's cache, which is exactly why `DefaultHasher` is not used.
        assert_eq!(file_stem("repo-explorer"), "56c4ce76647abff7");
    }

    #[test]
    fn get_is_a_miss_on_a_schema_version_bump() {
        let dir = temp_dir("schema_bump");
        let cache = DiskCache::open(&dir.display().to_string(), 1 << 20).unwrap();
        let mut stored = entry("k");
        stored.v = SCHEMA_VERSION - 1;
        fs::write(cache.entry_path("k"), serde_json::to_vec(&stored).unwrap()).unwrap();
        assert!(cache.get("k").is_none());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn corrupt_entry_file_is_a_miss_and_is_removed() {
        let dir = temp_dir("corrupt");
        let cache = DiskCache::open(&dir.display().to_string(), 1 << 20).unwrap();
        let path = cache.entry_path("k");
        fs::write(&path, b"{\"v\":1,\"key\":\"k\",\"resu").unwrap();
        assert!(cache.get("k").is_none());
        assert!(!path.exists(), "a corrupt entry must be self-healed away");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn open_returns_none_when_the_dir_cannot_be_created() {
        let dir = temp_dir("unusable");
        let file = dir.join("not-a-dir");
        fs::write(&file, b"x").unwrap();
        assert!(DiskCache::open(&file.display().to_string(), 1 << 20).is_none());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn open_returns_none_when_switched_off() {
        let dir = temp_dir("switched_off");
        assert!(DiskCache::open("", 1 << 20).is_none());
        assert!(DiskCache::open(&dir.display().to_string(), 0).is_none());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_write_failure_never_fails_a_query_and_is_recoverable() {
        let dir = temp_dir("write_failure");
        let cache = DiskCache::open(&dir.display().to_string(), 1 << 20).unwrap();
        // Remove the directory out from under the store: the write fails,
        // silently...
        fs::remove_dir_all(&cache.root).unwrap();
        cache.put("k", &entry("k"));
        assert!(cache.get("k").is_none());
        // ...and the store picks itself up once the condition clears, rather
        // than staying dead for the life of a long-running server.
        fs::create_dir_all(&cache.root).unwrap();
        cache.put("k", &entry("k"));
        assert!(cache.get("k").is_some());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn stage_exit_wire_strings_are_pinned() {
        // These four literals are the contract in three places at once: the
        // persisted entry above, the MCP response's `stage_exit`, and the
        // `path=` field `eval`'s log parser reads. `as_str` must not drift
        // from what serde writes. Core cannot host this test — it has no
        // `serde_json` and must not gain one.
        for (stage, wire) in [
            (StageExit::EarlyExit, "early-exit"),
            (StageExit::Verify, "verify"),
            (StageExit::Fallback, "fallback"),
            (StageExit::Cache, "cache"),
        ] {
            assert_eq!(
                serde_json::to_string(&stage).unwrap(),
                format!("\"{wire}\"")
            );
            assert_eq!(stage.as_str(), wire);
            assert_eq!(
                serde_json::from_str::<StageExit>(&format!("\"{wire}\"")).unwrap(),
                stage
            );
        }
    }

    #[test]
    fn stored_shape_is_pinned_to_the_schema_version() {
        // The stored shape is defined across two crates: the envelope here and
        // `ExplorationResult` in core. A literal makes any move on either side
        // fail loudly, so whoever moves it either reverts or bumps
        // `SCHEMA_VERSION` — which is the contract `CLAUDE.md` states. Update
        // BOTH together, never this literal alone.
        let stored = StoredEntry {
            v: SCHEMA_VERSION,
            key: "k".to_string(),
            head_sha: "aaa".to_string(),
            dirty_hash: "ddd".to_string(),
            created_at: 1_700_000_000,
            llm_turns: 3,
            tokens: 18_420,
            deps: vec![FileDep {
                path: "src/a.rs".to_string(),
                len: 12,
                mtime_ns: 7,
            }],
            result: outcome("src/a.rs"),
        };
        assert_eq!(
            serde_json::to_string(&stored).unwrap(),
            r#"{"v":2,"key":"k","head_sha":"aaa","dirty_hash":"ddd","created_at":1700000000,"llm_turns":3,"tokens":18420,"deps":[{"path":"src/a.rs","len":12,"mtime_ns":7}],"result":{"result":{"findings":[{"location":{"path":"src/a.rs","line_start":1,"line_end":4},"snippet":"fn main() {}","note":null}],"summary":"a summary"},"retrieval_confidence":92,"stage_exit":"verify","symbols":[[{"path":"src/a.rs","line_start":1,"line_end":4},"a::main"]]}}"#
        );
    }

    #[cfg(unix)]
    #[test]
    fn the_store_is_private_to_the_user() {
        // The plan's mitigation for "sensitive code on disk": entries hold
        // verbatim source snippets and absolute repository paths. Asserting
        // "no group/other bits" rather than an exact mode keeps this immune to
        // the umask.
        use std::os::unix::fs::PermissionsExt as _;
        let dir = temp_dir("permissions");
        let cache = DiskCache::open(&dir.display().to_string(), 1 << 20).unwrap();
        cache.put("k", &entry("k"));
        let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o077;
        assert_eq!(mode(&cache.entry_path("k")), 0, "entries must be 0600");
        assert_eq!(mode(&cache.root), 0, "the store directory must be 0700");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sweep_evicts_oldest_until_under_max_bytes() {
        let dir = temp_dir("sweep_bytes");
        let root = dir.join("results").join(format!("v{SCHEMA_VERSION}"));
        fs::create_dir_all(&root).unwrap();
        let base = SystemTime::now() - std::time::Duration::from_secs(10_000);
        for i in 0..5u64 {
            let path = root.join(format!("{i}.json"));
            fs::write(&path, vec![b'x'; 100]).unwrap();
            let f = fs::OpenOptions::new().write(true).open(&path).unwrap();
            f.set_times(
                fs::FileTimes::new().set_modified(base + std::time::Duration::from_secs(i * 60)),
            )
            .unwrap();
        }
        sweep(&root, 250);
        let left: Vec<String> = fs::read_dir(&root)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            left.len(),
            2,
            "expected the two newest to survive: {left:?}"
        );
        assert!(left.contains(&"4.json".to_string()));
        assert!(left.contains(&"3.json".to_string()));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sweep_removes_older_schema_dirs_but_never_newer_ones() {
        let dir = temp_dir("sweep_versions");
        let root = dir.join("results").join(format!("v{SCHEMA_VERSION}"));
        let stale = dir.join("results").join("v0");
        // A newer binary sharing this cache dir (mixed-version window): wiping
        // its store would leave both sides permanently cold.
        let newer = dir.join("results").join(format!("v{}", SCHEMA_VERSION + 1));
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&stale).unwrap();
        fs::create_dir_all(&newer).unwrap();
        fs::write(stale.join("old.json"), b"{}").unwrap();
        fs::write(newer.join("new.json"), b"{}").unwrap();
        sweep(&root, u64::MAX);
        assert!(!stale.exists());
        assert!(root.exists());
        assert!(newer.join("new.json").exists());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn two_stores_on_one_dir_see_each_others_entries() {
        // The cross-process stand-in: there is no shared handle, lock or WAL
        // to exercise, only an atomic replace under a shared directory.
        let dir = temp_dir("two_stores");
        let a = DiskCache::open(&dir.display().to_string(), 1 << 20).unwrap();
        let b = DiskCache::open(&dir.display().to_string(), 1 << 20).unwrap();
        a.put("k", &entry("k"));
        assert_eq!(b.get("k").unwrap().llm_turns, 3);
        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn concurrent_writes_to_one_key_leave_a_readable_entry() {
        let dir = temp_dir("concurrent");
        let cache =
            std::sync::Arc::new(DiskCache::open(&dir.display().to_string(), 1 << 20).unwrap());
        let mut tasks = Vec::new();
        for _ in 0..8 {
            let cache = cache.clone();
            tasks.push(tokio::task::spawn_blocking(move || {
                cache.put("k", &entry("k"))
            }));
        }
        for t in tasks {
            t.await.unwrap();
        }
        assert!(cache.get("k").is_some(), "a racing write must not corrupt");
        let strays = fs::read_dir(&cache.root)
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
            .count();
        assert_eq!(strays, 0, "temp files must not be left behind");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn get_touches_mtime() {
        let dir = temp_dir("touch");
        let cache = DiskCache::open(&dir.display().to_string(), 1 << 20).unwrap();
        cache.put("k", &entry("k"));
        let path = cache.entry_path("k");
        let old = SystemTime::now() - std::time::Duration::from_secs(3600);
        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(fs::FileTimes::new().set_modified(old))
            .unwrap();
        assert!(cache.get("k").is_some());
        let after = fs::metadata(&path).unwrap().modified().unwrap();
        assert!(after > old, "a hit must refresh the LRU stamp");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn stats_and_clear_report_and_empty_the_store() {
        let dir = temp_dir("stats_clear");
        let cache = DiskCache::open(&dir.display().to_string(), 1 << 20).unwrap();
        cache.put("a", &entry("a"));
        cache.put("b", &entry("b"));
        let s = stats(&dir).unwrap();
        assert_eq!(s.entries, 2);
        assert!(s.bytes > 0);
        assert_eq!(s.schema_version, SCHEMA_VERSION);
        assert!(s.oldest_unix.is_some() && s.newest_unix.is_some());

        let cleared = clear(&dir).unwrap();
        assert_eq!(cleared.removed_entries, 2);
        assert_eq!(cleared.removed_bytes, s.bytes);
        assert_eq!(stats(&dir).unwrap().entries, 0);
        // A missing store is an empty store, not an error.
        assert_eq!(clear(&dir).unwrap().removed_entries, 0);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn file_dep_stamps_an_existing_file_only() {
        let dir = temp_dir("file_dep");
        fs::write(dir.join("a.rs"), b"hello").unwrap();
        let dep = file_dep(&dir, "a.rs").expect("an existing file must stamp");
        assert_eq!(dep.path, "a.rs");
        assert_eq!(dep.len, 5);
        assert!(file_dep(&dir, "missing.rs").is_none());
        #[cfg(unix)]
        {
            // A symlinked source file must stamp its *target*: the snippet was
            // read through the link, so stamping the link inode would never
            // notice the target being edited.
            std::os::unix::fs::symlink(dir.join("a.rs"), dir.join("link.rs")).unwrap();
            assert_eq!(file_dep(&dir, "link.rs").unwrap().len, 5);
            fs::write(dir.join("a.rs"), b"hello world").unwrap();
            assert_eq!(file_dep(&dir, "link.rs").unwrap().len, 11);
        }
        fs::remove_dir_all(&dir).ok();
    }
}
