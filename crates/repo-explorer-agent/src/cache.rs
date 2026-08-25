//! In-memory, per-process result caches keyed by repository fingerprint:
//! tool-result memoization, retrieval-leg memoization, and the query→result
//! cache with path-level invalidation. Interior-mutable (`Mutex`) because the
//! agent is shared behind an `Arc` and every entry point takes `&self`.

use repo_explorer_core::domain::{
    Candidate, ExplorationFinding, ExplorationQuery, ExplorationResult,
};
use repo_explorer_core::fingerprint::RepoFingerprint;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

/// A capped `String`-keyed map with FIFO eviction (oldest inserted first out).
struct CappedMap<V> {
    map: HashMap<String, V>,
    order: VecDeque<String>,
    cap: usize,
}

impl<V: Clone> CappedMap<V> {
    fn new(cap: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            cap,
        }
    }

    fn get(&self, key: &str) -> Option<V> {
        self.map.get(key).cloned()
    }

    fn get_mut(&mut self, key: &str) -> Option<&mut V> {
        self.map.get_mut(key)
    }

    fn insert(&mut self, key: String, value: V) {
        if self.cap == 0 {
            return;
        }
        if !self.map.contains_key(&key) {
            self.order.push_back(key.clone());
            self.map.insert(key, value);
            while self.map.len() > self.cap {
                if let Some(oldest) = self.order.pop_front() {
                    self.map.remove(&oldest);
                }
            }
        } else {
            self.map.insert(key, value);
        }
    }

    fn remove(&mut self, key: &str) {
        self.map.remove(key);
        self.order.retain(|k| k != key);
    }
}

/// Length-prefix a field (`len:content`) so concatenating several fields into
/// a cache key can never collide across differing field boundaries,
/// regardless of what characters the fields themselves contain. Shared by
/// `query_key` below and the retrieval-leg keys in `pipeline.rs`.
pub(crate) fn encode_field(s: &str) -> String {
    format!("{}:{}", s.len(), s)
}

/// One cached query result plus what it depends on.
#[derive(Debug, Clone)]
pub(crate) struct QueryEntry {
    pub fingerprint: RepoFingerprint,
    pub result: ExplorationResult,
}

struct Inner {
    tools: CappedMap<(String, Vec<ExplorationFinding>)>,
    legs: CappedMap<Vec<Candidate>>,
    queries: CappedMap<QueryEntry>,
}

/// All three caches behind one lock (entries are small; contention is one
/// exploration at a time per repo in practice).
pub(crate) struct ResultCache {
    inner: Mutex<Inner>,
}

impl ResultCache {
    pub(crate) fn new(max_entries: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                tools: CappedMap::new(max_entries),
                legs: CappedMap::new(max_entries),
                queries: CappedMap::new(max_entries),
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        // Poison recovery mirrors ProviderRouter: one panicked holder must not
        // permanently disable caching.
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// `tool` and `args_json` are model-supplied and unvalidated at this
    /// point, so both are length-prefixed via `encode_field` rather than
    /// joined with a bare delimiter — differing field boundaries can never
    /// hash-collide regardless of what characters the fields themselves
    /// contain.
    pub(crate) fn tool_key(fp: &RepoFingerprint, tool: &str, args_json: &str) -> String {
        format!(
            "{}#{}#{}{}",
            fp.head_sha,
            fp.dirty_hash,
            encode_field(tool),
            encode_field(args_json)
        )
    }

    pub(crate) fn get_tool(&self, key: &str) -> Option<(String, Vec<ExplorationFinding>)> {
        self.lock().tools.get(key)
    }

    pub(crate) fn put_tool(&self, key: String, value: (String, Vec<ExplorationFinding>)) {
        self.lock().tools.insert(key, value);
    }

    pub(crate) fn leg_key(fp: &RepoFingerprint, leg: &str) -> String {
        format!("{}#{}#{leg}", fp.head_sha, fp.dirty_hash)
    }

    pub(crate) fn get_leg(&self, key: &str) -> Option<Vec<Candidate>> {
        self.lock().legs.get(key)
    }

    pub(crate) fn put_leg(&self, key: String, value: Vec<Candidate>) {
        self.lock().legs.insert(key, value);
    }

    /// Fingerprint-independent query key: invalidation is handled via the
    /// stored fingerprint, not the key. Fields are joined via `encode_field`
    /// so differing field boundaries can never collide.
    pub(crate) fn query_key(query: &ExplorationQuery) -> String {
        let text = query.text.trim().to_lowercase();
        let scope = query
            .scope_hint
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        let max_results = query.max_results.map(|m| m.to_string()).unwrap_or_default();
        format!(
            "{}{}{}",
            encode_field(&text),
            encode_field(&scope),
            encode_field(&max_results)
        )
    }

    pub(crate) fn get_query(&self, key: &str) -> Option<QueryEntry> {
        self.lock().queries.get(key)
    }

    pub(crate) fn put_query(&self, key: String, entry: QueryEntry) {
        self.lock().queries.insert(key, entry);
    }

    /// Keep a still-valid entry current after the repo moved without any
    /// actual diff (see `AgentLoop::query_cache_lookup`). Compare-and-swap
    /// against `expected` (the fingerprint the caller's empty-diff decision
    /// was based on): if a concurrent call already replaced the entry (e.g.
    /// a full recompute via `store_query_cache`), its fingerprint no longer
    /// matches `expected` and this becomes a no-op, so a stale relabel can
    /// never pair a fresher result with a fingerprint it wasn't produced
    /// from.
    pub(crate) fn refresh_query_fingerprint(
        &self,
        key: &str,
        expected: &RepoFingerprint,
        fingerprint: RepoFingerprint,
    ) {
        let mut inner = self.lock();
        if let Some(entry) = inner.queries.get_mut(key)
            && entry.fingerprint == *expected
        {
            entry.fingerprint = fingerprint;
        }
    }

    pub(crate) fn remove_query(&self, key: &str) {
        self.lock().queries.remove(key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp(sha: &str) -> RepoFingerprint {
        RepoFingerprint {
            head_sha: sha.to_string(),
            dirty_hash: "d".to_string(),
        }
    }

    fn entry(sha: &str) -> QueryEntry {
        QueryEntry {
            fingerprint: fp(sha),
            result: ExplorationResult {
                findings: vec![],
                summary: format!("from {sha}"),
            },
        }
    }

    #[test]
    fn fifo_eviction_drops_oldest() {
        let cache = ResultCache::new(2);
        cache.put_tool("k1".into(), ("v1".into(), vec![]));
        cache.put_tool("k2".into(), ("v2".into(), vec![]));
        cache.put_tool("k3".into(), ("v3".into(), vec![]));
        assert!(cache.get_tool("k1").is_none(), "oldest evicted");
        assert!(cache.get_tool("k2").is_some());
        assert!(cache.get_tool("k3").is_some());
    }

    #[test]
    fn zero_cap_stores_nothing() {
        let cache = ResultCache::new(0);
        cache.put_tool("k".into(), ("v".into(), vec![]));
        assert!(cache.get_tool("k").is_none());
    }

    #[test]
    fn reinsert_same_key_replaces_without_growth() {
        let cache = ResultCache::new(2);
        cache.put_tool("k".into(), ("v1".into(), vec![]));
        cache.put_tool("k".into(), ("v2".into(), vec![]));
        cache.put_tool("k2".into(), ("x".into(), vec![]));
        assert_eq!(cache.get_tool("k").map(|(v, _)| v), Some("v2".to_string()));
        assert!(cache.get_tool("k2").is_some());
    }

    #[test]
    fn query_refresh_and_remove() {
        let cache = ResultCache::new(4);
        cache.put_query("q".into(), entry("a"));
        cache.refresh_query_fingerprint("q", &fp("a"), fp("b"));
        assert_eq!(cache.get_query("q").unwrap().fingerprint, fp("b"));
        cache.remove_query("q");
        assert!(cache.get_query("q").is_none());
    }

    #[test]
    fn query_refresh_is_a_no_op_when_entry_moved_on() {
        // Simulates the race: a concurrent full recompute already replaced
        // the entry (fingerprint "c") before this stale refresh (based on
        // stale expectation "a") lands — it must not clobber "c"'s result.
        let cache = ResultCache::new(4);
        cache.put_query("q".into(), entry("a"));
        cache.put_query("q".into(), entry("c"));
        cache.refresh_query_fingerprint("q", &fp("a"), fp("b"));
        let got = cache.get_query("q").unwrap();
        assert_eq!(got.fingerprint, fp("c"), "stale refresh must not apply");
        assert_eq!(got.result.summary, "from c");
    }

    #[test]
    fn keys_distinguish_fingerprint_tool_and_args() {
        let a = ResultCache::tool_key(&fp("s1"), "grep", "{\"p\":1}");
        let b = ResultCache::tool_key(&fp("s2"), "grep", "{\"p\":1}");
        let c = ResultCache::tool_key(&fp("s1"), "find", "{\"p\":1}");
        assert_ne!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn tool_key_does_not_collide_across_tool_arg_boundary() {
        // "grep" + '{"pattern":"a#b"}' vs 'grep#{"pattern":"a' + 'b"}' — same
        // concatenation, different tool/args split; must not collide.
        let a = ResultCache::tool_key(&fp("s"), "grep", "{\"pattern\":\"a#b\"}");
        let b = ResultCache::tool_key(&fp("s"), "grep#{\"pattern\":\"a", "b\"}");
        assert_ne!(a, b);
    }

    #[test]
    fn query_key_normalizes_text() {
        let q1 = ExplorationQuery {
            text: "  Where Is Main ".to_string(),
            scope_hint: None,
            max_results: None,
        };
        let q2 = ExplorationQuery {
            text: "where is main".to_string(),
            scope_hint: None,
            max_results: None,
        };
        assert_eq!(ResultCache::query_key(&q1), ResultCache::query_key(&q2));
    }
}
