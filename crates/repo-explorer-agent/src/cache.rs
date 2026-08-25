//! In-memory, per-process result caches keyed by repository fingerprint:
//! tool-result memoization, retrieval-leg memoization, and the query→result
//! cache with path-level invalidation. Interior-mutable (`Mutex`) because the
//! agent is shared behind an `Arc` and every entry point takes `&self`.

use repo_explorer_core::domain::{
    Candidate, ExplorationFinding, ExplorationQuery, ExplorationResult,
};
use repo_explorer_core::fingerprint::RepoFingerprint;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
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
        if self.map.insert(key.clone(), value).is_none() {
            self.order.push_back(key);
            while self.map.len() > self.cap {
                if let Some(oldest) = self.order.pop_front() {
                    self.map.remove(&oldest);
                }
            }
        }
    }

    fn remove(&mut self, key: &str) {
        self.map.remove(key);
        self.order.retain(|k| k != key);
    }
}

/// One cached query result plus what it depends on.
#[derive(Debug, Clone)]
pub(crate) struct QueryEntry {
    pub fingerprint: RepoFingerprint,
    /// Repo-relative paths the result was derived from; a change to any of
    /// them invalidates the entry.
    pub paths: HashSet<PathBuf>,
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

    pub(crate) fn tool_key(fp: &RepoFingerprint, tool: &str, args_json: &str) -> String {
        format!("{}#{}#{tool}#{args_json}", fp.head_sha, fp.dirty_hash)
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
    /// stored fingerprint, not the key.
    ///
    /// Each field is length-prefixed (`len:content`) rather than joined with a
    /// bare delimiter, so differing field boundaries can never hash-collide
    /// regardless of what characters the fields themselves contain.
    pub(crate) fn query_key(query: &ExplorationQuery) -> String {
        let text = query.text.trim().to_lowercase();
        let scope = query
            .scope_hint
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        let max_results = query.max_results.map(|m| m.to_string()).unwrap_or_default();
        format!(
            "{}:{text}{}:{scope}{}:{max_results}",
            text.len(),
            scope.len(),
            max_results.len()
        )
    }

    pub(crate) fn get_query(&self, key: &str) -> Option<QueryEntry> {
        self.lock().queries.get(key)
    }

    pub(crate) fn put_query(&self, key: String, entry: QueryEntry) {
        self.lock().queries.insert(key, entry);
    }

    /// Keep a still-valid entry current after the repo moved without touching
    /// its contributing paths.
    pub(crate) fn refresh_query_fingerprint(&self, key: &str, fingerprint: RepoFingerprint) {
        let mut inner = self.lock();
        if let Some(entry) = inner.queries.get_mut(key) {
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

    fn entry(sha: &str, path: &str) -> QueryEntry {
        QueryEntry {
            fingerprint: fp(sha),
            paths: HashSet::from([PathBuf::from(path)]),
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
        cache.put_query("q".into(), entry("a", "src/x.rs"));
        cache.refresh_query_fingerprint("q", fp("b"));
        assert_eq!(cache.get_query("q").unwrap().fingerprint, fp("b"));
        cache.remove_query("q");
        assert!(cache.get_query("q").is_none());
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
