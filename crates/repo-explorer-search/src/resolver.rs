//! Pure rg-resolution logic: pick the `rg` binary, resolving an explicit
//! config path first and closure-injected PATH detection second. No subprocess
//! is spawned here, so resolution is unit-testable with no real binary (mirrors
//! `repo-explorer-memory::freshness`).

use std::path::{Path, PathBuf};

/// Resolve the rg binary: an explicit config path wins over PATH detection
/// (config doc says `None` -> auto-detect); a bad explicit path is trusted
/// as-is and surfaces as a spawn failure at run time, not here. Returns `None`
/// when rg cannot be resolved at all.
pub(crate) fn resolve_rg(
    rg_path: Option<&Path>,
    find_rg: impl Fn() -> Option<PathBuf>,
) -> Option<PathBuf> {
    match rg_path {
        Some(p) => Some(p.to_path_buf()),
        None => find_rg(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn some_rg() -> Option<PathBuf> {
        Some(PathBuf::from("/found/rg"))
    }
    fn none() -> Option<PathBuf> {
        None
    }

    #[test]
    fn explicit_path_wins_over_finder() {
        // The finder would return a different path, but the explicit config
        // path wins and the finder is not consulted.
        let resolved = resolve_rg(Some(Path::new("/cfg/rg")), some_rg).unwrap();
        assert_eq!(resolved, PathBuf::from("/cfg/rg"));
    }

    #[test]
    fn finder_used_when_no_explicit_path() {
        let resolved = resolve_rg(None, some_rg).unwrap();
        assert_eq!(resolved, PathBuf::from("/found/rg"));
    }

    #[test]
    fn none_when_unresolved() {
        assert!(resolve_rg(None, none).is_none());
    }
}
