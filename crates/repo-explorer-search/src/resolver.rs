//! Pure rtk-resolution logic: pick the `rtk` binary, resolving an explicit
//! config path first and closure-injected PATH detection second. No subprocess
//! is spawned here, so resolution is unit-testable with no real binary (mirrors
//! `repo-explorer-memory::freshness`).

use std::path::{Path, PathBuf};

/// Resolve the rtk binary: an explicit config path wins over PATH detection
/// (config doc says `None` -> auto-detect); a bad explicit path is trusted
/// as-is and surfaces as a spawn failure at run time, not here. Returns `None`
/// when rtk cannot be resolved at all.
pub(crate) fn resolve_rtk(
    rtk_path: Option<&Path>,
    find_rtk: impl Fn() -> Option<PathBuf>,
) -> Option<PathBuf> {
    match rtk_path {
        Some(p) => Some(p.to_path_buf()),
        None => find_rtk(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn some_rtk() -> Option<PathBuf> {
        Some(PathBuf::from("/found/rtk"))
    }
    fn none() -> Option<PathBuf> {
        None
    }

    #[test]
    fn explicit_path_wins_over_finder() {
        // The finder would return a different path, but the explicit config
        // path wins and the finder is not consulted.
        let resolved = resolve_rtk(Some(Path::new("/cfg/rtk")), some_rtk).unwrap();
        assert_eq!(resolved, PathBuf::from("/cfg/rtk"));
    }

    #[test]
    fn finder_used_when_no_explicit_path() {
        let resolved = resolve_rtk(None, some_rtk).unwrap();
        assert_eq!(resolved, PathBuf::from("/found/rtk"));
    }

    #[test]
    fn none_when_unresolved() {
        assert!(resolve_rtk(None, none).is_none());
    }
}
