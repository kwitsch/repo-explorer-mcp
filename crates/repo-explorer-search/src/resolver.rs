//! Pure binary-selection logic: pick which of `rtk`/`ripgrep` to run, resolving
//! explicit config paths first and closure-injected PATH detection second. No
//! subprocess is spawned here, so the fallback logic is unit-testable with no
//! real binaries (mirrors `repo-explorer-memory::freshness`).

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tool {
    Rtk,
    Ripgrep,
}

pub(crate) struct ResolvedBackend {
    pub tool: Tool,
    pub path: PathBuf,
}

/// Pick which binary to run. An explicit config path wins over PATH detection
/// for that tool (config doc says `None` -> auto-detect); a bad explicit path
/// is trusted as-is and surfaces as a spawn failure at run time, not here.
/// `prefer_rtk == true` tries rtk then rg; `false` tries rg then rtk. Returns
/// `None` when neither tool resolves.
pub(crate) fn resolve_backend(
    rtk_path: Option<&Path>,
    ripgrep_path: Option<&Path>,
    prefer_rtk: bool,
    find_rtk: impl Fn() -> Option<PathBuf>,
    find_rg: impl Fn() -> Option<PathBuf>,
) -> Option<ResolvedBackend> {
    let try_rtk = || -> Option<ResolvedBackend> {
        let path = match rtk_path {
            Some(p) => p.to_path_buf(),
            None => find_rtk()?,
        };
        Some(ResolvedBackend {
            tool: Tool::Rtk,
            path,
        })
    };
    let try_rg = || -> Option<ResolvedBackend> {
        let path = match ripgrep_path {
            Some(p) => p.to_path_buf(),
            None => find_rg()?,
        };
        Some(ResolvedBackend {
            tool: Tool::Ripgrep,
            path,
        })
    };
    if prefer_rtk {
        try_rtk().or_else(try_rg)
    } else {
        try_rg().or_else(try_rtk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn some_rtk() -> Option<PathBuf> {
        Some(PathBuf::from("/found/rtk"))
    }
    fn some_rg() -> Option<PathBuf> {
        Some(PathBuf::from("/found/rg"))
    }
    fn none() -> Option<PathBuf> {
        None
    }

    #[test]
    fn prefer_rtk_present_picks_rtk() {
        let r = resolve_backend(None, None, true, some_rtk, some_rg).unwrap();
        assert_eq!(r.tool, Tool::Rtk);
        assert_eq!(r.path, PathBuf::from("/found/rtk"));
    }

    #[test]
    fn prefer_rtk_but_absent_falls_back_to_rg() {
        let r = resolve_backend(None, None, true, none, some_rg).unwrap();
        assert_eq!(r.tool, Tool::Ripgrep);
        assert_eq!(r.path, PathBuf::from("/found/rg"));
    }

    #[test]
    fn prefer_rg_picks_rg_first() {
        let r = resolve_backend(None, None, false, some_rtk, some_rg).unwrap();
        assert_eq!(r.tool, Tool::Ripgrep);
    }

    #[test]
    fn prefer_rg_but_absent_falls_back_to_rtk() {
        let r = resolve_backend(None, None, false, some_rtk, none).unwrap();
        assert_eq!(r.tool, Tool::Rtk);
    }

    #[test]
    fn explicit_rtk_path_overrides_finder() {
        // finder returns a different path; the explicit config path wins and the
        // finder is not consulted for rtk.
        let explicit = PathBuf::from("/cfg/rtk");
        let r = resolve_backend(Some(Path::new("/cfg/rtk")), None, true, none, some_rg).unwrap();
        assert_eq!(r.tool, Tool::Rtk);
        assert_eq!(r.path, explicit);
    }

    #[test]
    fn neither_present_returns_none() {
        assert!(resolve_backend(None, None, true, none, none).is_none());
    }
}
