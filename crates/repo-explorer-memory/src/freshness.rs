//! Pure, `rmcp`-free staleness decision for `ensure_fresh_index`.

use std::time::{Duration, SystemTime};

/// A snapshot of the project's index state, assembled from `index_status` and
/// `detect_changes` before deciding whether to re-index.
pub(crate) struct IndexProbe {
    /// Is the project indexed at all?
    pub exists: bool,
    /// When the index was last built, if known.
    pub last_indexed_at: Option<SystemTime>,
    /// Number of changed files reported by `detect_changes`.
    pub changed_files: usize,
}

/// The re-index decision.
pub(crate) enum FreshnessDecision {
    Reindex,
    UpToDate,
}

/// Decide whether to re-index. Re-index when the project is not indexed, OR any
/// file changed, OR the index age exceeds `staleness`, OR the last-index time is
/// unknown; otherwise up-to-date. `age == staleness` counts as up-to-date; a
/// last-index time in the future (clock skew) is treated as up-to-date.
pub(crate) fn decide_freshness(
    probe: &IndexProbe,
    staleness: Duration,
    now: SystemTime,
) -> FreshnessDecision {
    if !probe.exists {
        return FreshnessDecision::Reindex;
    }
    if probe.changed_files > 0 {
        return FreshnessDecision::Reindex;
    }
    match probe.last_indexed_at {
        Some(t) => match now.duration_since(t) {
            Ok(age) if age > staleness => FreshnessDecision::Reindex,
            _ => FreshnessDecision::UpToDate,
        },
        None => FreshnessDecision::Reindex,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    fn base_now() -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000)
    }

    #[test]
    fn not_indexed_forces_reindex() {
        let probe = IndexProbe {
            exists: false,
            last_indexed_at: None,
            changed_files: 0,
        };
        assert!(matches!(
            decide_freshness(&probe, Duration::from_secs(3600), base_now()),
            FreshnessDecision::Reindex
        ));
    }

    #[test]
    fn changed_files_force_reindex_even_if_fresh() {
        let now = base_now();
        let probe = IndexProbe {
            exists: true,
            last_indexed_at: Some(now - Duration::from_secs(1)),
            changed_files: 3,
        };
        assert!(matches!(
            decide_freshness(&probe, Duration::from_secs(3600), now),
            FreshnessDecision::Reindex
        ));
    }

    #[test]
    fn age_over_threshold_reindexes() {
        let now = base_now();
        let probe = IndexProbe {
            exists: true,
            last_indexed_at: Some(now - Duration::from_secs(3601)),
            changed_files: 0,
        };
        assert!(matches!(
            decide_freshness(&probe, Duration::from_secs(3600), now),
            FreshnessDecision::Reindex
        ));
    }

    #[test]
    fn age_under_threshold_is_up_to_date() {
        let now = base_now();
        let probe = IndexProbe {
            exists: true,
            last_indexed_at: Some(now - Duration::from_secs(10)),
            changed_files: 0,
        };
        assert!(matches!(
            decide_freshness(&probe, Duration::from_secs(3600), now),
            FreshnessDecision::UpToDate
        ));
    }

    #[test]
    fn exact_boundary_is_up_to_date() {
        // age == staleness is NOT "> staleness", so up-to-date.
        let now = base_now();
        let probe = IndexProbe {
            exists: true,
            last_indexed_at: Some(now - Duration::from_secs(3600)),
            changed_files: 0,
        };
        assert!(matches!(
            decide_freshness(&probe, Duration::from_secs(3600), now),
            FreshnessDecision::UpToDate
        ));
    }

    #[test]
    fn unknown_age_reindexes() {
        let probe = IndexProbe {
            exists: true,
            last_indexed_at: None,
            changed_files: 0,
        };
        assert!(matches!(
            decide_freshness(&probe, Duration::from_secs(3600), base_now()),
            FreshnessDecision::Reindex
        ));
    }

    #[test]
    fn future_timestamp_is_up_to_date() {
        // duration_since errors when t is after now -> treat as fresh.
        let now = base_now();
        let probe = IndexProbe {
            exists: true,
            last_indexed_at: Some(now + Duration::from_secs(60)),
            changed_files: 0,
        };
        assert!(matches!(
            decide_freshness(&probe, Duration::from_secs(3600), now),
            FreshnessDecision::UpToDate
        ));
    }
}
