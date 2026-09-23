//! Deterministic scalar hashing (`splitmix64`, `fnv1a64`) and a tiny
//! splitmix64-based PRNG. No external RNG crate: every stream must be a pure
//! function of its seed so a run is byte-reproducible.

/// The splitmix64 mixing function: one full round on `x`. Used both to derive
/// a seed from a hash and as `Rng`'s state advance.
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// 64-bit FNV-1a over `bytes`.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// A splitmix64 generator. `Rng::new(seed).next_u64()` equals `splitmix64(seed)`.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_known_answers() {
        // Canonical splitmix64: first output for seed 0.
        assert_eq!(splitmix64(0), 0xE220A8397B1DCDAF);
        assert_eq!(Rng::new(0).next_u64(), 0xE220A8397B1DCDAF);
    }

    #[test]
    fn fnv1a64_known_answers() {
        assert_eq!(fnv1a64(b""), 0xcbf29ce484222325);
        assert_eq!(fnv1a64(b"a"), 0xaf63dc4c8601ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x85944171f73967e8);
    }

    #[test]
    fn stream_is_deterministic_for_a_seed() {
        let mut a = Rng::new(splitmix64(42 ^ fnv1a64(b"ripgrep")));
        let mut b = Rng::new(splitmix64(42 ^ fnv1a64(b"ripgrep")));
        for _ in 0..8 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }
}
