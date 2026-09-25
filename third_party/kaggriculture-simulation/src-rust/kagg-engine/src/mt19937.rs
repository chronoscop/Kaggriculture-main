//! CPython's `random.Random`, reproduced exactly.
//!
//! This is the foundation of the whole engine port. If the random stream
//! cannot be matched bit-for-bit, a Rust engine can only ever be an
//! approximation; if it can, the rest of the port is mechanical and a
//! differential harness can assert identical state at every step.
//!
//! The engine's stochastic surface is deliberately tiny, which is why this is
//! tractable at all. From `kaggriculture.py`:
//!
//! ```text
//! rng = random.Random((seed * 1_000_003) ^ day)   // RE-SEEDED EVERY DAY
//!   rng.random() < weed_chance      // ~100 empty tiles x 2 players per day
//!   rng.choice(sorted(SHOPS))       // once per shop-unlock day
//! ```
//!
//! Everything else in the interpreter is deterministic. Two details matter and
//! are easy to get wrong:
//!
//! 1. The per-day key `(seed * 1_000_003) ^ day` exceeds 2^32, so CPython seeds
//!    via `init_by_array` over the key's 32-bit little-endian words -- NOT
//!    `init_genrand` on a truncated value.
//! 2. `random()` is `genrand_res53`: two draws combined at 53-bit precision,
//!    not a single 32-bit draw scaled.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

pub struct MT {
    mt: [u32; N],
    mti: usize,
}

impl MT {
    fn init_genrand(s: u32) -> Self {
        let mut mt = [0u32; N];
        mt[0] = s;
        for i in 1..N {
            let prev = mt[i - 1];
            mt[i] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        MT { mt, mti: N }
    }

    /// CPython `init_by_array`, used for every integer seed.
    pub fn init_by_array(key: &[u32]) -> Self {
        let mut s = Self::init_genrand(19650218);
        let mut i: usize = 1;
        let mut j: usize = 0;
        let mut k = if N > key.len() { N } else { key.len() };
        while k > 0 {
            let prev = s.mt[i - 1];
            s.mt[i] = (s.mt[i] ^ (1664525u32.wrapping_mul(prev ^ (prev >> 30))))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                s.mt[0] = s.mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        let mut k = N - 1;
        while k > 0 {
            let prev = s.mt[i - 1];
            s.mt[i] = (s.mt[i] ^ (1566083941u32.wrapping_mul(prev ^ (prev >> 30))))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                s.mt[0] = s.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        s.mt[0] = UPPER_MASK;
        s.mti = N;
        s
    }

    /// Seed exactly as `random.Random(n)` does: absolute value, little-endian
    /// 32-bit words, and a single zero word when n == 0.
    pub fn seed_from_i128(n: i128) -> Self {
        let mut v = n.unsigned_abs();
        let mut key: Vec<u32> = Vec::new();
        while v > 0 {
            key.push((v & 0xffff_ffff) as u32);
            v >>= 32;
        }
        if key.is_empty() {
            key.push(0);
        }
        Self::init_by_array(&key)
    }

    /// The engine's per-day seeding: `random.Random((seed * 1_000_003) ^ day)`.
    pub fn for_day(seed: i64, day: i64) -> Self {
        let key = (seed as i128) * 1_000_003i128;
        // Python's ^ on a non-negative int is a plain bitwise xor; the engine's
        // seed is a non-negative 31-bit integer.
        Self::seed_from_i128(key ^ (day as i128))
    }

    pub fn genrand_u32(&mut self) -> u32 {
        if self.mti >= N {
            for kk in 0..(N - M) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ ((y & 1) * MATRIX_A);
            }
            for kk in (N - M)..(N - 1) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M - N] ^ (y >> 1) ^ ((y & 1) * MATRIX_A);
            }
            let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ ((y & 1) * MATRIX_A);
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `random.random()` -- genrand_res53. Two draws, 53-bit precision.
    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_u32() >> 5) as f64;
        let b = (self.genrand_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    /// `random.getrandbits(k)` for k <= 32.
    pub fn getrandbits(&mut self, k: u32) -> u32 {
        if k == 0 {
            return 0;
        }
        let k = k.min(32);
        self.genrand_u32() >> (32 - k)
    }

    /// `Random._randbelow_with_getrandbits(n)` -- rejection sampling on the
    /// bit length, which is what makes `choice()` reproducible.
    pub fn randbelow(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        // CPython uses `k = n.bit_length()`, NOT `(n-1).bit_length()`, and says
        // so explicitly: "don't use (n-1) here because n can be 1". The two
        // agree everywhere except exact powers of two, so this was invisible
        // while SHOPS was mistakenly 6 entries and broke the moment it became
        // the engine's real 8: k=4 draws in [0,15] and rejects >= 8, consuming
        // a different number of values than k=3 ever would.
        let k = 32 - n.leading_zeros(); // n.bit_length()
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }

    /// `random.choice(seq)`.
    pub fn choice<'a, T>(&mut self, seq: &'a [T]) -> &'a T {
        &seq[self.randbelow(seq.len() as u32) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values produced by CPython's `random.Random`.
    #[test]
    fn matches_cpython_random() {
        let mut r = MT::seed_from_i128(0);
        assert_eq!(r.random(), 0.8444218515250481);
        let mut d = MT::for_day(12345, 7);
        let got: Vec<u64> = (0..3).map(|_| d.random().to_bits()).collect();
        assert_eq!(
            got,
            vec![
                4605057039300141500,
                4606521014199147327,
                4605305558552737764
            ]
        );
        let mut b = MT::for_day(12345, 7);
        let bits: Vec<u32> = (0..3).map(|_| b.getrandbits(32)).collect();
        assert_eq!(bits, vec![3281507371, 2036887351, 3979585020]);
    }

    #[test]
    fn choice_matches_cpython() {
        let shops = crate::engine::SHOPS_SORTED;
        let mut c = MT::for_day(987654321, 2);
        let got: Vec<&str> = (0..4).map(|_| *c.choice(&shops)).collect();
        assert_eq!(
            got,
            vec!["PIZZA_SHOP", "SMOOTHIE_SHOP", "BAKERY", "BRUNCH_SPOT"]
        );
    }
}
