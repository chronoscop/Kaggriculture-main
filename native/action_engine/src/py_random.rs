//! CPython-compatible `random.Random` (MT19937).
//!
//! The official engine seeds `random.Random((seed * 1_000_003) ^ day)` per day and uses only
//! `random()` (weed spawns) and `choice()` (town shop unlocks). Bit-exact parity with CPython
//! is required for episode reproduction, so this reimplements CPython's `_randommodule.c`
//! seeding (`init_by_array` over 32-bit words of the seed) and `random_random` (53-bit doubles),
//! plus the `getrandbits`-based rejection sampling behind `choice`.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

pub struct PyRandom {
    mt: [u32; N],
    index: usize,
}

impl PyRandom {
    /// Equivalent to `random.Random(seed)` for a non-negative integer seed.
    pub fn new(seed: u64) -> Self {
        // CPython splits the absolute seed value into little-endian 32-bit words
        // and feeds them to init_by_array; a zero seed still yields one word.
        let mut key = Vec::new();
        let mut s = seed;
        loop {
            key.push((s & 0xffff_ffff) as u32);
            s >>= 32;
            if s == 0 {
                break;
            }
        }
        let mut rng = Self {
            mt: [0; N],
            index: N,
        };
        rng.init_by_array(&key);
        rng
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            self.mt[i] = (1_812_433_253u32.wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 30)))
                .wrapping_add(i as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let mut i: usize = 1;
        let mut j: usize = 0;
        let mut k = N.max(key.len());
        while k > 0 {
            self.mt[i] = (self.mt[i]
                ^ (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1_664_525))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = N - 1;
        while k > 0 {
            self.mt[i] = (self.mt[i]
                ^ (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        self.mt[0] = 0x8000_0000;
    }

    fn genrand_u32(&mut self) -> u32 {
        if self.index >= N {
            for i in 0..N {
                let y = (self.mt[i] & UPPER_MASK) | (self.mt[(i + 1) % N] & LOWER_MASK);
                let mut next = self.mt[(i + M) % N] ^ (y >> 1);
                if y & 1 != 0 {
                    next ^= MATRIX_A;
                }
                self.mt[i] = next;
            }
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// Equivalent to `random.Random.random()`: 53-bit uniform double in [0, 1).
    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_u32() >> 5) as u64; // 27 bits
        let b = (self.genrand_u32() >> 6) as u64; // 26 bits
        (a * 67_108_864 + b) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Equivalent to `random.Random.getrandbits(k)` for 0 < k <= 32.
    pub fn getrandbits(&mut self, k: u32) -> u32 {
        debug_assert!(0 < k && k <= 32);
        self.genrand_u32() >> (32 - k)
    }

    /// Equivalent to `random.Random._randbelow(n)` for n > 0: rejection sampling on bit_length(n) bits.
    pub fn randbelow(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0);
        let k = 32 - n.leading_zeros();
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }

    /// Equivalent to `random.Random.choice(seq)`, returning the chosen index.
    pub fn choice_index(&mut self, len: usize) -> usize {
        self.randbelow(len as u32) as usize
    }
}

#[cfg(test)]
// Ground-truth constants are pasted verbatim from CPython's repr output.
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    // Ground truth generated with CPython 3.12:
    //   r = random.Random(seed); [r.random() for _ in range(5)]
    //   r = random.Random(seed); [r._randbelow(n) for n in (8, 8, 5, 3, 100)]
    const RANDOM_CASES: &[(u64, [f64; 5])] = &[
        (
            0,
            [
                0.84442185152504812,
                0.75795440294030247,
                0.420571580830845,
                0.25891675029296335,
                0.51127472136860852,
            ],
        ),
        (
            1,
            [
                0.13436424411240122,
                0.84743373693723267,
                0.76377461897661403,
                0.2550690257394217,
                0.49543508709194095,
            ],
        ),
        (
            42,
            [
                0.63942679845788375,
                0.025010755222666936,
                0.27502931836911926,
                0.22321073814882275,
                0.7364712141640124,
            ],
        ),
        // (42 * 1_000_003) ^ 5 — the engine's per-day seed shape
        (
            42_000_123,
            [
                0.773128372030078,
                0.052250901057513177,
                0.68037689296618575,
                0.46887793713904191,
                0.70854445013113887,
            ],
        ),
        // (2**31 - 1) * 1_000_003 ^ 29 — max 31-bit seed, exercises the two-word key path
        (
            2_147_490_089_450_912,
            [
                0.54463874003150792,
                0.008264779804634026,
                0.70739533559687051,
                0.33701357809653454,
                0.47125210331532974,
            ],
        ),
    ];

    const RANDBELOW_BOUNDS: [u32; 5] = [8, 8, 5, 3, 100];
    const RANDBELOW_CASES: &[(u64, [u32; 5])] = &[
        (0, [6, 6, 0, 1, 65]),
        (1, [2, 1, 2, 0, 63]),
        (42, [1, 0, 2, 0, 28]),
        (42_000_123, [0, 3, 3, 1, 90]),
        (2_147_490_089_450_912, [4, 0, 2, 1, 79]),
    ];

    #[test]
    fn random_matches_cpython() {
        for (seed, expected) in RANDOM_CASES {
            let mut rng = PyRandom::new(*seed);
            for (i, want) in expected.iter().enumerate() {
                let got = rng.random();
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "seed={seed} draw={i}: {got} != {want}"
                );
            }
        }
    }

    #[test]
    fn randbelow_matches_cpython() {
        for (seed, expected) in RANDBELOW_CASES {
            let mut rng = PyRandom::new(*seed);
            for (n, want) in RANDBELOW_BOUNDS.iter().zip(expected) {
                assert_eq!(rng.randbelow(*n), *want, "seed={seed} n={n}");
            }
        }
    }
}
