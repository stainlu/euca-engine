//! Platform-abstracted SIMD primitives for 4-wide f32 operations.
//!
//! Provides a minimal `f32x4` wrapper over `__m128` (x86_64 SSE) or
//! `float32x4_t` (aarch64 NEON). All functions are `#[inline(always)]` so they
//! compile down to single instructions.

// ── x86_64 SSE implementation ───────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod arch {
    use core::arch::x86_64::*;

    /// 4-wide f32 SIMD register.
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    #[allow(non_camel_case_types)]
    pub struct f32x4(pub __m128);

    #[allow(dead_code)]
    impl f32x4 {
        /// Load four f32 values into a SIMD register.
        #[inline(always)]
        pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
            // Safety: SSE is always available on x86_64.
            // _mm_set_ps takes arguments in reverse order: (d, c, b, a) -> [a, b, c, d]
            Self(unsafe { _mm_set_ps(d, c, b, a) })
        }

        /// Splat a single f32 across all four lanes.
        #[inline(always)]
        pub fn splat(v: f32) -> Self {
            // Safety: SSE is always available on x86_64.
            Self(unsafe { _mm_set1_ps(v) })
        }

        /// Component-wise addition.
        #[inline(always)]
        pub fn add(self, rhs: Self) -> Self {
            // Safety: SSE is always available on x86_64.
            Self(unsafe { _mm_add_ps(self.0, rhs.0) })
        }

        /// Component-wise subtraction.
        #[inline(always)]
        pub fn sub(self, rhs: Self) -> Self {
            // Safety: SSE is always available on x86_64.
            Self(unsafe { _mm_sub_ps(self.0, rhs.0) })
        }

        /// Component-wise multiplication.
        #[inline(always)]
        pub fn mul(self, rhs: Self) -> Self {
            // Safety: SSE is always available on x86_64.
            Self(unsafe { _mm_mul_ps(self.0, rhs.0) })
        }

        /// Fused multiply-add: `self * a + b`. Falls back to mul+add on SSE.
        #[inline(always)]
        pub fn mul_add(self, a: Self, b: Self) -> Self {
            // self * a + b
            // On x86_64 without FMA, this is mul then add.
            // Safety: SSE is always available on x86_64.
            Self(unsafe { _mm_add_ps(_mm_mul_ps(self.0, a.0), b.0) })
        }

        /// Component-wise square root.
        #[inline(always)]
        pub fn sqrt(self) -> Self {
            // Safety: SSE is always available on x86_64.
            Self(unsafe { _mm_sqrt_ps(self.0) })
        }

        /// Component-wise negation.
        #[inline(always)]
        pub fn neg(self) -> Self {
            // Safety: SSE is always available on x86_64. XOR with sign-bit mask.
            Self(unsafe { _mm_xor_ps(self.0, _mm_set1_ps(-0.0)) })
        }

        /// Component-wise minimum.
        #[inline(always)]
        pub fn min(self, rhs: Self) -> Self {
            // Safety: SSE is always available on x86_64.
            Self(unsafe { _mm_min_ps(self.0, rhs.0) })
        }

        /// Component-wise maximum.
        #[inline(always)]
        pub fn max(self, rhs: Self) -> Self {
            // Safety: SSE is always available on x86_64.
            Self(unsafe { _mm_max_ps(self.0, rhs.0) })
        }

        /// Horizontal sum of all four lanes: a + b + c + d.
        #[inline(always)]
        pub fn horizontal_sum(self) -> f32 {
            // Safety: SSE is always available on x86_64.
            // Uses two shuffle+add sequences for SSE2 compatibility.
            unsafe {
                // [a, b, c, d] -> shuffle high pair down: [c, d, c, d]
                let hi = _mm_movehl_ps(self.0, self.0);
                // [a+c, b+d, ?, ?]
                let sum1 = _mm_add_ps(self.0, hi);
                // shuffle lane 1 into lane 0: [b+d, ?, ?, ?]
                let shuf = _mm_shuffle_ps(sum1, sum1, 0x01);
                // [a+c+b+d, ?, ?, ?]
                let sum2 = _mm_add_ss(sum1, shuf);
                _mm_cvtss_f32(sum2)
            }
        }

        /// Extract lane 0.
        #[inline(always)]
        pub fn x(self) -> f32 {
            // Safety: SSE is always available on x86_64.
            unsafe { _mm_cvtss_f32(self.0) }
        }

        /// Extract lane 1.
        #[inline(always)]
        pub fn y(self) -> f32 {
            // Safety: SSE is always available on x86_64.
            unsafe {
                let shuf = _mm_shuffle_ps(self.0, self.0, 0x55); // splat lane 1
                _mm_cvtss_f32(shuf)
            }
        }

        /// Extract lane 2.
        #[inline(always)]
        pub fn z(self) -> f32 {
            // Safety: SSE is always available on x86_64.
            unsafe {
                let shuf = _mm_shuffle_ps(self.0, self.0, 0xAA); // splat lane 2
                _mm_cvtss_f32(shuf)
            }
        }

        /// Extract lane 3.
        #[inline(always)]
        pub fn w(self) -> f32 {
            // Safety: SSE is always available on x86_64.
            unsafe {
                let shuf = _mm_shuffle_ps(self.0, self.0, 0xFF); // splat lane 3
                _mm_cvtss_f32(shuf)
            }
        }

        /// Shuffle: select lanes from self and rhs.
        /// `MASK` uses the same encoding as `_mm_shuffle_ps`:
        /// bits [1:0] select from self for result lane 0,
        /// bits [3:2] select from self for result lane 1,
        /// bits [5:4] select from rhs for result lane 2,
        /// bits [7:6] select from rhs for result lane 3.
        #[inline(always)]
        pub fn shuffle<const MASK: i32>(self, rhs: Self) -> Self {
            // Safety: SSE is always available on x86_64.
            Self(unsafe { _mm_shuffle_ps(self.0, rhs.0, MASK) })
        }

        /// Duplicate lane 0 across all lanes: [x, x, x, x].
        #[inline(always)]
        pub fn splat_x(self) -> Self {
            self.shuffle::<0x00>(self)
        }

        /// Duplicate lane 1 across all lanes: [y, y, y, y].
        #[inline(always)]
        pub fn splat_y(self) -> Self {
            self.shuffle::<0x55>(self)
        }

        /// Duplicate lane 2 across all lanes: [z, z, z, z].
        #[inline(always)]
        pub fn splat_z(self) -> Self {
            self.shuffle::<0xAA>(self)
        }

        /// Duplicate lane 3 across all lanes: [w, w, w, w].
        #[inline(always)]
        pub fn splat_w(self) -> Self {
            self.shuffle::<0xFF>(self)
        }

        /// XOR two registers (used for sign-flipping).
        #[inline(always)]
        pub fn xor(self, rhs: Self) -> Self {
            // Safety: SSE is always available on x86_64.
            Self(unsafe { _mm_xor_ps(self.0, rhs.0) })
        }
    }
}

// ── aarch64 NEON implementation ─────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
mod arch {
    use core::arch::aarch64::*;

    /// 4-wide f32 SIMD register.
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    #[allow(non_camel_case_types)]
    pub struct f32x4(pub float32x4_t);

    #[allow(dead_code)]
    impl f32x4 {
        /// Load four f32 values into a SIMD register.
        #[inline(always)]
        pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
            // Safety: NEON is always available on aarch64.
            // Creates [a, b, c, d] in lane order.
            let arr = [a, b, c, d];
            Self(unsafe { vld1q_f32(arr.as_ptr()) })
        }

        /// Splat a single f32 across all four lanes.
        #[inline(always)]
        pub fn splat(v: f32) -> Self {
            // Safety: NEON is always available on aarch64.
            Self(unsafe { vdupq_n_f32(v) })
        }

        /// Component-wise addition.
        #[inline(always)]
        pub fn add(self, rhs: Self) -> Self {
            // Safety: NEON is always available on aarch64.
            Self(unsafe { vaddq_f32(self.0, rhs.0) })
        }

        /// Component-wise subtraction.
        #[inline(always)]
        pub fn sub(self, rhs: Self) -> Self {
            // Safety: NEON is always available on aarch64.
            Self(unsafe { vsubq_f32(self.0, rhs.0) })
        }

        /// Component-wise multiplication.
        #[inline(always)]
        pub fn mul(self, rhs: Self) -> Self {
            // Safety: NEON is always available on aarch64.
            Self(unsafe { vmulq_f32(self.0, rhs.0) })
        }

        /// Fused multiply-add: `self * a + b`.
        #[inline(always)]
        pub fn mul_add(self, a: Self, b: Self) -> Self {
            // Safety: NEON is always available on aarch64.
            // vfmaq_f32(addend, factor1, factor2) = addend + factor1 * factor2
            Self(unsafe { vfmaq_f32(b.0, self.0, a.0) })
        }

        /// Component-wise square root.
        #[inline(always)]
        pub fn sqrt(self) -> Self {
            // Safety: NEON is always available on aarch64.
            Self(unsafe { vsqrtq_f32(self.0) })
        }

        /// Component-wise negation.
        #[inline(always)]
        pub fn neg(self) -> Self {
            // Safety: NEON is always available on aarch64.
            Self(unsafe { vnegq_f32(self.0) })
        }

        /// Component-wise minimum.
        #[inline(always)]
        pub fn min(self, rhs: Self) -> Self {
            // Safety: NEON is always available on aarch64.
            Self(unsafe { vminq_f32(self.0, rhs.0) })
        }

        /// Component-wise maximum.
        #[inline(always)]
        pub fn max(self, rhs: Self) -> Self {
            // Safety: NEON is always available on aarch64.
            Self(unsafe { vmaxq_f32(self.0, rhs.0) })
        }

        /// Horizontal sum of all four lanes: a + b + c + d.
        #[inline(always)]
        pub fn horizontal_sum(self) -> f32 {
            // Safety: NEON is always available on aarch64.
            // vaddvq_f32 sums all four lanes in one instruction on aarch64.
            unsafe { vaddvq_f32(self.0) }
        }

        /// Extract lane 0.
        #[inline(always)]
        pub fn x(self) -> f32 {
            // Safety: NEON is always available on aarch64.
            unsafe { vgetq_lane_f32(self.0, 0) }
        }

        /// Extract lane 1.
        #[inline(always)]
        pub fn y(self) -> f32 {
            // Safety: NEON is always available on aarch64.
            unsafe { vgetq_lane_f32(self.0, 1) }
        }

        /// Extract lane 2.
        #[inline(always)]
        pub fn z(self) -> f32 {
            // Safety: NEON is always available on aarch64.
            unsafe { vgetq_lane_f32(self.0, 2) }
        }

        /// Extract lane 3.
        #[inline(always)]
        pub fn w(self) -> f32 {
            // Safety: NEON is always available on aarch64.
            unsafe { vgetq_lane_f32(self.0, 3) }
        }

        /// Duplicate lane 0 across all lanes: [x, x, x, x].
        #[inline(always)]
        pub fn splat_x(self) -> Self {
            // Safety: NEON is always available on aarch64.
            Self(unsafe { vdupq_laneq_f32(self.0, 0) })
        }

        /// Duplicate lane 1 across all lanes: [y, y, y, y].
        #[inline(always)]
        pub fn splat_y(self) -> Self {
            // Safety: NEON is always available on aarch64.
            Self(unsafe { vdupq_laneq_f32(self.0, 1) })
        }

        /// Duplicate lane 2 across all lanes: [z, z, z, z].
        #[inline(always)]
        pub fn splat_z(self) -> Self {
            // Safety: NEON is always available on aarch64.
            Self(unsafe { vdupq_laneq_f32(self.0, 2) })
        }

        /// Duplicate lane 3 across all lanes: [w, w, w, w].
        #[inline(always)]
        pub fn splat_w(self) -> Self {
            // Safety: NEON is always available on aarch64.
            Self(unsafe { vdupq_laneq_f32(self.0, 3) })
        }

        /// XOR two registers (used for sign-flipping).
        #[inline(always)]
        pub fn xor(self, rhs: Self) -> Self {
            // Safety: NEON is always available on aarch64.
            // Reinterpret as u32, XOR, reinterpret back.
            Self(unsafe {
                vreinterpretq_f32_u32(veorq_u32(
                    vreinterpretq_u32_f32(self.0),
                    vreinterpretq_u32_f32(rhs.0),
                ))
            })
        }
    }
}

pub use arch::f32x4;
