// BLIS-style dgemm for WASM SIMD via xsimd.
// 5-loop structure with 8×4 micro-kernel optimized for 128-bit SIMD (2 doubles/vector).
// 16 vector accumulators fully utilize the WASM v128 register file.

#include <xsimd/xsimd.hpp>
#include <cstring>
#include <algorithm>
#include <vector>

using vd = xsimd::batch<double>;
static constexpr int VL = vd::size; // 2 for 128-bit

static constexpr int MR = 8;
static constexpr int NR = 4;
static constexpr int MC = 128;
static constexpr int KC = 256;
static constexpr int NC = 512;

// 8×4 micro-kernel: 16 vector accumulators (4 cols × 4 vectors per col)
// A is packed MR-contiguous, B is packed NR-contiguous.
static inline void micro_8x4(
    int kc,
    const double* __restrict__ pA,
    const double* __restrict__ pB,
    double* __restrict__ C, int ldc,
    double alpha, double beta,
    int mr, int nr)
{
    // 16 accumulators: c[col][row_vec]
    vd c00{0}, c10{0}, c20{0}, c30{0};
    vd c01{0}, c11{0}, c21{0}, c31{0};
    vd c02{0}, c12{0}, c22{0}, c32{0};
    vd c03{0}, c13{0}, c23{0}, c33{0};

    for (int p = 0; p < kc; ++p) {
        // Load 4 vectors from A (8 doubles)
        vd a0 = vd::load_unaligned(pA);
        vd a1 = vd::load_unaligned(pA + VL);
        vd a2 = vd::load_unaligned(pA + 2 * VL);
        vd a3 = vd::load_unaligned(pA + 3 * VL);

        // Broadcast B elements
        vd b0(pB[0]), b1(pB[1]), b2(pB[2]), b3(pB[3]);

        c00 = xsimd::fma(a0, b0, c00); c10 = xsimd::fma(a1, b0, c10);
        c20 = xsimd::fma(a2, b0, c20); c30 = xsimd::fma(a3, b0, c30);
        c01 = xsimd::fma(a0, b1, c01); c11 = xsimd::fma(a1, b1, c11);
        c21 = xsimd::fma(a2, b1, c21); c31 = xsimd::fma(a3, b1, c31);
        c02 = xsimd::fma(a0, b2, c02); c12 = xsimd::fma(a1, b2, c12);
        c22 = xsimd::fma(a2, b2, c22); c32 = xsimd::fma(a3, b2, c32);
        c03 = xsimd::fma(a0, b3, c03); c13 = xsimd::fma(a1, b3, c13);
        c23 = xsimd::fma(a2, b3, c23); c33 = xsimd::fma(a3, b3, c33);

        pA += MR;
        pB += NR;
    }

    // Store to C, handling edge cases (mr < MR or nr < NR)
    vd va(alpha);
    const vd* acc[4][4] = {
        {&c00, &c10, &c20, &c30},
        {&c01, &c11, &c21, &c31},
        {&c02, &c12, &c22, &c32},
        {&c03, &c13, &c23, &c33},
    };

    for (int j = 0; j < nr; ++j) {
        double* col = C + j * ldc;
        for (int iv = 0; iv < (mr + VL - 1) / VL; ++iv) {
            int row = iv * VL;
            int remain = std::min(VL, mr - row);
            vd val = va * (*acc[j][iv]);

            if (beta == 0.0) {
                if (remain == VL) {
                    val.store_unaligned(col + row);
                } else {
                    for (int r = 0; r < remain; ++r) col[row + r] = val.get(r);
                }
            } else {
                vd vb(beta);
                if (remain == VL) {
                    vd old = vd::load_unaligned(col + row);
                    (xsimd::fma(vb, old, val)).store_unaligned(col + row);
                } else {
                    for (int r = 0; r < remain; ++r)
                        col[row + r] = val.get(r) + beta * col[row + r];
                }
            }
        }
    }
}

// Pack MC×KC block of A into MR-contiguous panels
static void pack_a(
    const double* A, int lda,
    int mc, int kc,
    double* buf)
{
    for (int i = 0; i < mc; i += MR) {
        int mr = std::min(MR, mc - i);
        for (int p = 0; p < kc; ++p) {
            const double* col = A + p * lda + i;
            int r = 0;
            for (; r < mr; ++r) buf[r] = col[r];
            for (; r < MR; ++r) buf[r] = 0.0;
            buf += MR;
        }
    }
}

// Pack KC×NC block of B into NR-contiguous panels
static void pack_b(
    const double* B, int ldb,
    int kc, int nc,
    double* buf)
{
    for (int j = 0; j < nc; j += NR) {
        int nr = std::min(NR, nc - j);
        for (int p = 0; p < kc; ++p) {
            int c = 0;
            for (; c < nr; ++c) buf[c] = B[p + (j + c) * ldb];
            for (; c < NR; ++c) buf[c] = 0.0;
            buf += NR;
        }
    }
}

static std::vector<double> g_packA;
static std::vector<double> g_packB;

static void ensure_buffers(int mc, int kc, int nc) {
    // Round up to full micro-tile sizes to avoid OOB reads in micro-kernel
    int padded_mc = ((mc + MR - 1) / MR) * MR;
    int padded_nc = ((nc + NR - 1) / NR) * NR;
    int needA = padded_mc * kc;
    int needB = kc * padded_nc;
    if (static_cast<int>(g_packA.size()) < needA) g_packA.resize(needA);
    if (static_cast<int>(g_packB.size()) < needB) g_packB.resize(needB);
}

extern "C" {

// C = alpha * A * B + beta * C
// A: m×k, B: k×n, C: m×n, all column-major
void numbl_dgemm_f64(
    const double* A, int m, int k,
    const double* B, int n,
    double* C)
{
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    if (m == 0 || n == 0 || k == 0) {
        if (beta == 0.0) std::memset(C, 0, m * n * sizeof(double));
        return;
    }

    ensure_buffers(MC, KC, NC);

    for (int jc = 0; jc < n; jc += NC) {
        int nc = std::min(NC, n - jc);
        for (int pc = 0; pc < k; pc += KC) {
            int kc = std::min(KC, k - pc);

            pack_b(B + pc + jc * k, k, kc, nc, g_packB.data());

            for (int ic = 0; ic < m; ic += MC) {
                int mc = std::min(MC, m - ic);

                pack_a(A + ic + pc * m, m, mc, kc, g_packA.data());

                double useBeta = (pc == 0) ? beta : 1.0;

                for (int jr = 0; jr < nc; jr += NR) {
                    int nr = std::min(NR, nc - jr);
                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = std::min(MR, mc - ir);
                        micro_8x4(
                            kc,
                            g_packA.data() + (ir / MR) * MR * kc,
                            g_packB.data() + (jr / NR) * NR * kc,
                            C + (ic + ir) + (jc + jr) * m, m,
                            alpha, useBeta,
                            mr, nr);
                    }
                }
            }
        }
    }
}

} // extern "C"
