// SPDX-FileCopyrightText: Copyright (c) 2026 Vinny Romano
// SPDX-License-Identifier: MIT

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <string>
#include <sstream>
#include <fstream>
#include <string_view>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <new>
#include <atomic>
#include <thread>
#include <conio.h>
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <immintrin.h>
#include <intrin.h>

#ifndef LYCHREL_VERSION_MAJOR
#define LYCHREL_VERSION_MAJOR 1
#endif
#ifndef LYCHREL_VERSION_MINOR
#define LYCHREL_VERSION_MINOR 0
#endif
#ifndef LYCHREL_VERSION_PATCH
#define LYCHREL_VERSION_PATCH 0
#endif

#define LYCHREL_STRINGIFY_(x) #x
#define LYCHREL_STRINGIFY(x) LYCHREL_STRINGIFY_(x)
#define LYCHREL_VERSION                                                                                                \
  LYCHREL_STRINGIFY(LYCHREL_VERSION_MAJOR)                                                                             \
  "." LYCHREL_STRINGIFY(LYCHREL_VERSION_MINOR) "." LYCHREL_STRINGIFY(LYCHREL_VERSION_PATCH)

// Byte-reversal permutation index for a 64-lane vector.
alignas(64) static const uint8_t kRevIdx[64] = {63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48,
                                                47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
                                                31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
                                                15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0};

// Generate/propagate summary for one contiguous segment resolved with an assumed carry-in of 0. 'gen' means the
// segment emitted a carry-out on its own; 'prop' means every digit ended as 9, so an incoming carry would pass straight
// through. Carry-out = gen | (prop & carry_in).
struct SegSummary
{
  uint8_t gen;
  uint8_t prop;
};

// NUMA PLACEMENT
// On a single-node machine every function here is a no-op, which is deliberate: the cost of getting this wrong on a
// desktop part should be zero, and the topology query is the thing that decides, not a build flag.
//
// On a multi-node machine (EPYC in NPS2/NPS4, or any multi-socket box) the packed buffer is far larger than any cache,
// so essentially every access is a DRAM access. Whether that DRAM is local or remote is then the dominant term: remote
// latency is roughly 1.5-2x local and cross-node bandwidth is a fraction of aggregate local bandwidth. Two pieces have
// to agree for it to come out local:
//   1. Threads must stay on one node, so the node that faulted a page is the node still reading it. See
//      numa_bind_thread below.
//   2. Pages must be faulted by the thread that will use them. Windows allocates a page on the ideal node of whichever
//      thread first touches it, so whoever writes a page first decides where it lives for the buffer's whole lifetime.
//      See PackedBuffer::alloc.
// Neither is useful without the other: pinning alone still leaves every page on whichever node ran the allocation, and
// parallel first-touch alone lets the scheduler migrate a thread away from the pages it placed.
static bool g_numa_pin = true; // --no-pin disables both pinning and parallel first-touch
static int g_pin_total = 0;    // thread count the placement is computed against; set from --threads

// Number of NUMA nodes, or 1 if the topology is flat or unavailable. Cached because it cannot change while the process
// runs and the query is called from the first-touch path.
static int numa_node_count()
{
  static const int nodes = [] {
    ULONG highest = 0;
    if (!GetNumaHighestNodeNumber(&highest))
    {
      return 1;
    }
    return (int)highest + 1;
  }();
  return nodes;
}

// Maps dispatch id -> NUMA node. A BLOCK mapping, not round-robin: ids own contiguous spans of the buffer, so
// consecutive ids must share a node for a node's threads to own one contiguous region. Round-robin would interleave
// them and make almost every access remote - the exact failure this is meant to remove.
static int numa_node_for_id(int id)
{
  const int nodes = numa_node_count();
  const int total = (g_pin_total > 0) ? g_pin_total : 1;
  const int idx = (id < total) ? id : total - 1;
  return (int)((size_t)idx * (size_t)nodes / (size_t)total);
}

// Confines the calling thread to one node's processors. Affinity is set to the node's whole mask rather than to a
// single processor: the goal is only to stop cross-node migration, and leaving the scheduler free to pick a core within
// the node avoids fighting it over SMT siblings or a core that is busy for unrelated reasons.
static void numa_bind_thread(int id)
{
  if (!g_numa_pin || numa_node_count() < 2)
    return;

  GROUP_AFFINITY ga;
  if (GetNumaNodeProcessorMaskEx((USHORT)numa_node_for_id(id), &ga) && ga.Mask != 0)
  {
    SetThreadGroupAffinity(GetCurrentThread(), &ga, nullptr);
  }
}

// PERSISTENT SPIN-BARRIER THREAD POOL
// One reverse-add iteration at 1M digits costs tens of us (23us in one session, ~4x that in another - see the note on
// between-session drift above RunConfig), so per-iteration thread creation (tens of us) would swamp the
// work. Workers are therefore created once and parked on a spinning generation counter; a dispatch is a single atomic
// increment. Workers back off to yield() when idle so the pool does not burn cores during the (long) small-length phase
// of a run.
//
// The pool is sized on demand, to what --threads actually requests. Sizing it to hardware_concurrency() instead would
// scale badly: on a 128C/256T machine it would park 255 threads on a single shared generation counter regardless of the
// requested count, and every dispatch would then write a line all of those threads are watching. Growing to the
// requested size keeps that broadcast proportional to the work.
class SpinPool
{
public:
  using Fn = void (*)(void *, int);

  static SpinPool &instance()
  {
    static SpinPool pool;
    return pool;
  }

  int max_threads() const { return (int)workers_.size() + 1; }

  // Runs fn(ctx, id) for id in [0, n). The calling thread executes id 0.
  void run(Fn fn, void *ctx, int n)
  {
    // The caller runs as id 0 and owns a span like any worker, so it needs the same binding. Done here rather than at
    // startup because the main thread also does I/O and save work, and this is the point where it becomes a compute
    // thread with a span.
    bind_self_once();

    if (n <= 1)
    {
      fn(ctx, 0);
      return;
    }
    ensure_workers(n);
    fn_ = fn;
    ctx_ = ctx;
    active_ = n;

    // n - 1, not workers_.size(): the pool never shrinks, so a dispatch with fewer threads than a previous one leaves
    // surplus workers parked. Counting them here would make the barrier wait on threads that have no work - every one
    // of them would have to be scheduled in just to decrement, turning idle threads into dispatch latency. They still
    // observe the generation bump, but they take no part in the handshake.
    remaining_.store(n - 1, std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_release);

    fn(ctx, 0);

    while (remaining_.load(std::memory_order_acquire) != 0)
    {
      _mm_pause();
    }
  }

private:
  // Pause iterations a parked worker burns before dropping to yield(). Long enough to cover a back-to-back dispatch,
  // short enough not to hold a core through the small-length phase of a run.
  static const int kSpinLimit = 4096;

  SpinPool() = default;

  void bind_self_once()
  {
    if (!self_bound_)
    {
      self_bound_ = true;
      numa_bind_thread(0);
    }
  }

  // Grows the pool to n - 1 workers (the caller is id 0), never shrinking. Only ever called from the dispatching
  // thread and only while no dispatch is outstanding, so workers_ is not being read by anyone else here.
  //
  // Deliberately NOT clamped to hardware_concurrency(): run() dispatches ids [0, n) and every id must have a thread to
  // execute it, so a pool smaller than n would leave those segments unprocessed and silently corrupt the result.
  // --threads is bounded by kMaxThreads, not by the logical processor count, so n can legitimately
  // exceed hw; oversubscription is the caller's choice and merely slow, whereas under-sizing here would be wrong.
  void ensure_workers(int n)
  {
    // Seeded with the CURRENT generation rather than 0: a worker created after dispatches have already happened would
    // otherwise see generation_ != 0 on entry and immediately replay the most recent one. Reading it inside the worker
    // instead would race the other way - it could miss an increment that remaining_ has already counted, and run()
    // would spin forever.
    const uint64_t gen = generation_.load(std::memory_order_acquire);
    for (unsigned i = (unsigned)workers_.size() + 1; i < (unsigned)n; ++i)
    {
      workers_.emplace_back([this, i, gen] { worker_loop((int)i, gen); });
    }
  }

  ~SpinPool()
  {
    stop_.store(true, std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_release);
    for (auto &t : workers_)
    {
      t.join();
    }
  }

  void worker_loop(int id, uint64_t seen)
  {
    // Once, before parking: the binding has to outlive individual dispatches because it is what keeps this thread on
    // the node whose pages it faulted during first-touch.
    numa_bind_thread(id);
    bool idle_last_dispatch = false;
    for (;;)
    {
      // A worker that had no span last time is out of the current thread count and is unlikely to be back in it on the
      // next dispatch either, so it skips the spin phase and yields straight away rather than burning a core waiting
      // for work that is not addressed to it.
      int spins = idle_last_dispatch ? kSpinLimit : 0;
      while (generation_.load(std::memory_order_acquire) == seen)
      {
        if (++spins < kSpinLimit)
        {
          _mm_pause();
        }
        else
        {
          std::this_thread::yield();
        }
      }
      seen = generation_.load(std::memory_order_acquire);
      if (stop_.load(std::memory_order_relaxed))
      {
        return;
      }

      // Surplus workers for this dispatch: no span, and no decrement either - remaining_ was sized to the active set,
      // so touching it here would drive the count negative and hang run(). They just re-park on the next generation.
      idle_last_dispatch = (id >= active_);
      if (!idle_last_dispatch)
      {
        fn_(ctx_, id);
        remaining_.fetch_sub(1, std::memory_order_release);
      }
    }
  }

  std::vector<std::thread> workers_;
  std::atomic<uint64_t> generation_{0};
  std::atomic<int> remaining_{0};
  std::atomic<bool> stop_{false};
  Fn fn_ = nullptr;
  void *ctx_ = nullptr;
  int active_ = 0;
  bool self_bound_ = false;
};

// Upper bound on worker threads used by the parallel core.
static const size_t kMaxThreads = 64;

// ============================================================================
// PACKED BASE-100 CORE - 2 DECIMAL DIGITS PER BYTE
// ============================================================================
//
// Packing two digits into one byte halves the DRAM traffic and measured ~2.5x faster end to end at 8 threads.
// Sustained throughput at 25M digits is 97-107 GB/s across a --block-size sweep, 106 GB/s at the default.
//
// THE REFERENCE MACHINE
// Every measurement in this file that names a machine was taken on the same one, an Azure Standard_D16ds_v5: 16 vCPUs
// on 8 physical Xeon Platinum 8370C cores (Ice Lake-SP, 48 KB L1d, ~0.7 MB LLC per core), 64 GB. It is referred to
// below as "the 8-core Ice Lake-SP" or "the Xeon" interchangeably. Being a shared-tenant VM, the absolute bandwidth
// figures carry a noisy-neighbour component that bare metal would not, so treat GB/s as indicative of the regime the
// loop is in rather than as a hardware specification. The ratios and optima the tuning comments actually rest on are
// unaffected: those were measured within a session, where the machine is repeatable to within ~10%.
//
// REPRESENTATION
// Little-endian: limb[k] holds decimal positions 2k and 2k+1 counted from the units digit, with value = digit(2k) +
// 10*digit(2k+1). So limb[0] % 10 is the units digit. A number of n digits occupies L = (n+1)/2 limbs; when n is odd
// the top limb's high digit is 0.
//
// Little-endian is essential: growth happens at the HIGH end, so limb boundaries never move as the number grows. A
// big-endian layout would need front slack and a shifting offset; none of that exists here.
//
// REVERSAL
// Reversing maps decimal position i -> n-1-i, so the parity of the mapping depends on the parity of n:
//
//   n EVEN: n-1 is odd, so parity FLIPS. Both digits of a reversed limb come
//           from a single source limb, swapped:
//               rev[k] = swap(d[L-1-k])        swap(v) = (v%10)*10 + v/10
//
//   n ODD:  n-1 is even, so parity is PRESERVED and reversed limbs STRADDLE
//           two adjacent source limbs:
//               rev[k] = lo(d[L-1-k]) + 10*hi(d[L-2-k])
//
// Since hi(v)*10 == v - lo(v), only a mod-10 table is needed, never a division. The odd path costs about 1.13x the
// even path.
//
// CARRY
// Limb index ascends with significance, so a single ascending pass is already in LSB -> MSB carry order, with no
// reversal permute needed on the low side. Raw sums reach 9+9 + 9+9 = 198, which still fits in uint8, so the
// generate/propagate trick applies with 10 -> 100 and 9 -> 99.
//
// WRITE-AFTER-READ HAZARD
// The loop meets in the middle, reading one vector from each end and writing both. On the ODD path the high-side output
// needs hi(d[k-1]) where k is on the ASCENDING stream - but d[k-1] was already overwritten by the previous chunk. The
// fix is to keep the previous input vector in a register and synthesize the shifted vector with permutex2var instead of
// re-reading memory. The descending side has no such hazard within a chunk, since its neighbour is written only on the
// following chunk.

// Index vector selecting lanes 63..126 of a two-vector concatenation, i.e. "previous vector shifted right by one lane".
alignas(64) static uint8_t kShift1[64];
// swap(v) = (v%10)*10 + v/10 and lo(v) = v%10 as 128-entry byte tables split into halves for vpermi2b. All limb values
// are < 100 < 128, so the high half of the index space is never selected.
alignas(64) static uint8_t kSwapA[64], kSwapB[64], kLoA[64], kLoB[64];

static void packed_init_tables()
{
  for (int i = 0; i < 64; ++i)
  {
    kShift1[i] = (uint8_t)(63 + i);
  }
  for (int v = 0; v < 128; ++v)
  {
    const uint8_t sw = (uint8_t)((v % 10) * 10 + v / 10);
    const uint8_t lo = (uint8_t)(v % 10);
    if (v < 64)
    {
      kSwapA[v] = sw;
      kLoA[v] = lo;
    }
    else
    {
      kSwapB[v - 64] = sw;
      kLoB[v - 64] = lo;
    }
  }
}

static inline uint8_t swap1(uint8_t v) { return (uint8_t)((v % 10) * 10 + v / 10); }
static inline uint8_t hi10(uint8_t v) { return (uint8_t)(v - v % 10); }

static inline __m512i vswap(__m512i v)
{
  return _mm512_permutex2var_epi8(_mm512_load_si512(kSwapA), v, _mm512_load_si512(kSwapB));
}
static inline __m512i vlo10(__m512i v)
{
  return _mm512_permutex2var_epi8(_mm512_load_si512(kLoA), v, _mm512_load_si512(kLoB));
}

// Normalizes raw sums 0..198 down to 0..99 and threads the carry chain. Byte compares are signed, so the values are
// biased by -128 to order correctly.
static inline __m512i resolve100(__m512i v, uint8_t &carry)
{
  const __m512i bias = _mm512_set1_epi8((char)-128); // 0x80
  const __m512i n99 = _mm512_set1_epi8((char)(99 - 128));
  const __m512i c100 = _mm512_set1_epi8(100);
  const __m512i vb = _mm512_sub_epi8(v, bias);
  const __mmask64 g = _mm512_cmpgt_epi8_mask(vb, n99); // v >= 100 -> generates
  const __mmask64 p = _mm512_cmpeq_epi8_mask(vb, n99); // v == 99  -> propagates
  const uint64_t G = (uint64_t)g, P = (uint64_t)p;
  const uint64_t c = (((G << 1) | carry) + P) ^ P;
  carry = (uint8_t)(((G >> 63) | ((P >> 63) & (c >> 63))) & 1ull);
  const __m512i r = _mm512_mask_sub_epi8(v, g, v, c100);
  const __m512i t = _mm512_add_epi8(r, _mm512_maskz_set1_epi8((__mmask64)c, 1));
  return _mm512_mask_sub_epi8(t, _mm512_cmpgt_epi8_mask(_mm512_sub_epi8(t, bias), n99), t, c100);
}

// Extracts the top lane of a vector, used to seed the next chunk's straddle.
static inline uint8_t top_lane(__m512i v) { return (uint8_t)_mm_extract_epi8(_mm512_extracti32x4_epi32(v, 3), 15); }

// Adds 1 to the limb at 'lo' and propagates upward while limbs are 99.
static inline void packed_apply_carry(uint8_t *p, size_t lo, size_t hi)
{
  for (size_t k = lo; k < hi; ++k)
  {
    if (p[k] == 99)
    {
      p[k] = 0;
    }
    else
    {
      ++p[k];
      return;
    }
  }
}

// Determines the new digit count after an iteration. For odd n the top limb's sum is at most 9 + 9 + 1 = 19, so no
// carry can leave it and growth instead shows up as the top limb reaching 10.
static inline size_t packed_new_len(const uint8_t *p, size_t L, size_t n, bool odd, uint8_t carry_out)
{
  if (carry_out)
  {
    return 2 * L + 1;
  }
  if (odd)
  {
    return (p[L - 1] >= 10) ? n + 1 : n;
  }
  return n;
}

// SERIAL PACKED CORE
// One reverse-add in place over L limbs holding n decimal digits. The buffer must have room for L+1 limbs. Returns the
// new digit count.
static size_t packed_iterate(uint8_t *p, size_t L, size_t n)
{
  const bool odd = (n & 1) != 0;
  const __m512i rev = _mm512_load_si512(kRevIdx);
  const __m512i sh1 = _mm512_load_si512(kShift1);
  const size_t half = L / 2;
  uint8_t carry = 0;

  // Original value of the ascending limb just below the cursor; pass 1 overwrites it, so the odd-parity straddle
  // carries it forward instead.
  uint8_t prev_limb = 0;

  size_t k = 0;
  if (half >= 64)
  {
    __m512i prev_lo = _mm512_setzero_si512();
    for (; k + 64 <= half; k += 64)
    {
      const __m512i a = _mm512_loadu_si512(p + k);
      const __m512i b = _mm512_permutexvar_epi8(rev, _mm512_loadu_si512(p + L - k - 64));

      __m512i rev_lo, rev_hi;
      if (!odd)
      {
        rev_lo = vswap(b);
        rev_hi = vswap(a);
      }
      else
      {
        // Descending neighbour: not yet written, so load it directly.
        const __m512i b1 = _mm512_permutexvar_epi8(rev, _mm512_loadu_si512(p + L - k - 65));
        rev_lo = _mm512_add_epi8(vlo10(b), _mm512_sub_epi8(b1, vlo10(b1)));
        // Ascending neighbour: already clobbered, so rebuild it from the previous input vector rather than from memory.
        const __m512i a1 = _mm512_permutex2var_epi8(prev_lo, sh1, a);
        rev_hi = _mm512_add_epi8(vlo10(a), _mm512_sub_epi8(a1, vlo10(a1)));
      }
      prev_lo = a;
      prev_limb = top_lane(a);

      _mm512_storeu_si512(p + k, resolve100(_mm512_add_epi8(a, rev_lo), carry));
      _mm512_storeu_si512(p + L - k - 64, _mm512_permutexvar_epi8(rev, _mm512_add_epi8(b, rev_hi)));
    }
  }

  for (; k < half; ++k)
  {
    const uint8_t ak = p[k];
    const uint8_t bk = p[L - 1 - k];
    const uint8_t rl = odd ? (uint8_t)(bk % 10 + hi10(p[L - 2 - k])) : swap1(bk);
    const uint8_t rh = odd ? (uint8_t)(ak % 10 + hi10(prev_limb)) : swap1(ak);
    const uint16_t v = (uint16_t)(ak + rl + carry);
    carry = (v >= 100);
    p[k] = (uint8_t)(v - carry * 100);
    p[L - 1 - k] = (uint8_t)(bk + rh); // raw sum, normalized in pass 2
    prev_limb = ak;
  }

  // Middle limb (L odd) is its own mirror.
  if (L & 1)
  {
    const uint8_t am = p[half];
    const uint8_t rm = odd ? (uint8_t)(am % 10 + hi10(prev_limb)) : swap1(am);
    const uint16_t v = (uint16_t)(am + rm + carry);
    carry = (v >= 100);
    p[half] = (uint8_t)(v - carry * 100);
  }

  // Pass 2: normalize the high half, ascending (LSB -> MSB).
  size_t q = (L & 1) ? half + 1 : half;
  for (; q < L && (q & 63) != 0; ++q)
  {
    const uint16_t v = (uint16_t)(p[q] + carry);
    carry = (v >= 100);
    p[q] = (uint8_t)(v - carry * 100);
  }
  for (; q + 64 <= L; q += 64)
  {
    _mm512_storeu_si512(p + q, resolve100(_mm512_loadu_si512(p + q), carry));
  }
  for (; q < L; ++q)
  {
    const uint16_t v = (uint16_t)(p[q] + carry);
    carry = (v >= 100);
    p[q] = (uint8_t)(v - carry * 100);
  }

  if (carry)
  {
    p[L] = 1;
  }
  return packed_new_len(p, L, n, odd, carry);
}

// PACKED PALINDROME TEST
// The number is a palindrome exactly when limb k equals the reversed limb k for every k, which is the same rev[]
// expression the add uses. Checked low-to-high with an early exit; in practice this rejects inside the first chunk, so
// it contributes almost no memory traffic.
static bool packed_is_palindrome(const uint8_t *p, size_t L, size_t n)
{
  const bool odd = (n & 1) != 0;
  const __m512i rev = _mm512_load_si512(kRevIdx);
  const size_t half = L / 2;

  size_t k = 0;
  for (; k + 64 <= half; k += 64)
  {
    const __m512i a = _mm512_loadu_si512(p + k);
    const __m512i b = _mm512_permutexvar_epi8(rev, _mm512_loadu_si512(p + L - k - 64));
    __m512i r;
    if (!odd)
    {
      r = vswap(b);
    }
    else
    {
      const __m512i b1 = _mm512_permutexvar_epi8(rev, _mm512_loadu_si512(p + L - k - 65));
      r = _mm512_add_epi8(vlo10(b), _mm512_sub_epi8(b1, vlo10(b1)));
    }
    if (_mm512_cmpeq_epi8_mask(a, r) != (__mmask64)~0ull)
    {
      return false;
    }
  }
  for (; k < half; ++k)
  {
    const uint8_t bk = p[L - 1 - k];
    const uint8_t r = odd ? (uint8_t)(bk % 10 + hi10(p[L - 2 - k])) : swap1(bk);
    if (p[k] != r)
    {
      return false;
    }
  }
  if (L & 1)
  {
    const uint8_t am = p[half];
    const uint8_t r = odd ? (uint8_t)(am % 10 + hi10(half >= 1 ? p[half - 1] : 0)) : swap1(am);
    if (am != r)
    {
      return false;
    }
  }
  return true;
}

// PARALLEL PACKED CORE
// Same segmented carry-select decomposition as the byte version. Thread t owns pair range [bound[t], bound[t+1]), which
// maps to two disjoint position ranges: the low limbs [a, b) and the mirrored high limbs [L-b, L-a).
//
// Unlike the serial packed_iterate above, this core normalizes the high half within the sweep that produces it rather
// than in a second sweep over the buffer; see the commentary above g_carry_block for how the speculation works.
//
// Segment boundaries introduce a cross-thread version of the straddle hazard: the lowest chunk of a thread reaches one
// limb below its own range on both sides, and that limb belongs to a neighbour that may already have rewritten it. Both
// boundary values are therefore captured serially before dispatch and blended into the first/last vector.
struct PackedCtx
{
  uint8_t *p;
  size_t L;
  bool odd;
  const size_t *bound;
  const uint8_t *seed_lo; // original limb just below each low segment
  const uint8_t *seed_hi; // original limb just below each high segment
  SegSummary *lsb;
  SegSummary *msb;

  uint8_t *cflags = nullptr; // 1 byte per high-half block: bit0 = generates, bit1 = propagates
  size_t nblocks = 0;        // number of full g_carry_block-limb blocks in the high half
  int nthreads = 0;
};

// The high half is normalized in the same sweep that produces it, costing ~2L bytes of traffic per iteration for an
// L-limb buffer.
//
// The straightforward decomposition, still used by the serial packed_iterate above, needs ~3L instead: one sweep reads
// L and writes L, then a second re-reads and re-writes the high half. That third L exists only because the first sweep
// emits the high half in DESCENDING address order (the mirror of the ascending low-half cursor) while a carry chain
// has to resolve ASCENDING, so a limb's carry-in is not known at the moment it is produced.
//
// Both halves want an ascending carry and the mirror couples them, so one of the two must be speculative - that is
// unavoidable. Here the high half is the speculative one. Its limbs are accumulated into an L1-resident stage buffer
// until a whole block is in hand, then resolved assuming carry-in 0 and written out already normalized, recording one
// generate and one propagate bit for the block. A serial bit-parallel scan derives every block's true carry-in, and a
// fixup pass adds 1 only to the blocks that need it - which, because a block only propagates when all its limbs are
// 99, almost always terminates on the first limb it touches.
//
// Staging is what makes this cheaper than the extra sweep it replaces: it puts one speculation boundary per block
// rather than one per 64-byte chunk, so the fixup traffic is negligible instead of a scattered read-modify-write per
// chunk. See g_carry_block below for how the block size is chosen and tuned.
static std::vector<uint8_t> g_cflags;
static std::vector<uint64_t> g_ccin;

// Staging block size, in limbs. Runtime-settable via --block-size because the best value is a property of the machine
// rather than of the algorithm: it wants to be as large as possible while the stage buffer still sits in L1d alongside
// the low-half stream. 8192 (8 KB) suits the 48 KB L1d of the reference machine. Parts with a smaller L1d - Zen 4 has
// 32 KB - may well prefer 4096, which is why this is a knob and not a constant.
//
// The default is safe rather than sharply optimal: anything from 4096 up is close to a wash on this hardware. A sweep
// at 844422 digits with 8 threads, k in units of 1e-11:
//
//      2048  1.279            4096  1.151, 1.188, 1.151            8192  1.171, 1.161, 1.155
//     16384  1.276           32768  1.188, 1.307, 1.181
//
// so 2048 gives up ~10% and 16384/32768 land between with a much wider spread - 32768 varied 10% across identical
// runs, which is itself a reason to avoid it. A sweep at 25M digits is flatter still (2048 1.027, 4096 0.967,
// 8192 0.942, 16384 0.960, 32768 0.938, same units), where only 2048 is clearly behind.
//
// The shape is what matters, not the ranking: undersized blocks cost real throughput, oversized ones cost consistency,
// and the broad middle is flat. Differences of a few percent between neighbouring sizes are not separable from the
// run-to-run spread on this machine.
//
// WHY 8192 RATHER THAN 4096
// Not because it measures faster - it does not. Five interleaved pairs at 844422 digits and four at 25M, k in units
// of 1e-11:
//
//      844422    4096  1.197, 1.100, 1.051, 1.109, 1.057      8192  1.127, 1.029, 1.109, 1.101, 1.158
//         25M    4096  0.850, 0.907, 0.883, 0.901             8192  0.944, 0.869, 0.880, 0.811
//
// 4096 wins three of five pairs at 844422 and two of four at 25M; the means differ by under 1% at both lengths, well
// inside a spread where single runs move 5-10%. On measurement alone the two are indistinguishable and either would
// be a defensible default.
//
// The tie is broken on headroom rather than throughput. Two stage buffers of 8192 limbs occupy 16 KB of the 48 KB L1d,
// leaving room for the low-half stream; at 4096 the buffers are further from the capacity limit but the barrier fires
// twice as often for the same span, and dispatch cost is the one term that grows as threads are added. 8192 is the
// larger value that still fits comfortably, so it keeps the barrier count down without approaching the point where
// 16384 and 32768 start showing their wider spread. On a part with a smaller L1d that argument inverts and 4096 is
// the better starting point, which is what the knob is for.
//
// TUNING ON A NEW MACHINE, AND THE NUMA CAVEAT
// Sweep --block-size over 2048/4096/8192/16384 and keep the winner, but repeat each point several times and treat a
// difference under ~5% as a tie. Do this at a length that is representative of the intended workload, NOT only past
// last-level cache: the knob sizes the per-thread stage buffer against L1d, so it bites while the buffer is still
// LLC-resident.
//
// Block size also governs span variance, which is worth knowing but is not by itself a reason to retune: the spread
// between the fastest and slowest worker span rose monotonically with block size (31% of the mean span at 2048, 51% at
// 8192, 87% at 16384) while throughput stayed flat between 4096 and 8192. Lowering the block size buys a tighter
// barrier and a lower idle percentage without buying wall time, because the stall cycles are inherent to the streaming
// access pattern and are merely redistributed. Tune this on k, not on idle%.
//
// Thread count changes how much a wrong value costs, but not which value is right, so retuning per thread count is not
// worth it; the value only needs revisiting if the per-core L1d changes.
//
// Do not read that thread scaling as shared-cache pressure from the combined per-thread buffers: that model was tested
// directly and is wrong. Holding threads*block constant at 65536 limbs does NOT hold throughput constant, and
// conversely 8192 is at or near the optimum at 2, 4 and 8 threads, i.e. across a 4x range of combined footprint. What
// matters is the absolute per-thread block against one core's private cache. The thread count merely amplifies the
// penalty, which is what you would expect once enough cores are streaming to put the run near the memory bandwidth
// ceiling: the extra traffic from a block that no longer fits costs more when the bus is already saturated. This was
// inferred from timings alone - the counters that would settle it are not readable under a hypervisor with VBS on,
// which is where these numbers were taken - so treat the mechanism as likely, not established.

//
// Raising it far enough also costs threads outright rather than just time - see the fit cap in
// packed_iterate_parallel, which silently reduces the pool once half/block drops below the requested count.
//
// Be aware that on a multi-socket or multi-NUMA-domain machine - notably EPYC in NPS2/NPS4 - this knob is the SECOND
// thing to look at, not the first. Page placement sets a ceiling on throughput that no block size can lift, and bad
// placement also distorts a --block-size sweep by making the run latency-bound rather than L1-bound. Placement is
// handled by parallel first-touch matching the thread partition plus pinning threads to the node holding their span
// (see the NUMA PLACEMENT commentary above); confirm that is working - --no-pin turns it off, which is the easy A/B -
// before reading anything into a block-size sweep.
static const size_t kMinCarryBlockSize = 1024;
static const size_t kMaxCarryBlockSize = 65536;
static size_t g_carry_block = 8192; // limbs staged in L1 before normalizing

static SegSummary packed_seg_low(PackedCtx *c, int id)
{
  uint8_t *const p = c->p;
  const size_t L = c->L;
  const size_t nblocks = c->nblocks;
  const bool odd = c->odd;
  const size_t a = c->bound[id], b = c->bound[id + 1];
  const __m512i rev = _mm512_load_si512(kRevIdx);
  const __m512i sh1 = _mm512_load_si512(kShift1);
  const __m512i n99 = _mm512_set1_epi8(99);

  uint8_t carry = 0;
  bool all99 = true;
  uint8_t prev_limb = c->seed_lo[id];

  // Sized to the compile-time maximum so the buffer stays a stack array; only the leading g_carry_block bytes are used.
  const size_t block = g_carry_block;
  alignas(64) uint8_t stage[kMaxCarryBlockSize];

  size_t k = a;
  __m512i prev_lo = _mm512_maskz_set1_epi8((__mmask64)1ull << 63, (char)prev_limb);

  // Chunks of a block are produced in descending address order, so they are staged into L1 and only normalized once
  // the block's lowest chunk is in hand - at which point the carry can be chained ascending across the whole block,
  // leaving a single speculation boundary per block instead of per chunk.
  for (; k + block <= b; k += block)
  {
    for (size_t j = 0; j < block; j += 64)
    {
      const size_t kk = k + j;
      const __m512i av = _mm512_loadu_si512(p + kk);
      const __m512i bv = _mm512_permutexvar_epi8(rev, _mm512_loadu_si512(p + L - kk - 64));

      __m512i rev_lo, rev_hi;
      if (!odd)
      {
        rev_lo = vswap(bv);
        rev_hi = vswap(av);
      }
      else
      {
        __m512i b1 = _mm512_permutexvar_epi8(rev, _mm512_loadu_si512(p + L - kk - 65));
        if (kk + 64 == b)
        {
          b1 = _mm512_mask_set1_epi8(b1, (__mmask64)1ull << 63, (char)c->seed_hi[id]);
        }
        rev_lo = _mm512_add_epi8(vlo10(bv), _mm512_sub_epi8(b1, vlo10(b1)));
        const __m512i a1 = _mm512_permutex2var_epi8(prev_lo, sh1, av);
        rev_hi = _mm512_add_epi8(vlo10(av), _mm512_sub_epi8(a1, vlo10(a1)));
      }
      prev_lo = av;
      prev_limb = top_lane(av);

      const __m512i done = resolve100(_mm512_add_epi8(av, rev_lo), carry);
      all99 &= (_mm512_cmpeq_epi8_mask(done, n99) == (__mmask64)~0ull);
      _mm512_storeu_si512(p + kk, done);

      // Un-reversed raw sums, parked in memory order at this chunk's offset
      // within the block.
      _mm512_store_si512(stage + (k + (block - 64) - kk),
                         _mm512_permutexvar_epi8(rev, _mm512_add_epi8(bv, rev_hi)));
    }

    uint8_t hcarry = 0;
    bool hprop = true;
    uint8_t *const block_base = p + L - k - block;
    for (size_t o = 0; o < block; o += 64)
    {
      const __m512i done_hi = resolve100(_mm512_load_si512(stage + o), hcarry);
      hprop &= (_mm512_cmpeq_epi8_mask(done_hi, n99) == (__mmask64)~0ull);
      _mm512_storeu_si512(block_base + o, done_hi);
    }
    c->cflags[nblocks - 1 - (k / block)] = (uint8_t)(hcarry | (hprop ? 2u : 0u));
  }

  // Only the last segment can have a remainder, and it lands at the BOTTOM of the high half. Too short to stage as a
  // block, so raw sums are simply left there for the driver to normalize serially, which then feeds the resulting
  // carry into the block scan.
  for (; k + 64 <= b; k += 64)
  {
    const __m512i av = _mm512_loadu_si512(p + k);
    const __m512i bv = _mm512_permutexvar_epi8(rev, _mm512_loadu_si512(p + L - k - 64));

    __m512i rev_lo, rev_hi;
    if (!odd)
    {
      rev_lo = vswap(bv);
      rev_hi = vswap(av);
    }
    else
    {
      __m512i b1 = _mm512_permutexvar_epi8(rev, _mm512_loadu_si512(p + L - k - 65));
      if (k + 64 == b)
      {
        b1 = _mm512_mask_set1_epi8(b1, (__mmask64)1ull << 63, (char)c->seed_hi[id]);
      }
      rev_lo = _mm512_add_epi8(vlo10(bv), _mm512_sub_epi8(b1, vlo10(b1)));
      const __m512i a1 = _mm512_permutex2var_epi8(prev_lo, sh1, av);
      rev_hi = _mm512_add_epi8(vlo10(av), _mm512_sub_epi8(a1, vlo10(a1)));
    }
    prev_lo = av;
    prev_limb = top_lane(av);

    const __m512i done = resolve100(_mm512_add_epi8(av, rev_lo), carry);
    all99 &= (_mm512_cmpeq_epi8_mask(done, n99) == (__mmask64)~0ull);
    _mm512_storeu_si512(p + k, done);
    _mm512_storeu_si512(p + L - k - 64, _mm512_permutexvar_epi8(rev, _mm512_add_epi8(bv, rev_hi)));
  }

  for (; k < b; ++k)
  {
    const uint8_t ak = p[k];
    const uint8_t bk = p[L - 1 - k];
    const uint8_t below = (k + 1 == b) ? c->seed_hi[id] : p[L - 2 - k];
    const uint8_t rl = odd ? (uint8_t)(bk % 10 + hi10(below)) : swap1(bk);
    const uint8_t rh = odd ? (uint8_t)(ak % 10 + hi10(prev_limb)) : swap1(ak);
    const uint16_t v = (uint16_t)(ak + rl + carry);
    carry = (v >= 100);
    const uint8_t digit = (uint8_t)(v - carry * 100);
    p[k] = digit;
    all99 &= (digit == 99);
    p[L - 1 - k] = (uint8_t)(bk + rh);
    prev_limb = ak;
  }

  return {carry, (uint8_t)all99};
}

// Adds 1 to the base limb of every block whose carry-in bit is set. The range is clamped to the block: a block that
// propagates all the way through has already had its carry accounted for by the scan, which set the next block's
// carry-in, so running past the end would apply it twice.
//
// Runs serially on the dispatching thread rather than as a second SpinPool dispatch. It cannot be folded into the
// phase-0 sweep - the carry-in bits do not exist until every thread's blocks have been summarized and scanned - so the
// only way to merge it was to stop paying a barrier for it. That is worth doing because there is almost nothing here to
// parallelize: a block takes a carry only rarely, and when it does packed_apply_carry almost always terminates on the
// first limb it touches, so the whole pass is a scan of one bit per block. A barrier round-trip cost far more than the
// work it distributed, and it was doing so on every iteration.
static void packed_apply_high(uint8_t *p, size_t L, const uint64_t *cin, size_t nblocks)
{
  const size_t block = g_carry_block;
  for (size_t i = 0; i < nblocks; ++i)
  {
    const size_t m = nblocks - 1 - i;
    if ((cin[m >> 6] >> (m & 63)) & 1ull)
    {
      packed_apply_carry(p, L - block * (i + 1), L - block * i);
    }
  }
}

// DIAGNOSTIC INSTRUMENTATION
// Compiled only when LYCHREL_DIAG is defined (nmake diag). Everything below - the counters, the rdtsc stamps in
// packed_worker and packed_iterate_parallel, and span_report - vanishes entirely from a default build, so the shipping
// binary's hot path is unchanged. That is the reason for #ifdef rather than a runtime g_diag flag: the stamps sit
// inside the dispatch loop and around each serial section, where a predictable-but-present branch is still a cost that
// would have to be argued about. Compile-time removal makes the question moot and is verifiable by diffing the
// disassembly of the default build before and after.
#ifdef LYCHREL_DIAG

// Times each thread's span inside a dispatch so intra-dispatch skew can be separated from barrier latency. Each worker
// stamps rdtsc around its own packed_seg_low call; the driver reduces the deltas once run() has returned, which is safe
// because every worker has already decremented remaining_ by then.
static uint64_t g_span_cyc[kMaxThreads];
static uint64_t g_span_tot[kMaxThreads];
static uint64_t g_span_win[kMaxThreads];
static uint64_t g_slow_tot = 0;
static uint64_t g_span_iters = 0;
static int g_span_nthreads = 0;

// Serial-section timers, to test whether the fixed (non-parallel) part of an iteration sets an Amdahl floor. Every
// section below runs on the dispatching thread with the whole pool parked at the barrier, so its cost is paid in full
// by the iteration no matter how many workers there are. g_disp_tot brackets SpinPool::run only, so it is span plus
// barrier entry/exit and nothing else; the span reduction that follows run() is instrumentation and is left outside it.
static uint64_t g_disp_tot = 0;
// Pool cost with the critical-path span removed, and the aggregate thread-time lost waiting at the barrier. Kept apart
// because they have different fixes: the first is the SpinPool implementation, the second is span imbalance.
static uint64_t g_barrier_tot = 0;
static double g_idle_tot = 0.0;
static uint64_t g_ser_tail = 0;
static uint64_t g_ser_scan = 0;
static uint64_t g_ser_carry = 0;
static uint64_t g_ser_high = 0;
static uint64_t g_iter_tot = 0;

// Discriminates OS preemption from memory stalls as the source of span variance. QueryThreadCycleTime counts only
// cycles the thread was actually scheduled on a core, so comparing it against the rdtsc wall delta over the same span
// splits the two cleanly: if wall greatly exceeds executed the thread was descheduled, and if they track each other the
// thread was resident but stalled. g_exc_* accumulate the executed side; g_lost_* accumulate wall-minus-executed, which
// is involuntary time away from the core.
//
// The subtraction is only valid if the two windows nest, with the rdtsc pair outside the QueryThreadCycleTime pair -
// see packed_worker. It is also only valid if both are in the same unit, which is not guaranteed: rdtsc on an
// invariant-TSC part ticks at a fixed reference frequency while QueryThreadCycleTime follows the real core clock, so a
// part running well above its reference rate would report executed > wall. g_core_to_ref rescales for that. It
// measures ~1.00 on this machine, so the correction is inert here and is carried for parts where it is not; the figure
// is printed so a scaled result can be told from an unscaled one. It is calibrated lazily at the first dispatch, since
// calibrating at static-init time would capture an idle clock that does not apply to the run.
static uint64_t g_exec_cyc[kMaxThreads];
static uint64_t g_exec_tot[kMaxThreads];
static uint64_t g_lost_tot[kMaxThreads];
static uint64_t g_lost_max = 0;
static uint64_t g_preempt_iters = 0;
static double g_core_to_ref = 0.0;

// Spins for a short fixed reference interval and returns reference cycles per core cycle. The spin has to be a real
// busy loop: any sleep or wait would let the thread off the core and measure the wrong ratio entirely.
static double diag_calibrate_core_to_ref()
{
  const HANDLE th = GetCurrentThread();
  uint64_t c0 = 0, c1 = 0;
  QueryThreadCycleTime(th, &c0);
  const uint64_t r0 = __rdtsc();
  volatile uint64_t sink = 0;
  uint64_t r1 = r0;
  // ~20M reference cycles, a few ms; long enough to swamp the read overhead, short enough not to stall the first
  // dispatch noticeably.
  while ((r1 = __rdtsc()) - r0 < 20000000ull)
  {
    sink = sink + 1;
  }
  (void)sink;
  QueryThreadCycleTime(th, &c1);
  const uint64_t core = c1 - c0;
  // A ratio outside this range means the calibration was itself preempted or the counters are not usable; fall back to
  // 1.0, which leaves the two counters compared raw rather than inventing a scale factor.
  const double r = (core > 0) ? (double)(r1 - r0) / (double)core : 1.0;
  return (r > 0.25 && r < 4.0) ? r : 1.0;
}

// Span spread per dispatch (max-min), and a histogram of how far the slowest span sits above the mean. Preemption is
// rare and huge; cache-miss jitter is frequent and small, so the shape of this separates them independently of the
// cycle-time evidence above.
static uint64_t g_spread_tot = 0;
static uint64_t g_hist[6];

static void span_report()
{
  if (g_span_iters == 0)
  {
    return;
  }
  const double it = (double)g_span_iters;
  double sum = 0.0;
  for (int t = 0; t < g_span_nthreads; ++t)
  {
    sum += (double)g_span_tot[t];
  }
  const double slow = (double)g_slow_tot / it;
  fprintf(stderr, "\n[span] iters=%llu threads=%d slowest=%.0f cyc mean=%.0f cyc idle=%.1f%%\n",
          (unsigned long long)g_span_iters, g_span_nthreads, slow, sum / it / g_span_nthreads,
          100.0 * (1.0 - (sum / it) / (slow * g_span_nthreads)));
  for (int t = 0; t < g_span_nthreads; ++t)
  {
    const double mean_t = (double)g_span_tot[t] / it;
    fprintf(stderr, "[span]   id %d %9.0f cyc %+6.1f%% vs slowest, slowest on %5.1f%% of iters\n", t, mean_t,
            100.0 * (mean_t / slow - 1.0), 100.0 * (double)g_span_win[t] / it);
  }

  const double iter = (double)g_iter_tot / it;
  const double disp = (double)g_disp_tot / it;
  const double tail = (double)g_ser_tail / it;
  const double scan = (double)g_ser_scan / it;
  const double carry = (double)g_ser_carry / it;
  const double high = (double)g_ser_high / it;
  const double ser = tail + scan + carry + high;
  fprintf(stderr, "[ser] iter=%.0f cyc dispatch=%.0f (%.1f%%) serial=%.0f (%.1f%%)\n", iter, disp,
          100.0 * disp / iter, ser, 100.0 * ser / iter);
  // Dispatch broken into what it is actually made of. The critical path is the slowest span - the work the iteration
  // cannot proceed without - so only the remainder is attributable to the pool itself.
  fprintf(stderr, "[ser]   dispatch = critical-path span %.0f (%.1f%%) + pool overhead %.0f (%.1f%%)\n", slow,
          100.0 * slow / iter, (double)g_barrier_tot / it, 100.0 * ((double)g_barrier_tot / it) / iter);
  fprintf(stderr, "[ser]   thread-time idle at barrier = %.0f cyc/dispatch (%.1f%% of %d-thread capacity)\n",
          g_idle_tot / it, 100.0 * (g_idle_tot / it) / (slow * g_span_nthreads), g_span_nthreads);
  fprintf(stderr,
          "[ser]   tail=%.0f (%.1f%%) high_scan=%.0f (%.1f%%) apply_carry=%.0f (%.1f%%) apply_high=%.0f (%.1f%%)\n",
          tail, 100.0 * tail / iter, scan, 100.0 * scan / iter, carry, 100.0 * carry / iter, high,
          100.0 * high / iter);
  // Amdahl ceiling implied by the fixed part: even with an infinitely fast parallel section the iteration cannot drop
  // below the serial cost, so this is the best speedup any further threading work could buy.
  fprintf(stderr, "[ser]   serial-bound max speedup = %.2fx\n", ser > 0.0 ? iter / ser : 0.0);

  double exec_sum = 0.0, lost_sum = 0.0;
  for (int t = 0; t < g_span_nthreads; ++t)
  {
    exec_sum += (double)g_exec_tot[t];
    lost_sum += (double)g_lost_tot[t];
  }
  const double span_mean = sum / it / g_span_nthreads;
  const double exec_mean = exec_sum / it / g_span_nthreads;
  fprintf(stderr, "[var] span=%.0f cyc executed=%.0f cyc off-core=%.0f cyc (%.2f%% of span)\n", span_mean, exec_mean,
          span_mean - exec_mean, 100.0 * (span_mean - exec_mean) / span_mean);
  fprintf(stderr, "[var]   core:reference clock ratio = %.3f (executed scaled by this; 1.000 means uncalibrated)\n",
          g_core_to_ref);
  fprintf(stderr, "[var]   dispatches with a preempted span: %.3f%%  worst single loss: %llu cyc\n",
          100.0 * (double)g_preempt_iters / it, (unsigned long long)g_lost_max);
  fprintf(stderr, "[var]   mean spread (max-min) = %.0f cyc (%.1f%% of mean span)\n", (double)g_spread_tot / it,
          100.0 * ((double)g_spread_tot / it) / span_mean);
  static const char *lbl[6] = {"<2%", "2-5%", "5-10%", "10-25%", "25-50%", ">50%"};
  fprintf(stderr, "[var]   slowest-span excess over mean:");
  for (int b = 0; b < 6; ++b)
  {
    fprintf(stderr, "  %s:%.1f%%", lbl[b], 100.0 * (double)g_hist[b] / it);
  }
  fprintf(stderr, "\n");
  fflush(stderr);
}

#endif // LYCHREL_DIAG

// Stamps a serial section's cycle cost into an accumulator. Expands to nothing in a default build, so the sections
// below cost exactly what they would with no instrumentation present.
#ifdef LYCHREL_DIAG
#define DIAG_T0(var) const uint64_t var = __rdtsc()
#define DIAG_ADD(acc, var) (acc) += __rdtsc() - (var)
#else
#define DIAG_T0(var) ((void)0)
#define DIAG_ADD(acc, var) ((void)0)
#endif

static void packed_worker(void *v, int id)
{
  PackedCtx *c = static_cast<PackedCtx *>(v);
#ifdef LYCHREL_DIAG
  // The two windows must nest, with the rdtsc pair OUTSIDE the QueryThreadCycleTime pair. QTCT is not cheap - a read
  // through the KUSER shared page, a few hundred cycles - so interleaving them puts that cost inside the executed
  // window but outside the wall window, making executed systematically larger than span and driving the off-core
  // residual negative by a fixed offset. Nested this way the measurement overhead lands on the wall side, which biases
  // the residual towards a small positive and leaves the clamp in the reduction guarding only genuine read skew.
  uint64_t e0 = 0, e1 = 0;
  const HANDLE th = GetCurrentThread();
  const uint64_t t0 = __rdtsc();
  QueryThreadCycleTime(th, &e0);
  c->lsb[id] = packed_seg_low(c, id);
  QueryThreadCycleTime(th, &e1);
  g_span_cyc[id] = __rdtsc() - t0;
  g_exec_cyc[id] = e1 - e0;
#else
  c->lsb[id] = packed_seg_low(c, id);
#endif
}

// Derives each block's carry-in from the per-block generate/propagate bytes,
// 64 blocks at a time with the same carry-lookahead identity resolve100 uses.
static uint8_t packed_high_scan(const uint8_t *flags, uint64_t *cin, size_t nblocks, uint8_t carry)
{
  const __m512i one = _mm512_set1_epi8(1);
  const __m512i two = _mm512_set1_epi8(2);
  for (size_t base = 0; base < nblocks; base += 64)
  {
    const size_t n = (nblocks - base < 64) ? (nblocks - base) : 64;
    const __mmask64 keep = (n == 64) ? (__mmask64)~0ull : (__mmask64)((1ull << n) - 1);
    const __m512i f = _mm512_maskz_loadu_epi8(keep, flags + base);
    const uint64_t G = (uint64_t)_mm512_test_epi8_mask(f, one);
    const uint64_t P = (uint64_t)_mm512_test_epi8_mask(f, two);
    const uint64_t c = (((G << 1) | carry) + P) ^ P;
    cin[base >> 6] = c;
    carry = (uint8_t)(((G >> (n - 1)) | ((P >> (n - 1)) & (c >> (n - 1)))) & 1ull);
  }
  return carry;
}

static size_t packed_iterate_parallel(uint8_t *p, size_t L, size_t n, int nthreads)
{
  const size_t half = L / 2;

  // The number of threads this LENGTH can support, as opposed to the number asked for. Spans are whole blocks (the
  // staging loop only handles full blocks, and the per-block carry flags require every span boundary to be
  // block-aligned), so at most this many threads can be given a non-empty span.
  //
  // Reducing to fit rather than bailing out avoids a cliff: one thread too many would otherwise send the whole call to
  // the serial path, losing every thread instead of the few that did not fit. The requested count is therefore an
  // upper bound honoured where the length allows, not a promise.
  const size_t block = g_carry_block;
  const size_t fit = half / block;
  if (fit > 0 && (size_t)nthreads > fit)
  {
    // Warned because the reduction is otherwise invisible and silently invalidates measurements: a --block-size sweep
    // that crosses this boundary starts comparing thread counts rather than block sizes, and the resulting cliff reads
    // as a block-size effect. Once only, and only when the cap actually binds - the smallest fit is reported, so a
    // growing run warns on its worst case rather than on every step back up.
    //
    // Diagnostic build only, and unconditional within it: the reduction is worth reporting whenever someone is
    // measuring, and this is the measuring binary. Gating on the build rather than on the stop flags also keeps the
    // branch out of the default build's hot path entirely.
    //
    // The text leads with the clearing length rather than with advice, because most readers are growing runs for which
    // the correct action is none - the condition is a band they pass through, not a setting they got wrong. Stating
    // when the threads come back lets that reader dismiss it without reasoning about block arithmetic; the
    // fixed-length case, where --block-size is genuinely the lever, is the qualified branch rather than the headline.
#ifdef LYCHREL_DIAG
    static int warned_fit = -1;
    if (warned_fit == -1 || (int)fit < warned_fit)
    {
      warned_fit = (int)fit;
      fprintf(stderr,
              "WARNING: --threads %d reduced to %zu until %zu digits: at %zu digits a block of %zu limbs leaves only "
              "%zu whole blocks per half, and a thread cannot be given less than one block. A growing run needs no "
              "action and regains all %d threads at that length; lower --block-size only to widen the split at a "
              "fixed length below it.\n",
              nthreads, fit, (size_t)nthreads * block * 2, L, block, fit, nthreads);
      fflush(stderr);
    }
#endif
    nthreads = (int)fit;
  }

  // Only used to detect the degenerate "no thread gets even one block" case; the split itself is by block count below.
  const size_t per = ((half / (size_t)nthreads) / block) * block;
  if (nthreads < 2 || (size_t)nthreads > kMaxThreads || per == 0)
  {
    return packed_iterate(p, L, n);
  }

  const bool odd = (n & 1) != 0;

  // Spans are whole blocks, so the block count rarely divides evenly by the thread count. Handing every leftover block
  // to one thread is what a uniform t*per split does, and the imbalance it creates is severe rather than marginal: the
  // last span is (half - (nthreads-1)*per) limbs, and just past the parallel threshold that is over half the work while
  // the other threads take one block each. run() cannot return until the slowest span finishes, so the whole iteration
  // is paced by that thread and most of the pool idles.
  //
  // Dealing the remainder blocks out one apiece instead bounds the spread across threads to a single block. Only the
  // sub-block tail is left over, and it is picked up by the aligned_end scan below rather than by any worker.
  const size_t nblk = half / block;
  const size_t base_blocks = nblk / (size_t)nthreads;
  const size_t extra_blocks = nblk % (size_t)nthreads;

  size_t bound[kMaxThreads + 1];
  bound[0] = 0;
  for (int t = 0; t < nthreads; ++t)
  {
    bound[t + 1] = bound[t] + (base_blocks + ((size_t)t < extra_blocks ? 1 : 0)) * block;
  }
  // Absorbs the final partial block. Every other bound stays block-aligned, which is what the per-block carry flags and
  // the aligned_end computation below both rely on.
  bound[nthreads] = half;

  uint8_t seed_lo[kMaxThreads], seed_hi[kMaxThreads];
  for (int t = 0; t < nthreads; ++t)
  {
    const size_t a = bound[t], b = bound[t + 1];
    seed_lo[t] = (a == 0) ? 0 : p[a - 1];
    seed_hi[t] = (L - b >= 1) ? p[L - b - 1] : 0;
  }
  const uint8_t mid_prev = (half >= 1) ? p[half - 1] : 0;

  const size_t last = bound[nthreads - 1];
  const size_t aligned_end = last + ((half - last) / block) * block;
  const size_t nblocks = aligned_end / block;
  if (g_cflags.size() < nblocks + 64)
  {
    g_cflags.assign(nblocks + 64, 0);
    g_ccin.assign((nblocks + 64) / 64 + 2, 0);
  }

  SegSummary lsb[kMaxThreads], msb[kMaxThreads];
  PackedCtx ctx{p, L, odd, bound, seed_lo, seed_hi, lsb, msb};
  ctx.cflags = g_cflags.data();
  ctx.nblocks = nblocks;
  ctx.nthreads = nthreads;
  #ifdef LYCHREL_DIAG
    if (g_core_to_ref == 0.0)
    {
      g_core_to_ref = diag_calibrate_core_to_ref();
    }
    const uint64_t iter_t0 = __rdtsc();
    SpinPool::instance().run(&packed_worker, &ctx, nthreads);
    const uint64_t disp_this = __rdtsc() - iter_t0;
    g_disp_tot += disp_this;

    {
      uint64_t slowest = 0;
      uint64_t fastest = ~0ull;
      int slow_id = 0;
      double span_sum = 0.0;
      bool preempted = false;
      for (int t = 0; t < nthreads; ++t)
      {
        g_span_tot[t] += g_span_cyc[t];
        span_sum += (double)g_span_cyc[t];
        // Converted to reference cycles so it is comparable with the rdtsc span above and with the [span] figures.
        const uint64_t exec_ref = (uint64_t)((double)g_exec_cyc[t] * g_core_to_ref);
        g_exec_tot[t] += exec_ref;
        // Clamped because the two clocks are read at slightly different instants and are not the same counter, so a
        // tiny negative is expected noise rather than signal. It should never fire for a structural reason; if this
        // report shows a systematic negative, suspect the window nesting in packed_worker before believing the data.
        const uint64_t lost = (g_span_cyc[t] > exec_ref) ? g_span_cyc[t] - exec_ref : 0;
        g_lost_tot[t] += lost;
        if (lost > g_lost_max)
        {
          g_lost_max = lost;
        }
        // A span that lost more than a quarter of its wall time to being off-core is not explicable by stalls.
        if (lost * 4 > g_span_cyc[t])
        {
          preempted = true;
        }
        if (g_span_cyc[t] > slowest)
        {
          slowest = g_span_cyc[t];
          slow_id = t;
        }
        if (g_span_cyc[t] < fastest)
        {
          fastest = g_span_cyc[t];
        }
      }
      g_slow_tot += slowest;
      g_span_win[slow_id]++;
      g_spread_tot += slowest - fastest;
      // The dispatch window ENCLOSES the spans, so its raw value is span time plus pool cost and says nothing on its
      // own about barrier efficiency - reading it as barrier overhead is what made a 93% figure look alarming when it
      // was mostly just the work. Subtracting the critical path leaves what the pool actually costs: wake, hand out
      // the work item, and rendezvous. Separately, the gap between the slowest span and the mean is time the other
      // threads spent parked at the barrier - real waste, but attributable to span imbalance rather than to the pool.
      g_barrier_tot += (disp_this > slowest) ? disp_this - slowest : 0;
      g_idle_tot += (double)slowest * nthreads - span_sum;
      g_preempt_iters += preempted ? 1 : 0;
      const double mean_span = span_sum / (double)nthreads;
      const double excess = mean_span > 0.0 ? (double)slowest / mean_span - 1.0 : 0.0;
      const int b = excess < 0.02   ? 0
                    : excess < 0.05 ? 1
                    : excess < 0.10 ? 2
                    : excess < 0.25 ? 3
                    : excess < 0.50 ? 4
                                    : 5;
      g_hist[b]++;
      g_span_nthreads = nthreads;
      if (++g_span_iters % 100000 == 0)
      {
        span_report();
      }
    }
  #else
    SpinPool::instance().run(&packed_worker, &ctx, nthreads);
  #endif

  SegSummary mid{0, 1};
  if (L & 1)
  {
    const uint8_t am = p[half];
    const uint8_t rm = odd ? (uint8_t)(am % 10 + hi10(mid_prev)) : swap1(am);
    const uint16_t v = (uint16_t)(am + rm);
    const uint8_t g = (v >= 100);
    const uint8_t digit = (uint8_t)(v - g * 100);
    p[half] = digit;
    mid = {g, (uint8_t)(digit == 99)};
  }

  uint8_t cin_lo[kMaxThreads];
  uint8_t c = 0;
  for (int t = 0; t < nthreads; ++t)
  {
    cin_lo[t] = c;
    c = lsb[t].gen | (lsb[t].prop & c);
  }
  const uint8_t cin_mid = c;
  c = mid.gen | (mid.prop & c);

  // Remainder of the high half, below the lowest full block. Its length is half mod g_carry_block, so it is anywhere
  // from nothing to one block short of 8192 limbs - and every one of those limbs is normalized here, on the dispatching
  // thread, while the whole pool sits at the barrier. Vectorized for that reason rather than for its own cost: the same
  // ascending carry chain resolve100 already threads in the block normalize above, 64 limbs at a time.
  const size_t tail_lo = L - half, tail_hi = L - aligned_end;
  DIAG_T0(ser_t0);
  size_t q = tail_lo;
  for (; q + 64 <= tail_hi; q += 64)
  {
    _mm512_storeu_si512(p + q, resolve100(_mm512_loadu_si512(p + q), c));
  }
  for (; q < tail_hi; ++q)
  {
    const uint16_t v = (uint16_t)(p[q] + c);
    c = (uint8_t)(v >= 100);
    p[q] = (uint8_t)(v - c * 100);
  }
  DIAG_ADD(g_ser_tail, ser_t0);

  DIAG_T0(scan_t0);
  const uint8_t carry_out = packed_high_scan(g_cflags.data(), g_ccin.data(), nblocks, c);
  DIAG_ADD(g_ser_scan, scan_t0);

  DIAG_T0(carry_t0);
  for (int t = 0; t < nthreads; ++t)
  {
    if (cin_lo[t])
    {
      packed_apply_carry(p, bound[t], bound[t + 1]);
    }
  }
  if (cin_mid && (L & 1))
  {
    p[half] = (p[half] == 99) ? 0 : (uint8_t)(p[half] + 1);
  }
  DIAG_ADD(g_ser_carry, carry_t0);

  DIAG_T0(high_t0);
  packed_apply_high(p, L, g_ccin.data(), nblocks);
  DIAG_ADD(g_ser_high, high_t0);
#ifdef LYCHREL_DIAG
  g_iter_tot += __rdtsc() - iter_t0;
#endif

  if (carry_out)
  {
    p[L] = 1;
  }
  return packed_new_len(p, L, n, odd, carry_out);
}

// PACKED BUFFER WITH PARALLEL FIRST-TOUCH
// Deliberately not a std::vector. The reason is narrow but decisive: vector's constructor value-initializes, so the
// allocating thread writes every byte and therefore first-touches every page. That single act pins the whole buffer to
// one NUMA node before any worker has run, and no amount of later pinning can undo it - placement is decided by the
// first write and is permanent for the lifetime of the mapping. So the memory is obtained raw and left untouched, and
// the zeroing is done by the workers, in the same partition they will later compute over.
//
// PARTITION AND ITS DRIFT
// The compute split is mirrored: thread t owns the low span [bound[t], bound[t+1]) and its reflection near the top of
// the number, because a reverse-add reads a limb and its mirror together. The touch below reproduces that shape rather
// than a flat contiguous split, so both halves of a thread's working set land on its own node.
//
// The mirror position depends on L, which grows during a run, so the placement is exact when made and drifts as the
// number lengthens. It is re-established on every doubling, since a grown buffer is a fresh mapping that gets touched
// the same way, which bounds the drift to one doubling's worth of growth. Perfect tracking would require migrating
// pages mid-run and is not worth it: the low span is contiguous and dominant, and it is the part that stays put.
struct TouchCtx
{
  uint8_t *p;
  size_t bytes; // total buffer size, including the spare carry-out limb
  size_t L;     // limbs in use at the time of the touch
  const size_t *bound;
  int nthreads;
};

static void touch_worker(void *v, int id)
{
  const TouchCtx *c = static_cast<const TouchCtx *>(v);
  numa_bind_thread(id);

  const size_t a = c->bound[id], b = c->bound[id + 1];
  std::memset(c->p + a, 0, b - a);

  // The mirrored split covers [0, half) and [L-half, bytes), which leaves the middle limb uncovered when L is odd.
  // id 0 takes it; it is a single limb, so where it lands does not matter, but it must not be skipped - the buffer
  // above the packed digits has to be zero for the carry-out write and for growth headroom.
  if (id == 0 && c->L / 2 < c->L - c->L / 2)
  {
    std::memset(c->p + c->L / 2, 0, (c->L - c->L / 2) - c->L / 2);
  }

  // The mirrored high span. The last id also takes everything above the number - the growth headroom, which it will be
  // the one to use as the top of the buffer fills.
  const size_t hi_end = (id == c->nthreads - 1) ? c->bytes : c->L - a;
  const size_t hi_beg = c->L - b;
  if (hi_beg < hi_end)
  {
    std::memset(c->p + hi_beg, 0, hi_end - hi_beg);
  }
}

class PackedBuffer
{
public:
  PackedBuffer() = default;

  // Allocates 'bytes' and zeroes it from the worker threads so each page is faulted by the thread that will use it.
  // Throws std::bad_alloc on failure, so the caller's out-of-memory handling applies unchanged.
  PackedBuffer(size_t bytes, size_t L, int nthreads) : p_((uint8_t *)_aligned_malloc(bytes, 64)), bytes_(bytes)
  {
    if (!p_)
    {
      throw std::bad_alloc();
    }

    // Serial fallback: with one thread, or a length too short to split, or pinning disabled, there is nothing to
    // distribute and a single memset is the fastest way to do it.
    const size_t half = L / 2;
    const size_t per = (nthreads > 1) ? half / (size_t)nthreads : 0;
    if (!g_numa_pin || numa_node_count() < 2 || nthreads < 2 || (size_t)nthreads > kMaxThreads || per == 0)
    {
      std::memset(p_, 0, bytes);
      return;
    }

    // Mirrors the block-dealt split in packed_iterate_parallel so a thread faults the pages it will later compute over.
    // Exactness is not required here - placement is a heuristic and drifts as L grows anyway - but using the same shape
    // keeps first touch and compute on the same node for the whole span rather than only its leading blocks.
    const size_t block = g_carry_block;
    const size_t nblk = half / block;
    size_t bound[kMaxThreads + 1];
    bound[0] = 0;
    if (nblk >= (size_t)nthreads)
    {
      const size_t base_blocks = nblk / (size_t)nthreads;
      const size_t extra_blocks = nblk % (size_t)nthreads;
      for (int t = 0; t < nthreads; ++t)
      {
        bound[t + 1] = bound[t] + (base_blocks + ((size_t)t < extra_blocks ? 1 : 0)) * block;
      }
    }
    else
    {
      for (int t = 0; t < nthreads; ++t)
      {
        bound[t + 1] = (size_t)(t + 1) * per;
      }
    }
    bound[nthreads] = half;

    TouchCtx ctx{p_, bytes, L, bound, nthreads};
    SpinPool::instance().run(&touch_worker, &ctx, nthreads);
  }

  ~PackedBuffer() { _aligned_free(p_); }

  PackedBuffer(const PackedBuffer &) = delete;
  PackedBuffer &operator=(const PackedBuffer &) = delete;

  void swap(PackedBuffer &o)
  {
    std::swap(p_, o.p_);
    std::swap(bytes_, o.bytes_);
  }

  uint8_t *data() { return p_; }

private:
  uint8_t *p_ = nullptr;
  size_t bytes_ = 0;
};

// Converts raw decimal digits (MSB first) into little-endian base-100 limbs. Only used for the short starting value;
// the save and load paths read and write limbs directly, a digit at a time, and never materialise a digit array.
static void pack_digits(const uint8_t *digits, size_t n, uint8_t *limbs)
{
  const size_t L = (n + 1) / 2;
  std::fill_n(limbs, L, (uint8_t)0);
  for (size_t i = 0; i < n; ++i)
  {
    const uint8_t d = digits[n - 1 - i]; // i counts up from the units digit
    limbs[i / 2] = (uint8_t)(limbs[i / 2] + ((i & 1) ? d * 10 : d));
  }
}

static int default_parallel_threads(); // Defined further down with the rest of the thread-count reasoning.

// Crossover between the serial and parallel cores, measured at 8 threads with --parallel-threshold 0 so the parallel
// path is reachable at every length (mean of N, interleaved serial/parallel to cancel machine drift):
//
//      65536  0.83x       73728  0.90x       81920  0.97x       98304  1.12x
//     131072  1.30x      196608  1.81x      262144  2.08x      393216  2.37x
//     524288  2.75x      786432  2.90x
//
// The parallel path first pays for its carry scan between 81920 and 98304, so the threshold sits at 98304. The choice
// only matters inside that band and the gain there is modest; every length past ~200k digits is unaffected by where
// this sits.
//
// The crossover does NOT move with thread count, which is why one number serves all pool sizes: 2, 4 and 8 threads all
// lose at 65536, sit at break-even (0.97-0.99x) at 81920 and win at 98304. Thread count changes the payoff past the
// crossover (at 393216: 1.31x at 2 threads against 2.37x at 8), not where the crossover is. The fixed cost being
// amortized here is the carry scan, and both arms scale with the pool together.
//
// These figures are x64-only and the architecture is a hard prerequisite for reproducing them, not a detail: a 32-bit
// build puts the crossover somewhere else entirely and invents a non-monotonic dip that does not exist here. See the
// guard in the makefile.
//
// Architecture dominates absolute k, so check it before reading anything into a level shift. Measured interleaved in a
// single session at 524288 digits, 1 thread, --parallel-threshold 0:
//
//      x64  3.41e-11, 3.33e-11        x86  1.19e-10, 1.24e-10
//
// a 3.6x ratio, matching the guard in the makefile. Absolute k is stable across sessions for a fixed binary: it is
// repeatable to within ~10% and two independent timing methods agree. Ratios remain the safer thing to quote, but a
// cross-session absolute that has moved by more than that spread is evidence of a changed binary, not of a noisy
// machine - check the PE machine type before blaming the environment.

//
// Runtime-settable via --parallel-threshold, for the same reason --block-size is: the crossover is a property of the
// machine's core count and cache rather than of the algorithm, so tuning it on new hardware should not need a rebuild.
// The default can also be changed at build time with -DDEFAULT_PARALLEL_THRESHOLD=<digits>.
//
// Setting it to 0 forces the parallel core on at every length, which is how the sweep above is reproduced;
// sub-threshold lengths are otherwise unmeasurable because --threads is inert below the cutover.
#ifndef DEFAULT_PARALLEL_THRESHOLD
#define DEFAULT_PARALLEL_THRESHOLD 98304
#endif

// Tunables for reporting, autosave, stopping, and save-file handling. Every interval is independent and any
// combination may be active at once; a value of 0 disables that trigger.
struct RunConfig
{
  uint64_t status_iterations = 100'000;      // iterations between progress lines; 0 disables
  size_t status_digits = 1'000'000;          // digits of growth between progress lines; 0 disables
  double status_minutes = 0.0;               // wall minutes between progress lines; 0 disables
  uint64_t save_iterations = 100'000'000;    // iterations between autosaves; 0 disables
  size_t save_digit_interval = 100'000'000;  // digits of growth between autosaves; 0 disables
  double save_minutes = 1440.0;              // wall minutes between autosaves; 0 disables
  size_t stop_digits = 0;                   // stop once the number reaches this many digits; 0 disables
  double stop_minutes = 0.0;                // stop after this many wall minutes; 0 disables
  uint64_t stop_iteration = 0;              // stop on reaching this absolute iteration; 0 disables
  int threads = default_parallel_threads(); // worker threads for the parallel core; 1 forces serial
  size_t parallel_threshold = DEFAULT_PARALLEL_THRESHOLD; // digits before the parallel core engages; 0 always
  bool no_crc_check = false;              // treat a save-file CRC mismatch as a warning rather than an error
};

static RunConfig g_config;
static std::atomic<bool> g_out_of_memory{false};

// CRC-16/CCITT-FALSE - poly 0x1021, init 0xFFFF, non-reflected in and out, no final XOR. Table-driven so the 50M-digit
// case stays cheap.
static uint16_t crc16_table[256];

static void crc16_init_table()
{
  for (uint32_t i = 0; i < 256; ++i)
  {
    uint16_t c = (uint16_t)(i << 8);
    for (int b = 0; b < 8; ++b)
    {
      c = (c & 0x8000) ? (uint16_t)((c << 1) ^ 0x1021) : (uint16_t)(c << 1);
    }
    crc16_table[i] = c;
  }
}

static inline uint16_t crc16_update(uint16_t crc, const uint8_t *data, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    crc = (uint16_t)((crc << 8) ^ crc16_table[(crc >> 8) ^ data[i]]);
  }
  return crc;
}

// RESULT FILE WRITER
//
// Layout - every header value starts on column 19 except the save number, which starts on column 17:
//
//   Automatic save #12345
//   Initial value:    196
//   Iteration:        2415836
//   Number of digits: 1000000
//   <digits, 70 per line>
//
// CRC coverage matches ISFTOOL's file_crc16(): it skips the first two lines (the save number line, which holds the CRC
// itself, and the initial value line) and checksums every remaining byte to EOF, terminators included.
//
// Lines MUST end with a bare LF. ISFTOOL skips those two lines with two loops that each read until a byte < ' ', and it
// opens the file "rb". With CRLF the first loop stops on the \r and leaves the \n, which the second loop then consumes
// as its only byte - so only ONE line gets skipped and every CRC mismatches. With LF the skip works as intended.
//
// ISFTOOL reads the stored value with strtod(strn+16), so it is written in decimal, not hex.
static const size_t kDigitsPerLine = 70;

// Block size for the streaming save and load paths. Large enough that the per-write overhead is negligible, small
// enough to stay resident in L2 no matter how large the number is.
static const size_t kStreamBlockBytes = 64 * 1024;

// Emits the CRC'd region - everything from line 3 on - in bounded blocks, expanding each decimal digit straight out of
// the packed base-100 limbs. Nothing proportional to the number is ever allocated, so a save at 1e10 digits costs the
// same memory as one at 100.
//
// The digit at MSB-first position 'pos' lives in limb (len-1-pos)/2: the low decimal of the limb when that index is
// even, the high decimal when it is odd. Same mapping the packed core uses, just read one digit at a time.
template <typename Sink>
static void stream_save_body(uint64_t iteration, const uint8_t *limbs, size_t len, Sink &&sink)
{
  std::string block;
  block.reserve(kStreamBlockBytes + kDigitsPerLine + 1);

  block += "Iteration:        " + std::to_string(iteration) + "\n";
  block += "Number of digits: " + std::to_string(len) + "\n";

  for (size_t i = 0; i < len; i += kDigitsPerLine)
  {
    const size_t n = std::min(kDigitsPerLine, len - i);
    for (size_t j = 0; j < n; ++j)
    {
      const size_t k = len - 1 - (i + j);
      const uint8_t v = limbs[k / 2];
      block.push_back((char)('0' + ((k & 1) ? v / 10 : v % 10)));
    }
    block.push_back('\n');

    // Flushed on a line boundary, so a block never splits a line and the reserve above bounds the buffer exactly.
    if (block.size() >= kStreamBlockBytes)
    {
      sink(block.data(), block.size());
      block.clear();
    }
  }

  if (!block.empty())
  {
    sink(block.data(), block.size());
  }
}

// 'limbs' is the live packed buffer; the digits are expanded on the fly rather than unpacked into a scratch array
// first.
//
// The body is generated twice: once to obtain the CRC, which has to be written on line 1 before any of it, and once to
// write it. Regenerating is pure arithmetic over a buffer that is already resident, and a save happens only on
// milestones and interrupts, so that is a far better trade than holding a second copy of the number in memory.
//
// Written to a temporary name and renamed into place only after the last byte is down. A save can be cut short - the
// console handler gives up waiting on a bounded budget during shutdown, and the process is killed wherever it happens
// to be - and without this that leaves a truncated .isf sitting under a name that looks like a valid checkpoint. The
// rename is a single directory operation, so an interrupted save now leaves either the previous contents or nothing at
// all, never half a number.
//
// This protects against the process dying, not against the machine dying: the bytes are handed to the OS but not forced
// to the platter, so a power loss can still lose a save that was reported as written. Guarding that would need
// FlushFileBuffers and costs a full sync on every checkpoint, which is not a trade worth making for a file that is
// rewritten on every milestone.
bool save_result(const std::string &path, std::string_view initial, uint64_t iteration, const uint8_t *limbs,
                 size_t len)
{
  const std::string temp_path = path + ".partial";

  {
    std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
    if (!out)
    {
      return false;
    }

    crc16_init_table();
    uint16_t crc = 0xFFFF;
    stream_save_body(iteration, limbs, len,
                     [&](const char *p, size_t n) { crc = crc16_update(crc, (const uint8_t *)p, n); });

    out << "Automatic save #" << crc << "\n"
        << "Initial value:    " << initial << "\n";
    stream_save_body(iteration, limbs, len, [&](const char *p, size_t n) { out.write(p, (std::streamsize)n); });

    out.flush();
    if (!out)
    {
      out.close();
      DeleteFileA(temp_path.c_str());
      return false;
    }
  }

  // MOVEFILE_REPLACE_EXISTING because a checkpoint may legitimately be rewritten - the same iteration and length can be
  // reached twice across a resume.
  if (!MoveFileExA(temp_path.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING))
  {
    DeleteFileA(temp_path.c_str());
    return false;
  }

  return true;
}

// MOD-9 CONSISTENCY CHECK - port of ISFTOOL's verify_checksum().
//
// A reverse-add step maps n -> n + reverse(n). Both terms share the same digit sum, so the residue mod 9 simply doubles
// every iteration. Starting from the initial value's residue, the stored number's residue must therefore be
// val1 * 2^iter (mod 9).
//
// Doubling mod 9 has period 6 in general, but 3 and 6 form the short cycle 3 -> 6 -> 3 (period 2), and 0 is fixed.
// ISFTOOL also treats the two cycles as closed sets: a trajectory starting at 3 or 6 can only ever land on 3 or 6, and
// one starting elsewhere (nonzero) can never reach 0, 3, or 6.
//
// Returns 0 on success, 1 if the residue is wrong for this iteration count, or 2 if the stored number could not arise
// from the stored start value at all.
int verify_checksum(int val1, int val2, uint64_t iter)
{
  int flag = 0;
  int temp;

  if (val1 == 0)
  {
    if (val1 != val2)
    {
      flag = 2;
    }
  }
  else if ((val1 == 3) || (val1 == 6))
  {
    temp = val1;
    for (uint64_t i = 0; i < (iter % 2); ++i)
    {
      temp = ((temp * 2) % 9);
    }
    if (temp != val2)
    {
      flag = 1;
    }
    if ((val2 != 6) && (val2 != 3))
    {
      flag = 2;
    }
  }
  else
  {
    temp = val1;
    for (uint64_t i = 0; i < (iter % 6); ++i)
    {
      temp = ((temp * 2) % 9);
    }
    if (temp != val2)
    {
      flag = 1;
    }
    if ((val2 == 0) || (val2 == 3) || (val2 == 6))
    {
      flag = 2;
    }
  }

  return flag;
}

// RESULT FILE READER
//
// Parses a file written by save_result and recovers the state needed to resume: the initial value, the iteration count,
// and the number itself, delivered already packed into little-endian base-100 limbs. The stored CRC is recomputed
// exactly the way ISFTOOL's file_crc16() does - skip the first two lines, checksum the rest byte for byte - and a
// mismatch is reported so a truncated or corrupted save is never silently resumed from.
//
// The file is read twice in bounded blocks rather than slurped into a body string: pass one checksums, pass two parses
// digits straight into the limb array.
// 
// Header values are read from their fixed columns: the CRC at index 16, the rest at index 18, matching ISFTOOL's
// strtod(strn+16) / strtod(strn+18).
static bool all_digits(std::string_view s);

static bool parse_uint64(std::string_view text, uint64_t &value)
{
  if (text.empty())
  {
    return false;
  }

  uint64_t parsed = 0;
  for (const char c : text)
  {
    if (c < '0' || c > '9' || parsed > (std::numeric_limits<uint64_t>::max() - (uint64_t)(c - '0')) / 10)
    {
      return false;
    }
    parsed = parsed * 10 + (uint64_t)(c - '0');
  }
  value = parsed;
  return true;
}

bool load_result(const std::string &path, std::string &initial, uint64_t &iteration, std::vector<uint8_t> &limbs,
                 size_t &digit_count)
{
  try
  {
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
      std::cout << "ERROR: cannot open " << path << "\n";
      return false;
    }

    std::string line1, line2;
    if (!std::getline(in, line1) || !std::getline(in, line2))
    {
      std::cout << "ERROR: " << path << " is truncated\n";
      return false;
    }
    constexpr std::string_view kSavePrefix = "Automatic save #";
    constexpr std::string_view kInitialPrefix = "Initial value:    ";
    uint64_t stored_value = 0;
    if (!line1.starts_with(kSavePrefix) ||
        !parse_uint64(std::string_view(line1).substr(kSavePrefix.size()), stored_value) ||
        stored_value > std::numeric_limits<uint16_t>::max() || !line2.starts_with(kInitialPrefix) ||
        !all_digits(std::string_view(line2).substr(kInitialPrefix.size())))
    {
      std::cout << "ERROR: " << path << " has a malformed header\n";
      return false;
    }

    // Everything past the first two lines is the CRC'd region. Checksummed in a streaming pass so the body never has
    // to exist in memory as a whole.
    const std::streampos body_start = in.tellg();
    std::vector<char> block(kStreamBlockBytes);

    crc16_init_table();
    uint16_t actual = 0xFFFF;
    for (;;)
    {
      in.read(block.data(), (std::streamsize)block.size());
      const size_t got = (size_t)in.gcount();
      if (got == 0)
      {
        break;
      }
      actual = crc16_update(actual, (const uint8_t *)block.data(), got);
    }
    in.clear();
    in.seekg(body_start);

    const uint16_t stored = (uint16_t)stored_value;
    if (actual != stored)
    {
    // A mismatch normally means the file is damaged and its digits cannot be trusted, so the load fails. --no-crc-check
    // downgrades it to a warning for the case where the save is known-good but its checksum is not - a file repaired by
    // hand, or one written by a tool that computes the CRC over a different region.
      if (g_config.no_crc_check)
      {
        std::cout << "WARNING: CRC mismatch in " << path << " (file says " << stored << ", computed " << actual
                  << ") - continuing because --no-crc-check was given\n";
      }
      else
      {
        std::cout << "ERROR: CRC mismatch in " << path << " (file says " << stored << ", computed " << actual << ")\n";
        return false;
      }
    }

    initial = line2.substr(kInitialPrefix.size());

  // Body line 1 is "Iteration:", line 2 is "Number of digits:", then digits. Read from the reopened body position, so
  // the parse walks the file a second time instead of a buffered copy of it.
    std::string l3, l4;
    constexpr std::string_view kIterationPrefix = "Iteration:        ";
    constexpr std::string_view kDigitsPrefix = "Number of digits: ";
    uint64_t declared_digits = 0;
    if (!std::getline(in, l3) || !std::getline(in, l4) || !l3.starts_with(kIterationPrefix) ||
        !parse_uint64(std::string_view(l3).substr(kIterationPrefix.size()), iteration) ||
        !l4.starts_with(kDigitsPrefix) ||
        !parse_uint64(std::string_view(l4).substr(kDigitsPrefix.size()), declared_digits) || declared_digits == 0 ||
        declared_digits > std::numeric_limits<size_t>::max())
    {
      std::cout << "ERROR: " << path << " has a malformed header\n";
      return false;
    }
    const size_t expect = (size_t)declared_digits;
    if ((expect + 1) / 2 > limbs.max_size())
    {
      std::cout << "ERROR: " << path << " declares more digits than this build can load\n";
      return false;
    }

    // Packed on the fly: the digit at MSB-first position 'pos' belongs in limb (expect-1-pos)/2, low decimal when that
    // index is even and high decimal when it is odd. Every position is written exactly once into a zeroed array, so
    // accumulating with += is safe and no digit array is needed.
    std::vector<uint8_t> packed((expect + 1) / 2, (uint8_t)0);
    size_t pos = 0;
    int csum = 0;
    for (;;)
    {
      in.read(block.data(), (std::streamsize)block.size());
      const size_t got = (size_t)in.gcount();
      if (got == 0)
      {
        break;
      }
      for (size_t i = 0; i < got; ++i)
      {
        const char c = block[i];
        if (c >= '0' && c <= '9')
        {
          if (pos >= expect)
          {
            std::cout << "ERROR: " << path << " declares " << expect << " digits but contains more\n";
            return false;
          }
          const uint8_t d = (uint8_t)(c - '0');
          const size_t k = expect - 1 - pos;
          packed[k / 2] = (uint8_t)(packed[k / 2] + ((k & 1) ? d * 10 : d));
          csum = (csum + d) % 9;
          ++pos;
        }
        else if (c != '\n')
        {
          std::cout << "ERROR: " << path << " contains invalid digit data\n";
          return false;
        }
      }
    }

    if (pos != expect)
    {
    std::cout << "ERROR: " << path << " declares " << expect << " digits but contains " << pos << "\n";
      return false;
    }

  // MOD-9 consistency: the stored number must be reachable from the stored initial value in exactly the stored number
  // of iterations. This catches mismatched header fields that the CRC cannot - a save whose digits and checksum agree
  // but whose iteration count or start value was wrong.
    int corg = 0;
    for (char c : initial)
    {
      corg = (corg + (c - '0')) % 9;
    }
    const int flag = verify_checksum(corg, csum, iteration);
    if (flag != 0)
    {
    std::cout << "ERROR: MOD-9 checksum failed in " << path << " - the stored " << expect
              << "-digit number cannot ";
    if (flag == 1)
    {
      std::cout << "occur on the stored iteration (" << iteration << ")\n";
    }
    else
    {
      std::cout << "result from the stored starting value (" << initial << ")\n";
    }
      return false;
    }

    limbs.swap(packed);
    digit_count = expect;
    return true;
  }
  catch (const std::bad_alloc &)
  {
    g_out_of_memory = true;
    std::cout << "ERROR: not enough memory to load " << path << "\n";
    return false;
  }
}

// RUN CONTROL
//
// Console control handlers run on a separate thread injected by Windows, so they must not touch the digit buffer - a
// save taken from there could catch the number mid-reverse-add. Instead the handler only raises a flag and then blocks
// until the main loop reaches a safe point, performs the save, and signals completion.
//
// That wait is mandatory for shutdown and logoff: returning from the handler lets Windows terminate the process, so the
// save must finish first. The OS bounds how long a handler may take before killing the process anyway, so the wait is
// bounded to stay inside that budget.
//
// The exact budget is deliberately NOT relied on. It is not a single documented constant: it differs between the close
// and the logoff/shutdown paths, is registry-tunable (HungAppTimeout, WaitToKillAppTimeout), and its defaults have
// changed across Windows versions. kShutdownSaveBudget below is therefore chosen to sit comfortably under the smallest
// value any of those has plausibly taken, rather than to match a specific figure. Overrunning is survivable in any
// case - save_result renames into place, so a save cut short leaves no file rather than a corrupt one.
//
// Detection latency comes out of that same 5 seconds, which is why the request flag is polled every iteration rather
// than on the cadence below. See the service block in the main loop.

// CEILING on iterations between control polls, not a fixed cadence. The loop retunes the live interval against measured
// iteration cost so the WALL-CLOCK gap between polls stays near kControlPollTarget at every length.
//
// A fixed count cannot hold a time budget, because iteration cost scales with the number: 4096 iterations is under a
// millisecond at 1000 digits, about 5 seconds at 100M, and half a minute at 400M. Anchoring the cadence to a count
// therefore means responsiveness silently decays to nothing exactly where a run is most expensive to lose.
static const uint32_t kControlPollInterval = 4096;

// Wall-clock target between control polls. Governs keyboard, pause and the timed save/status cadences; the save request
// itself does not wait for this, so the target is set for comfortable interactive response rather than for the
// shutdown deadline.
static const std::chrono::steady_clock::duration kControlPollTarget = std::chrono::milliseconds(250);

// How long the handler will wait for the main loop to finish a save.
//
// Ctrl+C and Ctrl+Break do NOT start an OS kill timer - the process lives until the handler returns - so they get room
// to finish. Capping those at the shutdown budget would abandon a large save for no reason: at 400M digits the write
// alone is hundreds of megabytes and will not fit in a shutdown window however early it starts. The cap is there only
// so a wedged main loop cannot make the process unkillable from the console.
//
// CLOSE, LOGOFF and SHUTDOWN do start one, and overrunning it means being killed mid-write, so they stay short.
static const std::chrono::steady_clock::duration kInterruptSaveBudget = std::chrono::seconds(60);
static const std::chrono::steady_clock::duration kShutdownSaveBudget = std::chrono::milliseconds(4500);

static std::atomic<bool> g_save_requested{false};
static std::atomic<bool> g_save_complete{false};
static std::atomic<bool> g_quit_requested{false};
static std::atomic<bool> g_paused{false};

static BOOL WINAPI console_ctrl_handler(DWORD type)
{
  std::chrono::steady_clock::duration budget;
  switch (type)
  {
  case CTRL_C_EVENT:
  case CTRL_BREAK_EVENT:
    budget = kInterruptSaveBudget;
    break;
  case CTRL_CLOSE_EVENT:
  case CTRL_LOGOFF_EVENT:
  case CTRL_SHUTDOWN_EVENT:
    budget = kShutdownSaveBudget;
    break;
  default:
    return FALSE;
  }

  // A pause would otherwise stall the loop and stop it from servicing this.
  g_paused = false;
  g_quit_requested = true;
  g_save_complete = false;
  g_save_requested = true;

  const auto deadline = std::chrono::steady_clock::now() + budget;
  while (!g_save_complete.load(std::memory_order_acquire))
  {
    if (std::chrono::steady_clock::now() > deadline)
    {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  return TRUE;
}

// Polls the console for a spacebar press without blocking. Returns true if the user toggled pause.
static bool space_pressed()
{
  if (!_kbhit())
  {
    return false;
  }
  const int c = _getch();
  return (c == ' ');
}

// Local wall-clock time, "2024-06-01 14:23:45". Fixed width, so the column stays aligned, and sortable as text.
static const char *const kTimestampHeader = "               Time";

static std::string timestamp_now()
{
  const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

  std::tm local{};
  localtime_s(&local, &t);

  char buf[32];
  std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d", local.tm_year + 1900, local.tm_mon + 1,
                local.tm_mday, local.tm_hour, local.tm_min, local.tm_sec);
  return buf;
}

// CONVENIENCE WRAPPER
// ONE buffer, reused in place for every iteration. The number is stored as little-endian base-100 limbs (2 decimal
// digits per byte), which halves the memory traffic of the hot loop - the run is DRAM-bandwidth bound past ~40M digits,
// so that converts almost directly into throughput.
//
// Because the representation is little-endian, growth happens at the HIGH end of the buffer and limb boundaries never
// move, so a carry-out simply writes one more limb - no front slack or shifting offset is required.

// Fallback when the topology query fails, chosen to land in a reasonable range on an unqueryable machine.
static const int kFallbackParallelThreads = 8;

// Default thread count: PHYSICAL cores, not logical processors.
//
// Past the parallel threshold this loop is DRAM-bandwidth bound, which makes hardware_concurrency() the obvious choice
// and close to the worst one, for two reasons:
//   1. SMT siblings share a core's L1d, and that L1d is the resource the carry-staging block is sized against. Counting
//      logical processors double-counts each core and halves the per-thread L1 budget, which fights --block-size.
//   2. Throughput saturates the memory controllers well before it saturates the cores, so threads past the knee add
//      dispatch and carry-scan overhead while contending for bandwidth that is already spoken for. The pool spins, so
//      those extra threads burn cores rather than waiting politely.
//
// Physical cores are a better PROXY, not the true optimum: the knee is a property of the memory subsystem and cannot be
// derived from topology at all - on a bandwidth-saturated part the best value may well be half the core count. This
// only aims to land in the right neighbourhood on an unfamiliar machine; --threads and a sweep are how the actual
// number gets found. See the tuning walkthrough in README.md.
static int default_parallel_threads()
{
  DWORD bytes = 0;
  // Expected to fail with ERROR_INSUFFICIENT_BUFFER; that is how the required size is obtained.
  if (GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes) ||
      GetLastError() != ERROR_INSUFFICIENT_BUFFER || bytes == 0)
  {
    return kFallbackParallelThreads;
  }

  std::vector<uint8_t> buf(bytes);
  auto *info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data());
  if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &bytes))
  {
    return kFallbackParallelThreads;
  }

  // One variable-length record per physical core, so counting records counts cores regardless of SMT. Walked by byte
  // offset because Size varies per record; this is not an array and cannot be indexed.
  //
  // The bound is against the record HEADER, not sizeof(the struct): a record is only as large as its own group-affinity
  // array needs, so the final record is typically smaller than the full struct. Bounding by sizeof would drop it and
  // undercount by one core.
  const DWORD kHeader = offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor);
  int cores = 0;
  for (DWORD off = 0; off + kHeader <= bytes;)
  {
    auto *rec = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data() + off);
    if (rec->Size == 0 || off + rec->Size > bytes)
    {
      break;
    }
    if (rec->Relationship == RelationProcessorCore)
    {
      ++cores;
    }
    off += rec->Size;
  }

  if (cores <= 0)
  {
    return kFallbackParallelThreads;
  }
  // Clamped for the same reason --threads is: kMaxThreads sizes the fixed per-thread arrays in the core.
  return (cores > (int)kMaxThreads) ? (int)kMaxThreads : cores;
}

// The parallel core splits the low half into per-thread spans rounded DOWN to a whole carry block, so the threads a
// given length can keep busy is
//   digits / (4 * block-size).
// That bound is length-dependent and the core reduces its own thread count to fit, so it is NOT mirrored as a CLI
// clamp: deriving one from the parallel threshold would impose the cutover's worst case on every length. At the 98304
// cutover only 3 threads fit, but 16M digits fits ~512 and 1e9 fits ~30000, and a long run spends almost none of its
// life near the cutover. --threads is therefore bounded only by kMaxThreads, which sizes the fixed per-thread arrays
// in the core and is a real limit at every length.

// Zero-padding width for the iteration and digit counts in save file names. Ten covers ~1.6e10 iterations and 1e10
// digits, both far past what a run can physically reach - a 1e10-digit number needs 5 GB packed and a 15 GB transient
// while the buffer doubles.
static const int kNameCountWidth = 10;

// Set when a --stop-* limit ends the run, to distinguish that from a completed search.
static bool g_stop_limit_reached = false;

// 'start_limbs' holds the number to begin from, already packed as little-endian base-100 limbs ((start_len+1)/2 of
// them), and 'first_iteration' the count it is already at, so a run can either start fresh from the initial value or
// resume mid-trajectory from a loaded save without ever materialising a one-byte-per-digit copy.
bool is_sum_palindrome_from(std::string_view num, const uint8_t *start_limbs, size_t start_len,
                            uint64_t first_iteration)
{
  if (start_len == 0)
  {
    return false;
  }

  packed_init_tables();

  size_t len = start_len;   // decimal digits
  size_t L = (len + 1) / 2; // limbs in use

  // Room to grow at the top, plus one spare limb for the carry-out write.
  size_t cap = std::max<size_t>(1'000'000, L * 2);
  PackedBuffer storage(cap + 1, L, g_config.threads);
  std::copy_n(start_limbs, L, storage.data());

  // Fires immediately on the first iteration (a baseline line for the starting value), then on absolute multiples of
  // status_iterations - see the reload below.
  uint64_t print_countdown = 1;
  uint32_t control_interval = kControlPollInterval;
  uint32_t control_countdown = kControlPollInterval;
  const size_t save_digits = g_config.save_digit_interval;
  size_t next_length_milestone = save_digits ? (len / save_digits + 1) * save_digits : 0;
  const size_t status_digits = g_config.status_digits;
  size_t next_status_length = status_digits ? (len / status_digits + 1) * status_digits : 0;
  const auto start_time = std::chrono::steady_clock::now();
  auto last_control_poll = start_time;
  auto next_save_time = start_time + std::chrono::duration<double, std::ratio<60>>(g_config.save_minutes);
  auto next_status_time = start_time + std::chrono::duration<double, std::ratio<60>>(g_config.status_minutes);
  const auto stop_time = start_time + std::chrono::duration<double, std::ratio<60>>(g_config.stop_minutes);

  // Column widths for the status table. Both quantities are bounded by RAM, not by size_t: at 0.5 bytes per digit
  // packed, with a 3x transient while the buffer doubles, a 1 TB machine tops out near 7e11 digits (12 digits) and
  // ~1.6e12 iterations (13 digits). Exceeding a width only widens the row - setw pads, it never truncates.
  const int kIterationWidth = 13;
  const int kLengthWidth = 12;

  std::cout << kTimestampHeader << ' ' << std::setw(kIterationWidth) << "Iteration" << ' ' << std::setw(kLengthWidth)
            << "Length" << std::endl;

  // Flushed explicitly: when stdout is redirected to a file it is block buffered, so progress lines would otherwise sit
  // unwritten for a long time on a run this slow to produce output. Milestones are rare enough that the flush costs
  // nothing.
  //
  // Suppresses an exact repeat of the previous line. The cadences are independent by design, so several can come due on
  // the same iteration - a digit milestone, an iteration milestone and a timed status can all land together, and the
  // terminating and stopping paths report before saving as well. The row is identical in every case, so printing it
  // more than once conveys nothing and only makes the log look like the run stalled.
  uint64_t last_reported_iteration = UINT64_MAX;
  size_t last_reported_len = 0;
  auto report = [&](uint64_t iteration)
  {
    if (iteration == last_reported_iteration && len == last_reported_len)
    {
      return;
    }
    last_reported_iteration = iteration;
    last_reported_len = len;

    std::cout << timestamp_now() << ' ' << std::setw(kIterationWidth) << iteration << ' ' << std::setw(kLengthWidth)
              << len << std::endl;
  };

  // Called from the low-memory path as well, so an allocation failure here must not escape. The writer streams the
  // packed limbs straight to disk, so the only allocations left are the file name and the writer's fixed block.
  auto autosave = [&](uint64_t iteration)
  {
    std::ostringstream name;
    name << num << '_' << std::setfill('0') << std::setw(kNameCountWidth) << iteration << '_'
         << std::setw(kNameCountWidth) << len << ".isf";
    const std::string path = name.str();
    try
    {
      if (!save_result(path, num, iteration, storage.data(), len))
      {
        std::cout << "WARNING: failed to write " << path << std::endl;
      }
    }
    catch (const std::bad_alloc &)
    {
      std::cout << "WARNING: not enough memory to write " << path << std::endl;
    }
  };

  // 0-based: iteration N is the Nth reverse-add, so the first add is 0. At the top of the loop the counter is also the
  // number of adds already completed.
  //
  // The palindrome test sits at the TOP rather than after the add so it also sees the starting value: a number that is
  // already a palindrome (or a resumed save that ended on one) terminates at iteration 0. It also means 'iteration' is
  // always the true count of completed adds, so report and autosave need no adjustment.
  for (uint64_t iteration = first_iteration;; ++iteration)
  {
    if (packed_is_palindrome(storage.data(), L, len))
    {
      report(iteration);
      autosave(iteration);
      return true;
    }

    // Reloaded with the distance to the next MULTIPLE of the interval rather than a full interval, so the printed
    // counts are round regardless of where the run started. A save resumed at iteration 48,316,988 with a 100,000
    // interval therefore reports 48,400,000 next, not 48,416,988, and its lines align with those of any other run.
    if (--print_countdown == 0) [[unlikely]]
    {
      const uint64_t interval = g_config.status_iterations;
      print_countdown = interval ? interval - iteration % interval : UINT64_MAX;
      report(iteration);
    }

    // Absolute iteration limit. Checked before the add so the run stops HAVING completed exactly this many iterations.
    if (g_config.stop_iteration && iteration >= g_config.stop_iteration) [[unlikely]]
    {
      report(iteration);
      autosave(iteration);
      std::cout << "Stopping at iteration " << iteration << " (--stop-iteration)." << std::endl;
      g_stop_limit_reached = true;
      return false;
    }

    // Length limit. Same placement as the iteration limit: the reported length is the one actually reached.
    if (g_config.stop_digits && len >= g_config.stop_digits) [[unlikely]]
    {
      report(iteration);
      autosave(iteration);
      std::cout << "Stopping at " << len << " digits (--stop-digits)." << std::endl;
      g_stop_limit_reached = true;
      return false;
    }

    // DEADLINE-CRITICAL, so it is checked every iteration instead of on the poll cadence below. Windows gives a logoff
    // or shutdown handler about 5 seconds in total, and at 100M digits one poll interval alone already exceeded that -
    // the process was killed with the trajectory unsaved before the loop ever looked at the flag.
    //
    // A relaxed load of a bool that is almost never written costs a couple of cycles against an iteration that streams
    // the entire number through memory, so this is free precisely where it matters, and merely cheap at small lengths.
    //
    // This only pulls the service block forward; the handling itself is not duplicated.
    //
    // The floor is one iteration: a safe point exists only between reverse-adds, so past a few hundred million digits a
    // single iteration can still outlast the shutdown window on its own. Periodic autosave, not this, is what protects
    // a run at that scale.
    if (g_save_requested.load(std::memory_order_relaxed)) [[unlikely]]
    {
      control_countdown = 0;
    }

    // Keyboard, pause and the timed cadences. Serviced on a measured wall-clock interval rather than a fixed iteration
    // count so that response stays bounded as iteration cost grows - see kControlPollInterval.
    if (control_countdown-- == 0) [[unlikely]]
    {
      if (space_pressed())
      {
        g_paused = !g_paused;
      }

      if (g_paused)
      {
        std::cout << "Paused at iteration " << iteration << " - press space to continue." << std::endl;
        while (g_paused && !g_save_requested)
        {
          if (space_pressed())
          {
            g_paused = false;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (!g_save_requested)
        {
          std::cout << "Resumed." << std::endl;
        }
      }

      if (g_save_requested) [[unlikely]]
      {
        autosave(iteration);
        g_save_complete.store(true, std::memory_order_release);
        g_save_requested = false;
        if (g_quit_requested)
        {
          report(iteration);
          std::cout << "Interrupted - state saved." << std::endl;
          return false;
        }
      }

      // Time-based autosave. Polled here rather than in the hot loop so the clock is never read per iteration; the
      // control poll interval bounds how late a save can be.
      if (g_config.save_minutes > 0.0) [[unlikely]]
      {
        const auto now = std::chrono::steady_clock::now();
        if (now >= next_save_time)
        {
          next_save_time = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                     std::chrono::duration<double, std::ratio<60>>(g_config.save_minutes));
          report(iteration);
          autosave(iteration);
        }
      }

      // Time-based progress line, on the same poll for the same reason.
      if (g_config.status_minutes > 0.0) [[unlikely]]
      {
        const auto now = std::chrono::steady_clock::now();
        if (now >= next_status_time)
        {
          next_status_time = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                       std::chrono::duration<double, std::ratio<60>>(g_config.status_minutes));
          report(iteration);
        }
      }

      // Wall-clock limit. Also on the control poll, so the run can overshoot by at most one poll interval.
      if (g_config.stop_minutes > 0.0 && std::chrono::steady_clock::now() >= stop_time) [[unlikely]]
      {
        report(iteration);
        autosave(iteration);
        std::cout << "Stopping after " << g_config.stop_minutes << " minutes (--stop-minutes)." << std::endl;
        g_stop_limit_reached = true;
        return false;
      }

      // Retune the interval against what the last stretch of iterations actually cost, holding the poll cadence near
      // kControlPollTarget in wall-clock terms at any length. Measuring rather than deriving from 'len' is deliberate:
      // iteration cost also depends on thread count, on whether the parallel core is engaged, and on the machine - the
      // same workload has been observed to differ 3.5x between sessions on identical hardware.
      //
      // Timed from the END of the previous service block, so a pause, a save or a status print is not billed to the
      // compute stretch. Charging those here would collapse the interval to 1 after every save and then let it climb
      // back, polling hard for no reason in between.
      const auto poll_end = std::chrono::steady_clock::now();
      const auto spent = poll_end - last_control_poll;
      if (spent > std::chrono::steady_clock::duration::zero())
      {
        const double tuned = (double)control_interval *
                             (std::chrono::duration<double>(kControlPollTarget).count() /
                              std::chrono::duration<double>(spent).count());
        control_interval = (uint32_t)std::clamp(tuned, 1.0, (double)kControlPollInterval);
      }
      else
      {
        // Too fast to measure, which only happens at trivial lengths. Back off to the ceiling.
        control_interval = kControlPollInterval;
      }
      last_control_poll = poll_end;
      control_countdown = control_interval;
    }

    // Length milestone. The number grows well under a digit per iteration, so a milestone is never skipped, but loop
    // anyway to stay correct if the growth rate assumption ever changes.
    if (save_digits && len >= next_length_milestone) [[unlikely]]
    {
      while (len >= next_length_milestone)
      {
        next_length_milestone += save_digits;
      }
      report(iteration);
      autosave(iteration);
    }

    // Digit milestone for progress lines only. Separate from the autosave milestone so status and save cadences can be
    // set independently; when both fire on the same iteration the line is printed once by whichever runs first.
    if (status_digits && len >= next_status_length) [[unlikely]]
    {
      while (len >= next_status_length)
      {
        next_status_length += status_digits;
      }
      report(iteration);
    }

    // Iteration-count autosave, on absolute multiples so saves line up across runs the way progress lines do.
    if (g_config.save_iterations && iteration && iteration % g_config.save_iterations == 0) [[unlikely]]
    {
      report(iteration);
      autosave(iteration);
    }

    // Growth happens at the high end, so the check is against the top of the buffer. One spare limb beyond L is
    // needed for a carry-out write, hence the +2 margin.
    if (L + 2 > cap) [[unlikely]]
    {
      // The search is unbounded, so this keeps doubling until the process is interrupted or the allocation fails.
      // Doubling makes growth amortized O(1) per digit, with a peak transient of 3x while both buffers live.
      //
      // A failed doubling is the natural end of an unbounded run, so treat it like an interrupt rather than letting
      // bad_alloc escape and kill the process with the trajectory unsaved. The save buffer needs about len bytes
      // against the 2*cap that just failed, so it usually still fits.
      const size_t new_cap = cap * 2;
      try
      {
        // Touched in parallel like the original, so the doubling also re-establishes NUMA placement against the
        // current L rather than inheriting the old buffer's layout.
        PackedBuffer grown(new_cap + 1, L, g_config.threads);
        std::copy_n(storage.data(), L, grown.data());
        storage.swap(grown);
        cap = new_cap;
      }
      catch (const std::bad_alloc &)
      {
        g_out_of_memory = true;
        report(iteration);
        std::cout << "Out of memory growing buffer to " << new_cap << " bytes - saving state." << std::endl;
        autosave(iteration);
        return false;
      }
    }

    // Contract: the palindrome test happens at the top of the next pass, so this only performs the add.
    //
    // Barrier dispatch costs a few microseconds, so threading only pays off once a single iteration is long enough to
    // amortize it; the default threshold sits well past that break-even point.
    len = (len >= g_config.parallel_threshold)
              ? packed_iterate_parallel(storage.data(), L, len, g_config.threads)
              : packed_iterate(storage.data(), L, len);
    L = (len + 1) / 2;
  }
}

// Starts a fresh run from the decimal string 'num'.
bool is_sum_palindrome(std::string_view num)
{
  try
  {
    std::vector<uint8_t> start(num.size());
    for (size_t i = 0; i < num.size(); ++i)
    {
      start[i] = (uint8_t)(num[i] - '0');
    }
    std::vector<uint8_t> start_limbs((num.size() + 1) / 2);
    pack_digits(start.data(), start.size(), start_limbs.data());
    return is_sum_palindrome_from(num, start_limbs.data(), start.size(), 0);
  }
  catch (const std::bad_alloc &)
  {
    g_out_of_memory = true;
    std::cout << "Out of memory allocating search state." << std::endl;
    return false;
  }
}

// Resumes a run from a previously saved .isf file.
bool is_sum_palindrome_resume(const std::string &path)
{
  std::string initial;
  uint64_t iteration = 0;
  std::vector<uint8_t> limbs;
  size_t digit_count = 0;
  if (!load_result(path, initial, iteration, limbs, digit_count))
  {
    return false;
  }
  std::cout << "Resuming " << initial << " from iteration " << iteration << " (" << digit_count << " digits)"
            << std::endl;
  try
  {
    return is_sum_palindrome_from(initial, limbs.data(), digit_count, iteration);
  }
  catch (const std::bad_alloc &)
  {
    g_out_of_memory = true;
    std::cout << "Out of memory allocating search state." << std::endl;
    return false;
  }
}

// Exit codes. The search itself is unbounded unless a --stop-* limit is given - otherwise it runs until a palindrome is
// found, the user interrupts it, or the machine runs out of memory.
//
// 1 is deliberately left unused: the CRT and the OS both produce it for abnormal termination (an unhandled exception,
// or a fatal startup error before main runs), so reserving it keeps "the program itself decided to stop" distinct from
// "the program died". A --stop-* limit is a normal, requested outcome and gets its own code below.
static const int kExitPalindrome = 0;     // Palindrome found.
static const int kExitInterrupted = 2;    // Ctrl+C / close / logoff / shutdown.
static const int kExitLoadFailed = 3;     // Could not load the requested save.
static const int kExitOutOfMemory = 4;    // Buffer growth failed; state saved.
static const int kExitUnsupportedCpu = 5; // Required ISA extensions missing.
static const int kExitBadArguments = 6;   // Malformed command line.
static const int kExitStopLimit = 7;      // A --stop-digits/-minutes/-iteration limit was reached; state saved.

// Verifies the CPU implements every instruction-set extension this binary was compiled for. The whole project is built
// with /arch:AVX512, so there is no fallback path - the goal is a clear diagnostic instead of an illegal instruction
// fault on an older machine.
//
// VBMI is the one that actually matters and is easy to overlook. The digit reversal and the base-100 nibble handling
// use _mm512_permutexvar_epi8 and _mm512_permutex2var_epi8, which are AVX512VBMI (Ice Lake and later). A Skylake-SP or
// Cascade Lake part has F/BW/DQ/VL but NOT VBMI, so checking only for "AVX-512" would wave those through and then crash
// in the hot loop.
//
// Checking CPUID feature bits alone is not sufficient: the OS must also have enabled ZMM state saving on context
// switch. That is what the OSXSAVE bit plus the XCR0 opmask/ZMM bits confirm. Without this a feature-capable CPU under
// an OS that does not preserve the wide registers would corrupt state rather than fault cleanly.
static bool check_cpu_support()
{
  int regs[4] = {0, 0, 0, 0};

  __cpuid(regs, 0);
  const int max_leaf = regs[0];
  if (max_leaf < 7)
  {
    std::cout << "ERROR: CPU does not support CPUID leaf 7; AVX-512 required." << std::endl;
    return false;
  }

  // OSXSAVE (leaf 1, ECX bit 27) must be set before XGETBV may be executed.
  __cpuid(regs, 1);
  const bool osxsave = (regs[2] & (1 << 27)) != 0;
  if (!osxsave)
  {
    std::cout << "ERROR: OS does not support XSAVE/XGETBV; AVX-512 required." << std::endl;
    return false;
  }

  // XCR0 bits 5/6/7 = opmask, ZMM_Hi256, Hi16_ZMM. Bit 2 = YMM, bit 1 = XMM. All are needed for the OS to preserve full
  // ZMM state across a switch.
  const unsigned long long xcr0 = _xgetbv(0);
  const unsigned long long kZmmState = (1ull << 1) | (1ull << 2) | (1ull << 5) | (1ull << 6) | (1ull << 7);
  if ((xcr0 & kZmmState) != kZmmState)
  {
    std::cout << "ERROR: OS is not preserving AVX-512 register state." << std::endl;
    return false;
  }

  __cpuidex(regs, 7, 0);
  const int ebx = regs[1];
  const int ecx = regs[2];

  struct Feature
  {
    const char *name;
    bool present;
  };
  const Feature features[] = {
      {"AVX512F", (ebx & (1 << 16)) != 0},  {"AVX512DQ", (ebx & (1 << 17)) != 0},  {"AVX512BW", (ebx & (1 << 30)) != 0},
      {"AVX512VL", (ebx & (1 << 31)) != 0}, {"AVX512VBMI", (ecx & (1 << 1)) != 0},
  };

  bool ok = true;
  for (const Feature &f : features)
  {
    if (!f.present)
    {
      std::cout << "ERROR: CPU is missing required extension " << f.name << "." << std::endl;
      ok = false;
    }
  }

  if (!ok)
  {
    std::cout << "This build requires an Intel Ice Lake / AMD Zen 4 class CPU or newer." << std::endl;
  }
  return ok;
}

// Prints usage. The starting number and a saved .isf are alternative ways to supply the same thing: an initial value
// and the iteration count it is already at.
static void print_usage(const char *exe)
{
  std::cout << "lychrel.exe version " << LYCHREL_VERSION << "\n"
            << "\n"
            << "Usage: " << exe << " <start-number> | <save.isf> [options]\n"
            << "\n"
            << "  <start-number>  decimal digits to start a fresh run from, e.g. 196\n"
            << "  <save.isf>      resume a previously saved run\n"
            << "\n"
            << "Options:\n"
            << "  --status-iterations <n>  iterations between progress lines (default 100000, 0 disables)\n"
            << "  --status-digits <n>      digits of growth between progress lines (default 1000000, 0 disables)\n"
            << "  --status-minutes <n>     wall minutes between progress lines (default 0, disabled)\n"
            << "  --save-iterations <n>    iterations between autosaves (default 100000000, 0 disables)\n"
            << "  --save-digits <n>        digits of growth between autosaves (default 100000000, 0 disables)\n"
            << "  --save-minutes <n>       wall minutes between autosaves (default 1440, 0 disables)\n"
            << "  --stop-iteration <n>     stop on reaching this iteration (default 0, disabled)\n"
            << "  --stop-digits <n>        stop on reaching this many digits (default 0, disabled)\n"
            << "  --stop-minutes <n>       stop after this many wall minutes (default 0, disabled)\n"
            << "  --threads <n>            worker threads for the parallel core (default " << default_parallel_threads()
            << " = physical cores, 1 forces serial)\n"
            << "  --block-size <n>         carry staging block size in limbs (default 8192, power of two)\n"
            << "  --parallel-threshold <n> digits before the parallel core engages (default "
            << DEFAULT_PARALLEL_THRESHOLD << ", 0 always)\n"
            << "  --no-crc-check           treat a save-file CRC mismatch as a warning instead of an error\n"
            << "  --no-pin                 disable NUMA thread pinning and parallel first-touch\n"
            << "\n"
            << "While running:\n"
            << "  space           pause or resume; the run stays in memory while paused\n"
            << "  Ctrl+C          save the current state and exit\n"
            << "\n"
            << "  Closing the window, logging off, and shutting down also save before exiting.\n"
            << "  Saves are written to <start>_<iteration>_<digits>.isf in the current directory,\n"
            << "  and any of them can be passed back in to resume.\n"
            << "\n"
            << "  The search is unbounded: 196 and other candidates run until interrupted.\n"
            << std::endl;
}

// A .isf save is distinguished from a starting number by extension; a start value is all decimal digits.
static bool looks_like_save_file(std::string_view s)
{
  return s.size() > 4 && s.substr(s.size() - 4) == ".isf";
}

static bool all_digits(std::string_view s)
{
  if (s.empty())
  {
    return false;
  }
  for (const char c : s)
  {
    if (c < '0' || c > '9')
    {
      return false;
    }
  }
  return true;
}

// Entry point. Parses the command line, then runs a fresh search or resumes a saved one.
int main(int argc, char **argv)
{
  // Checked before anything else touches a vector code path.
  if (!check_cpu_support())
  {
    return kExitUnsupportedCpu;
  }

  SetConsoleCtrlHandler(console_ctrl_handler, TRUE);

  std::string_view source;
  for (int i = 1; i < argc; ++i)
  {
    const std::string_view arg = argv[i];

    if (arg == "--help" || arg == "-h" || arg == "/?")
    {
      print_usage(argv[0]);
      return kExitPalindrome;
    }

    if (arg == "--no-crc-check")
    {
      g_config.no_crc_check = true;
      continue;
    }

    if (arg == "--no-pin")
    {
      g_numa_pin = false;
      continue;
    }

    // Options taking a value. --save-minutes, --status-minutes and --stop-minutes accept a decimal; the rest are
    // non-negative integers.
    const bool is_minutes = (arg == "--save-minutes" || arg == "--status-minutes" || arg == "--stop-minutes");
    if (is_minutes || arg == "--status-iterations" || arg == "--status-digits" || arg == "--save-iterations" ||
        arg == "--save-digits" || arg == "--stop-iteration" || arg == "--stop-digits" || arg == "--threads" ||
        arg == "--block-size" || arg == "--parallel-threshold")
    {
      if (i + 1 >= argc)
      {
        std::cout << "ERROR: " << arg << " requires a value." << std::endl;
        return kExitBadArguments;
      }
      const std::string_view value = argv[++i];
      if (!all_digits(value) && !is_minutes)
      {
        std::cout << "ERROR: " << arg << " expects a non-negative integer, got '" << value << "'." << std::endl;
        return kExitBadArguments;
      }

      if (is_minutes)
      {
        char *end = nullptr;
        const double v = std::strtod(value.data(), &end);
        if (end != value.data() + value.size() || !std::isfinite(v) || v < 0.0)
        {
          std::cout << "ERROR: " << arg << " expects a non-negative number, got '" << value << "'." << std::endl;
          return kExitBadArguments;
        }
        if (arg == "--save-minutes")
        {
          g_config.save_minutes = v;
        }
        else if (arg == "--status-minutes")
        {
          g_config.status_minutes = v;
        }
        else
        {
          g_config.stop_minutes = v;
        }
      }
      else if (arg == "--status-iterations")
      {
        g_config.status_iterations = std::strtoull(value.data(), nullptr, 10);
      }
      else if (arg == "--status-digits")
      {
        g_config.status_digits = (size_t)std::strtoull(value.data(), nullptr, 10);
      }
      else if (arg == "--save-iterations")
      {
        g_config.save_iterations = std::strtoull(value.data(), nullptr, 10);
      }
      else if (arg == "--save-digits")
      {
        g_config.save_digit_interval = (size_t)std::strtoull(value.data(), nullptr, 10);
      }
      else if (arg == "--stop-iteration")
      {
        g_config.stop_iteration = std::strtoull(value.data(), nullptr, 10);
      }
      else if (arg == "--stop-digits")
      {
        g_config.stop_digits = (size_t)std::strtoull(value.data(), nullptr, 10);
      }
      else if (arg == "--block-size")
      {
        const unsigned long long n = std::strtoull(value.data(), nullptr, 10);
        // A power of two is required, not merely preferred: block boundaries are derived by integer division and the
        // staging loop steps 64 bytes at a time, so a non-power-of-two would leave a partial vector at the end of every
        // block. The range keeps the stage buffer within the fixed-size stack array it is stored in.
        if (n < kMinCarryBlockSize || n > kMaxCarryBlockSize || (n & (n - 1)) != 0)
        {
          std::cout << "ERROR: --block-size expects a power of two between " << kMinCarryBlockSize << " and "
                    << kMaxCarryBlockSize << ", got '" << value << "'." << std::endl;
          return kExitBadArguments;
        }
        g_carry_block = (size_t)n;
      }
      else if (arg == "--parallel-threshold")
      {
        // No range check beyond what size_t gives: 0 deliberately means "always parallel" (the sweep mode described
        // above RunConfig), and an absurdly large value simply pins the run to the serial core, which is a
        // legitimate A/B rather than an error.
        g_config.parallel_threshold = (size_t)std::strtoull(value.data(), nullptr, 10);
      }
      else if (arg == "--threads")
      {
        const unsigned long long n = std::strtoull(value.data(), nullptr, 10);
        if (n < 1)
        {
          std::cout << "ERROR: --threads expects a value of 1 or more, got '" << value << "'." << std::endl;
          return kExitBadArguments;
        }
        // kMaxThreads is the only hard ceiling: it sizes the fixed per-thread arrays in the core. The per-length
        // structural bound is applied inside packed_iterate_parallel, which reduces its own thread count to what the
        // current length can keep busy, so a value that is too large for short lengths simply takes effect later in
        // the run rather than being refused.
        if (n > (unsigned long long)kMaxThreads)
        {
          std::cout << "WARNING: --threads " << value << " clamped to " << kMaxThreads << "." << std::endl;
          g_config.threads = (int)kMaxThreads;
        }
        else
        {
          g_config.threads = (int)n;
        }

        // Oversubscription is permitted but almost never wanted: the pool spins, so more workers than logical
        // processors means they contend for cores instead of overlapping memory traffic. Warned rather than clamped
        // because the core requires a thread per dispatched id - silently using fewer would leave segments
        // unprocessed.
        const unsigned hw = std::thread::hardware_concurrency();
        if (hw != 0 && (unsigned)g_config.threads > hw)
        {
          std::cout << "WARNING: --threads " << g_config.threads << " exceeds the " << hw
                    << " logical processors available; the pool spins, so this will oversubscribe." << std::endl;
        }
      }
      continue;
    }

    if (!arg.empty() && arg[0] == '-')
    {
      std::cout << "ERROR: unknown option '" << arg << "'." << std::endl;
      print_usage(argv[0]);
      return kExitBadArguments;
    }

    if (!source.empty())
    {
      std::cout << "ERROR: more than one starting value given ('" << source << "' and '" << arg << "')." << std::endl;
      return kExitBadArguments;
    }
    source = arg;
  }

  // After the whole parse, so it reflects the final --threads regardless of argument order. This is the denominator the
  // id -> node mapping divides by, and first-touch and pinning must both use the same one or they would disagree about
  // which node owns a span.
  g_pin_total = g_config.threads;

  // One line of effective configuration, to stderr so it never contaminates a redirected result stream. It guards
  // against comparing two binaries that differ in a way their output does not show: an x86 build is correct but around
  // half the speed of an x64 one, and pointer width is the cheap proxy for the architecture the Makefile enforces.
  // Pinning is reported as its EFFECTIVE state, not as the --no-pin flag, because numa_bind_thread silently does
  // nothing on a flat or unavailable topology - a machine where pinning is expected but absent is a real cliff with no
  // other signal.
  //
  // Diagnostic build only: the failure it guards against is a measurement failure, and measurement is what
  // lychrel-diag.exe is for. A normal run has no use for it, so the default build stays silent on stderr.
#ifdef LYCHREL_DIAG
  {
    const int nodes = numa_node_count();
    const char *pin = !g_numa_pin ? "off (--no-pin)" : (nodes < 2 ? "inactive (1 NUMA node)" : "on");
    fprintf(stderr,
            "[config] %zu-bit  threads=%d  block-size=%zu  parallel-threshold=%zu  numa-nodes=%d  pinning=%s\n",
            sizeof(void *) * 8, g_config.threads, g_carry_block, g_config.parallel_threshold, nodes, pin);
    fflush(stderr);
  }
#endif

  if (!source.empty())
  {
    // A save file resumes mid-trajectory; a bare number starts fresh from iteration 0.
    if (looks_like_save_file(source))
    {
      if (is_sum_palindrome_resume(std::string(source)))
      {
        return kExitPalindrome;
      }
      if (g_quit_requested)
      {
        return kExitInterrupted;
      }
      if (g_stop_limit_reached)
      {
        return kExitStopLimit;
      }
      if (g_out_of_memory)
      {
        return kExitOutOfMemory;
      }
      // The only other way out is a failed load, already reported by load_result.
      return kExitLoadFailed;
    }

    if (!all_digits(source))
    {
      std::cout << "ERROR: '" << source << "' is neither a decimal number nor a .isf save file." << std::endl;
      return kExitBadArguments;
    }

    std::cout << "Starting " << source << " from iteration 0 (" << source.size() << " digits)" << std::endl;
    if (is_sum_palindrome(source))
    {
      return kExitPalindrome;
    }
    if (g_quit_requested)
    {
      return kExitInterrupted;
    }
    if (g_stop_limit_reached)
    {
      return kExitStopLimit;
    }
    if (g_out_of_memory)
    {
      return kExitOutOfMemory;
    }
    return kExitPalindrome;
  }

  // A starting value is the whole point of the program, so there is nothing sensible to do without one.
  std::cout << "ERROR: no starting number or save file given." << std::endl;
  print_usage(argv[0]);
  return kExitBadArguments;
}
