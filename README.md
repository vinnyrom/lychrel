# lychrel

A fast reverse-and-add search for [Lychrel numbers](https://en.wikipedia.org/wiki/Lychrel_number).

A number is tested by repeatedly adding it to its own decimal reversal and checking whether the sum is a palindrome.
Most numbers converge within a few steps. Some — 196 being the famous one — have resisted for over a billion iterations,
carrying the number past a billion digits, without ever producing a palindrome. This program runs that search,
unbounded, until it finds a palindrome or you stop it.

The interesting part is not the algorithm, which is trivial, but making it fast at scale. Past 40 million digits the
loop is DRAM-bandwidth bound, so the implementation is built around moving fewer bytes.

## Will 196 ever run to completion?

Almost certainly not — and no one can prove it either way.

The reason to expect it never terminates is a counting argument. Palindromes get sparse fast: among *n*-digit numbers,
roughly 10^⌈n/2⌉ of 10^n are palindromic, so the density is about 10^−n/2. A reverse-add roughly preserves length while
adding about 0.4 digits per step, so each iteration is heuristically an independent draw against a target that is
shrinking exponentially. Sum those probabilities over all future iterations and the tail converges: the expected number
of remaining hits is already essentially zero a few thousand digits in. Every iteration makes success *less* likely than
the one before, so the total remaining chance is dominated by the steps you have already computed and failed.

The empirical record agrees. 196 was carried past a billion digits by
[Romain Dolbeau](http://www.dolbeau.name/dolbeau/p196/p196.html) in February 2015. Every candidate that was going to
converge did so almost immediately: 89 takes 24 steps, 10911 takes 55.  It was discovered by
[Anton Stefanov](https://jasondoucette.com/worldrecords.html#CurrentMostRecord) in January 2021 that the 23 digit number
13968441660506503386020 solves after 289 iterations to form a 142 digit palindrome. No other number is known to take more
iterations, let alone billions.

None of that is a proof, and the gap is not a technicality. The independence assumption is doing all the work, and it is
plainly false: reverse-add is deterministic, and the digit distributions it produces have structure the counting
argument ignores. No number has ever been proven Lychrel in base 10.

It is provable in principle, just not here. In base 2, 10110 (22 in decimal) *is* provably Lychrel, shown by Ethan Brown
in 1990 via an invariant on the binary digit pattern that survives every reverse-add. Base 10 has resisted every attempt
at finding a comparable invariant, and a proof — if one ever arrives — will come from that direction rather than from
iteration count. No amount of running this program can settle the question; a search can only ever falsify the
conjecture, never confirm it.

So this implementation is an exercise in probing engineering limits, not searching in any hopeful sense. What makes it
worth building is that the constraints are real and unforgiving: unbounded growth, checkpoints that must survive being
killed mid-write, and a working set that outgrows every level of cache and then main memory. Those are the same problems
any long-running unbounded computation faces, and 196 is an honest way to meet them.

## What would catching up cost?

The published milestones are the natural yardstick, and both belong to Dolbeau's `p196_mpi`: a billion *iterations* on
October 2, 2011, which carried 196 to over 413M digits, and a billion *digits* in February 2015. The two are easy to
confuse. At the observed growth of about 0.4139 digits per iteration they are far apart — a billion digits is roughly
2.4 billion iterations, so the 2015 result is the more demanding by a factor of nearly six.

Reaching a billion digits on one machine is a memory-bandwidth problem end to end. Normalized cost per digit per
iteration, measured with `tools\measure.ps1` on an Azure `Standard_D16ds_v5` (16 vCPUs on 8 physical Xeon Platinum
8370C cores, Ice Lake-SP, 64GB):

|      length | k (s/digit/iteration) |  GB/s |
| ----------: | --------------------: | ----: |
|  25,000,000 |             9.012e-12 | 111.1 |
| 100,000,000 |             1.111e-11 |  90.2 |
| 400,000,000 |             1.367e-11 |  73.2 |

k climbs about 21% per 4x in length as the loop becomes progressively more DRAM-bound, and it is still climbing at the
largest length measured here. That trend is the whole story, because cost compounds: one iteration at length L costs
k*L, and length grows 0.4139 digits per iteration, so time to reach a length integrates as k*L^2 / (2 * 0.4139). The
quadratic dominates so completely that everything below 100M digits is under 2% of the run: essentially all of the time
is spent in the final stretch, at the largest and slowest lengths.

No total is quoted here on purpose. Putting a figure on a billion digits means extrapolating k a factor of 2.5 beyond
anything measured above, and these are shared-tenant numbers: a D-series VM shares host memory bandwidth with other
guests, so the GB/s column carries a noisy-neighbour component that bare metal would not. Memory capacity, at least, is
not the binding constraint - a billion digits packs into ~500MB before growth headroom and the parallel carry
structures, comfortably inside this machine's 64GB. Plug the measured trend in and the answer lands in months of
uninterrupted compute rather than weeks; that is as precise as the data supports. It would also be months to
*replicate* a decade-old result, against a distributed MPI code that had 60+ InfiniBand-connected nodes to work with -
the comparison is less flattering than any single number suggests.

Better hardware moves that, and by roughly the amount you would guess from the bandwidth alone: past 40M digits the
loop is DRAM-bound, so time scales inversely with achievable memory bandwidth, and a modern many-channel server - a
single-socket EPYC with twelve DDR5 channels, say - should bring a billion digits down to weeks rather than months. No
figure is offered because none was measured, and on that class of machine the tuning walkthrough below is what would
decide it: a bad NUMA placement makes the run latency-bound rather than bandwidth-bound and gives back most of the
advantage.

## How it works

**Packed base-100 storage.** Digits are stored two per byte, little-endian, so limb `k` holds decimal positions `2k` and
`2k+1` counted from the units digit. Compared with one byte per digit this halves memory traffic and measured about 2.5x
faster end to end at 8 threads. Little-endian is essential: growth happens at the high end, so limb boundaries never move
and a carry-out just writes one more limb — no front slack, no shifting offset.

**AVX-512 inner loop.** The reverse, the add, and the carry resolution are vectorized 64 lanes at a time. Reversal is a
`vpermb` byte shuffle plus a mod-10 lookup table (never a division). Carries use the standard generate/propagate trick
adapted to base 100. The loop meets in the middle, reading one vector from each end and writing both.

**Parallel segmented carry.** Above 98,304 digits the work is split across threads using a carry-select decomposition:
each thread resolves its segment assuming a carry-in of zero and reports whether it generated or would propagate a carry,
then a short serial scan threads the real carries through and fixes up only the segments that need it. Workers live in a
persistent spin-barrier pool, since at ~50µs per iteration thread creation would swamp the actual work. On multi-node
machines the threads are pinned and the buffer is first-touched in parallel, so each thread's pages live on its own NUMA
node.

**Checkpointing.** State is written to `.isf` files compatible with Istvan Bozsik's 'Istvan Standard Format', protected
by both a CRC-16 and a mod-9 consistency check from Ben Despres's [ISFTOOL](https://p196.org/files/mod9.zip) that
verifies the stored number could actually have arisen from the stored starting value in the stored number of iterations.
Checkpoints are taken on an iteration, digit, or time interval, on interrupt, on reaching a `--stop-*` limit, on running
out of memory, and on success — so a completed run leaves the palindrome itself on disk.

## Requirements

- Windows, x64
- A CPU with AVX512F, AVX512DQ, AVX512BW, AVX512VL, and **AVX512VBMI** — Intel Ice Lake / AMD Zen 4 or newer
- MSVC with C++20

VBMI is the one that's easy to overlook. Skylake-SP and Cascade Lake have AVX-512 but *not* VBMI, and the hot loop uses
`vpermb` and `vpermi2b` throughout. The program checks for every required extension at startup (including verifying the
OS actually preserves ZMM state) and exits with a clear message rather than faulting on an illegal instruction.

There is no scalar fallback. The entire binary is compiled with `/arch:AVX512`.

## Building

From an **x64 Native Tools Command Prompt for VS** (or any shell where you've run `vcvars64.bat`):

```
nmake
```

The binary lands in `bin\lychrel.exe`.

| Target | Effect |
| ------ | ------ |
| `nmake` | Build — `/O2 /arch:AVX512 /GL` with `/LTCG` at link |
| `nmake diag` | Build `bin\lychrel-diag.exe`, the instrumented binary (see [Diagnostic build](#diagnostic-build)) |
| `nmake run` | Build, then run against 196 |
| `nmake clean` | Delete build output |
| `nmake distclean` | Also delete `.isf` checkpoints in the repo root and `bin\` |

The x64 prompt is a hard requirement and the makefile enforces it, refusing to build if the toolchain targets anything
else. This is worth a guard rather than a note because the failure is silent: an x86 prompt compiles this source
without error — `/arch:AVX512` is accepted for 32-bit targets — and produces a binary that passes every correctness
check while running about 2.6x slower, because the packed core's 64-bit limb arithmetic gets split across register
pairs. Nothing about the output tells you, so it is easy to spend a long time tuning a build that was handicapped from
the start.

It's a single translation unit, so building by hand works too — but note that you are then on your own for the check
above, so confirm what you actually produced with `dumpbin /headers bin\lychrel.exe | findstr machine`:

```
cl /std:c++20 /O2 /arch:AVX512 /EHsc lychrel.cpp
```

`/GL` and `/LTCG` are worth about 2% here and are not essential. The usual reason for whole-program optimization —
inlining across translation units — does not apply to a single `.cpp`; the gain comes from link-time codegen giving
internal statics non-standard calling conventions and dropping more dead code.

## Usage

```
lychrel.exe version 1.0.0

lychrel.exe <start-number> | <save.isf> [options]

	<start-number>  decimal digits to start a fresh run from, e.g. 196
	<save.isf>      resume a previously saved run

Options:
	--status-iterations <n>  iterations between progress lines (default 100000, 0 disables)
	--status-digits <n>      digits of growth between progress lines (default 1000000, 0 disables)
	--status-minutes <n>     wall minutes between progress lines (default 0, disabled)
	--save-iterations <n>    iterations between autosaves (default 100000000, 0 disables)
	--save-digits <n>        digits of growth between autosaves (default 100000000, 0 disables)
	--save-minutes <n>       wall minutes between autosaves (default 1440, 0 disables)
	--stop-iteration <n>     stop on reaching this iteration (default 0, disabled)
	--stop-digits <n>        stop on reaching this many digits (default 0, disabled)
	--stop-minutes <n>       stop after this many wall minutes (default 0, disabled)
	--threads <n>            worker threads for the parallel core (default = physical cores, 1 forces serial)
	--no-crc-check           treat a save-file CRC mismatch as a warning instead of an error
	--block-size <n>         carry staging block size in limbs (default 8192, power of two)
	--parallel-threshold <n> digits before the parallel core engages (default 98304, 0 always)
	--no-pin                 disable NUMA thread pinning and parallel first-touch
```

The three cadences are independent and may be combined; whichever fires first wins. A value of `0` disables that
particular trigger.

Start a fresh search:

```
lychrel 196
```

Resume from a checkpoint:

```
lychrel 196_48316988_20000000.isf
```

Report every million iterations and checkpoint every 30 minutes:

```
lychrel 196 --status-iterations 1000000 --save-minutes 30
```

Run a bounded benchmark — 8 threads, stop after 10 minutes:

```
lychrel 196 --threads 8 --stop-minutes 10
```

### Output

Progress is a fixed-width table, so it stays aligned and sorts as text:

```
> lychrel 196 --status-iterations 50000
Starting 196 from iteration 0 (3 digits)
               Time     Iteration       Length
2026-08-26 21:14:02             0            3
2026-08-26 21:14:09         50000        20779
2026-08-26 21:14:21        100000        41482
```

A line is printed on the first iteration as a baseline, then on whichever status cadences are active, on every
checkpoint, and on exit. Lines are flushed explicitly, so redirecting to a file gives you progress as it happens rather
than in block-buffered bursts hours apart.

Progress goes to stdout; diagnostics go to stderr, so `lychrel 196 > run.log` captures the table alone and
`2> diag.log` captures the diagnostics alone. The default binary uses stderr for real errors only — a normal run,
bounded or not, leaves it empty.

The diagnostic build (`nmake diag`, see [Building](#building)) opens stderr with a summary of the settings actually in
effect:

```
> lychrel-diag 196
[config] 64-bit  threads=8  block-size=8192  parallel-threshold=98304  numa-nodes=1  pinning=inactive (1 NUMA node)
```

Worth checking before any measurement — which is why it lives in the diagnostic binary rather than the shipping one.
`64-bit` catches a 32-bit build, which is correct but roughly half the speed and otherwise indistinguishable in the
output — the Makefile refuses to produce one, but a stray binary can still be lying around. `pinning` reports its
a single-node machine and on any topology the OS does not report, which is the case where pinning is silently doing
nothing. See the tuning walkthrough for why that matters on a multi-node part.

Tuning warnings are also diagnostic-build-only, and appear on any run that trips them:

```
> lychrel-diag 196 --threads 8
WARNING: --threads 8 reduced to 3 until 131072 digits: at 49152 digits a block of 8192 limbs leaves only 3 whole
blocks per half, and a thread cannot be given less than one block. A growing run needs no action and regains all 8
threads at that length; lower --block-size only to widen the split at a fixed length below it.
```

These concern conditions that bind at a fixed length and clear themselves as a number grows (the thread-fit cap in the
`--block-size` section is the current example). The warning names the length at which it clears, so a growing run can
read that figure and move on — the reduction is a band the number passes through, not a setting to correct. Only a run
pinned below that length has anything to act on.

On exit the diagnostic build reports where an iteration's cycles went, averaged over every parallel dispatch:

- **`[span]`** — per-thread mean span, each thread's deviation from the slowest, and how often each was the slowest.
  The summary line also gives idle percentage, i.e. how much thread-time was spent waiting at the barrier.
- **`[ser]`** — the iteration split into dispatch and serial work, dispatch further split into the critical-path span
  and pool overhead, and the serial part itself broken into tail, high scan, carry apply and high apply. It ends with
  the Amdahl ceiling: the best speedup any further threading work could buy against the current fixed cost.
- **`[var]`** — why the spans differ. It compares each span's wall time against the cycles the thread was actually
  scheduled on a core, so preemption is separated from memory stalls, and prints the spread between the fastest and
  slowest span plus a histogram of how far the slowest sat above the mean.

Tune on `[ser]` and on `k` from the harness; `[span]` and `[var]` explain a result rather than deciding one. In
particular a high idle percentage is not by itself a problem — see the `--block-size` walkthrough, where lowering the
block size tightens the barrier without buying any wall time.

### While running

- **Space** — pause or resume. The run stays in memory while paused.
- **Ctrl+C** — save the current state and exit.

Closing the window, logging off, and shutting down also save before exiting. Because console control handlers run on a
thread Windows injects, the handler never touches the digit buffer itself — a save taken from there could catch the
number mid-reverse-add. Instead it raises a flag and blocks until the main loop reaches a safe point, saves, and signals
completion.

The loop checks that flag on every iteration, so a request is picked up at the next boundary between reverse-adds rather
than waiting for the periodic control poll. Ctrl+C and Ctrl+Break then let the save run to completion, since neither
starts an OS kill timer. Closing the window, logging off and shutting down *do* start one, so those waits stay bounded
and give up rather than be killed part-way through a write. The bound is a conservative few seconds rather than a figure
read from anywhere: the deadline differs between the close and logoff/shutdown paths, is registry-tunable, and has
changed across Windows versions, so the code aims to stay well inside the smallest of them instead of matching one.

Overrunning it is survivable regardless. Checkpoints are written to a temporary file and renamed into place once the
last byte is down, so a save cut short by the process being killed leaves the previous checkpoint intact — never a
half-written one.

At extreme lengths there is a limit worth knowing about. A safe point only exists between reverse-adds, so detection can
never be faster than a single iteration, and the checkpoint has to be written inside the same budget. Past a few hundred
million digits one iteration plus a multi-hundred-megabyte write no longer fits in the shutdown window, and it is the
periodic autosave rather than the shutdown handler that protects the run.

Running out of memory is treated the same way as an interrupt: the state is saved rather than letting `bad_alloc` escape
and lose the trajectory.

### Save files

Checkpoints are written to `<start>_<iteration>_<digits>.isf` in the current directory, and any of them can be passed
back in to resume. The two counts are zero-padded to 10 digits so a directory listing sorts chronologically. The
iteration is the number of reverse-adds completed, so a run that finds a palindrome writes the result under the step
count it took to get there:

```
> lychrel 89
Starting 89 from iteration 0 (2 digits)
               Time     Iteration       Length
2026-08-26 21:31:20             0            2
2026-08-26 21:31:20            24           13
```

leaves `89_0000000024_0000000013.isf` holding 8813200023188.

The palindrome test runs before each add rather than after, so an input that is already a palindrome stops immediately
at iteration 0.

### Tuning

`--threads` defaults to the machine's **physical** core count, not its logical processor count. Past the parallel
threshold the loop is DRAM-bandwidth bound, so SMT siblings mostly contend for a shared L1d rather than adding
throughput, and the pool spins — extra threads burn cores instead of waiting. Physical cores are only a proxy: the real
knee is set by the memory subsystem and on a bandwidth-saturated part may be *fewer* than the core count, so sweep it
(step 3 below) rather than trusting the default. The parallel core only engages past 98,304 digits; below that the
split costs more than it saves and the serial path is used regardless. The core also reduces its own thread count when a
length is too short to give every thread a whole block, so asking for more threads than a length can use is harmless
rather than an error. `lychrel --help` prints the detected default.

`--block-size` sets how many limbs are staged in L1 before being normalized. 8192 measured best on a 48 KB L1d Xeon,
beating 4096 by 3.5–8% and 2048 by 14%; parts with a smaller L1d (Zen 4 has 32 KB) may prefer 4096.

Sweep it at a length past last-level cache, but do not assume the effect is invisible below that. On the same Xeon with
a fully cache-resident ~0.7 MB working set, 4096 and 8192 still tied while 2048 lost 8% and 16384 lost 12% — the knob
sizes a per-thread buffer against L1d, so it bites long before the working set stops being LLC-resident.

The optimum does not move with thread count, so there is no need to re-sweep it per `--threads` value: 8192 won at 2, 4
and 8 threads, and at 1 thread the whole 2048–32768 range was flat to within noise. What thread count changes is how
much a wrong value costs — an oversized 32768 gave up 12% at 2 threads, 31% at 4 and 45% at 8. It is tempting to read
that as the per-thread buffers colliding in shared cache, but that model was tested and is wrong: holding
`threads × block-size` constant does *not* hold throughput constant (8×8192 is optimal while 4×16384 and 2×32768 each
lose 12%). What matters is the absolute per-thread block against one core's private cache; more threads merely amplify
the penalty, as you would expect once enough cores are streaming to sit near the bandwidth ceiling.

One caveat applies in a narrow band just above the parallel threshold. Spans are whole blocks, so a length supports only
`(digits/2)/block-size` threads: at 98,304 digits an 8192-limb block leaves 3 whole blocks per half, and a requested pool
of 8 is reduced to 3. The diagnostic build prints a warning to stderr when this binds (see [Output](#output)).
Measured at that fixed length, dropping to 4096 or 2048
recovers the lost threads and is worth 1.21x — but the condition is transient, since a growing run needs only
`threads × block-size × 2` digits (131,072 for 8×8192) to regain the full pool and never binds again. End-to-end the gain
is therefore much smaller and shrinks with run length: 1.06x growing to 393,216 digits, 1.03x growing to 1.5M, and
negligible beyond. Since 4096 and 2048 are *worse* than 8192 at the longer lengths a real run spends its time in,
lowering the block size to clear this warning is a net loss for anything but a fixed-length benchmark.

`--parallel-threshold` sets the digit count at which the parallel core takes over from the serial one. The default of
98,304 was measured on an 8-core machine, where the parallel path breaks even between 81,920 (0.97x) and 98,304
(1.12x), then climbs steadily — 1.30x at 131,072, 1.81x at 196,608, 2.75x at 524,288. The crossover depends on core
count and cache, so it is worth re-measuring on new hardware.

It does not, however, depend on thread count: 2, 4 and 8 threads all break even at the same length (0.85x at 65,536,
level at 81,920, first win at 98,304), so one sweep at your usual thread count settles it. What thread count changes is
the payoff past the crossover, not its position — 98,304 is worth 1.13x at 4 and 8 threads but only 1.05x at 2, and at
2 threads the curve is not even monotonic (131,072 measures slightly below 98,304). With only two cores the parallel
core is barely worth engaging at any length near the threshold; the gains here are a 4-plus-core effect.

Pass `0` to force the parallel core on at every length: below the threshold `--threads` is otherwise inert, so `0` is
what makes short lengths measurable at all. Sweep it by comparing a forced-serial run against a forced-parallel one at
each length:

```powershell
foreach ($d in 65536,81920,98304,131072,196608,262144) {
  .\tools\measure.ps1 -isf "probe\t$d.isf" -extra "--parallel-threshold 999999999" # serial
  .\tools\measure.ps1 -isf "probe\t$d.isf" -extra "--parallel-threshold 0"         # parallel
}
```

Take the shortest length where the parallel `k` beats the serial one. Interleave the two arms and use the best of
several repeats. Absolute `k` is not reliably comparable across sessions so always re-measure both arms together and
compare their ratio. A figure from an earlier session is not a baseline.

When comparing two builds of the current source — before and after a change — run both through `measure.ps1` at the
same length, back to back, and compare their `k`. Check that the digit counts match exactly at equal iteration
numbers: if they diverge the two builds are not computing the same trajectory and the timing means nothing.

On multi-socket or multi-NUMA-domain machines (notably EPYC in NPS2/NPS4), threads are pinned to a node and the digit
buffer is zeroed in parallel so each page is faulted by the thread that will use it. Windows places a page on the node
of whichever thread first touches it, so this is what keeps a thread's memory accesses local. Both behaviors are
automatic and become no-ops on single-node systems; `--no-pin` disables them, which is mainly useful for measuring what
they're worth. On a single-node machine the two arms land within about 1% of each other, which is the expected result
rather than a sign that pinning failed. Tune `--block-size` *after* confirming placement is right, or the sweep measures
remote-memory latency instead of L1 residency.

### Tuning walkthrough (EPYC / multi-node)

Work through these in order. Each step assumes the previous one is settled, because measuring a later knob while an
earlier one is wrong gives you a number that describes the wrong bottleneck.

**0. Measure at a realistic length.** This is the step that most often invalidates everything after it. A fresh
`lychrel 196` crosses the 98,304-digit parallel threshold in about 3 seconds, so a short run spends nearly all of its
time single-threaded on a cache-resident working set — none of what follows is visible there.

**10M digits is enough.** What matters is being past last-level cache and into the DRAM-bound regime, and that happens
well before 10M.

Parallel speedup climbs steeply up to 10M and then flattens — 20M buys only a few percent more than 10M for twice the
memory and twice the time to get there. Normalized cost per digit per iteration is likewise flat from 5M to 20M,
confirming the loop is already memory-bound at 10M. Block-size differences are still clearly resolvable there too,
which is what step 4 needs.

Below about 5M the numbers actively mislead: the working set is still partly cache-resident, so parallel speedup is
understated and the block-size optimum shifts. **Use 10M as the default tuning length; drop to 5M only if memory is
tight, and go to 20M+ only to confirm a result you already have.**

Build the checkpoint once and reuse it for every run:

```
lychrel 196 --stop-digits 10000000 --save-digits 0
```

That takes roughly 20 minutes on a high-bandwidth machine, and leaves `196_0024159531_0010000000.isf` (the trajectory
grows about 0.4139 digits per iteration, so 10M digits lands at iteration 24,159,531). It is a one-time cost — every
timing run below resumes from that file, so all runs start at the same length and do the same work:

```
lychrel 196_0024159531_0010000000.isf --status-iterations 0 --save-digits 0 --stop-iteration 24199531
```

That is 40,000 iterations past the checkpoint, which lasts a few seconds at the physical-core default. Scale it so a
run lasts several seconds — long enough to swamp process startup, short enough to sweep repeatedly. A run of a second
or less is mostly measuring process startup, not the loop.

Fix the *work*, not the elapsed time: give every run the same `--stop-iteration`, set a fixed distance past the
checkpoint's own iteration, and compare how long each takes. A fixed `--stop-minutes` would instead let each run do a
different amount of work, which is exactly what you are trying to measure.

**1. Confirm the topology.** If Windows reports one node, every NUMA path in the program is inert and you can skip to
step 3:

```powershell
(Get-CimInstance Win32_PerfRawData_PerfOS_NUMANodeMemory |
    Where-Object { $_.Name -ne '_Total' } | Measure-Object).Count
```

(The `_Total` instance has to be filtered out or a single-node machine reports 2.)

On EPYC this is a firmware setting (NPS1/NPS2/NPS4 in the BIOS). NPS1 presents the whole socket as one node and hides
the internal topology from the OS, which means the program cannot place pages usefully even though the underlying
memory is still physically partitioned. NPS2 or NPS4 exposes it and is what you want here.

**2. Measure what placement is worth.** Same checkpoint, same iteration count, pinning on versus off:

```
lychrel 196_0024159531_0010000000.isf --stop-iteration 24199531
lychrel 196_0024159531_0010000000.isf --stop-iteration 24199531 --no-pin
```

Omitting `--threads` uses the detected physical-core default, which is the right basis for this comparison; set it
explicitly only if you have already found a better value in step 3.

With `--no-pin`, one thread faults the entire buffer, so every page sits on one node and most accesses are remote.
Expect the pinned run to win by a wide margin on NPS2/NPS4 and by nothing at all on NPS1. If
pinning is *slower*, the likely cause is that another process is already confining this one — check for an outer
affinity mask or a job object before concluding the feature is at fault.

**3. Sweep `--threads`.** Past the parallel threshold this is bandwidth-bound, not compute-bound, so it stops scaling
once the memory controllers saturate — usually well before every core is busy. Sweep in powers of two up to the core
count and keep the knee, not the maximum:

```powershell
foreach ($t in 2,4,8,16) {
  $sw = [Diagnostics.Stopwatch]::StartNew()
  .\lychrel.exe 196_0024159531_0010000000.isf --threads $t --status-iterations 0 --save-digits 0 --stop-iteration 24199531 | Out-Null
  $sw.Stop(); "threads=$t  $([math]::Round($sw.Elapsed.TotalSeconds,1))s"
}
```

Note that `--threads` is capped at 64. SMT siblings share a core's L1, which is the resource `--block-size` is tuned
against, so two threads per core is the one case where the next step's optimum could plausibly come out smaller —
plain thread count does not move it.

If every thread count comes back with the same time, you are almost certainly still below the parallel threshold and
measuring the serial path — go back to step 0 and use a longer checkpoint.

**4. Sweep `--block-size`, last.** Only now does this measure what it claims to. Same harness, holding the thread count
you just chose:

```powershell
foreach ($b in 2048,4096,8192,16384) { ... --threads <chosen> --block-size $b ... }
```

Zen 4 and Zen 5 have a 32 KB L1d against the 48 KB of the Xeon the 8192 default was chosen on, so 4096 is a reasonable
bet — but measure rather than assume, since the stage buffer shares L1 with the streaming reads.

You only need this sweep once, at the thread count from step 3 — the optimum is a property of one core's private cache
and does not shift with thread count. Do not skip it on the grounds that step 3 already found a good time, though: the
cost of an oversized block grows with the thread count, so the value that looked harmless in a single-threaded check
is the one that costs you 45% at 8 threads.

**5. Re-check correctness.** None of these knobs may change results. Two runs from the same checkpoint to the same
iteration, differing only in tuning, must produce byte-identical saves:

```powershell
Get-FileHash *.isf -Algorithm SHA256
```

A mismatch is a bug, not a tuning outcome — please report it.

## Exit codes

| Code | Meaning |
| ---- | ------- |
| 0 | Palindrome found (or `--help`) |
| 2 | Interrupted — Ctrl+C, close, logoff, or shutdown; state saved |
| 3 | Could not load the requested save file |
| 4 | Out of memory during buffer growth; state saved |
| 5 | Required CPU extensions missing |
| 6 | Malformed command line |
| 7 | A `--stop-*` limit was reached; state saved |

`1` is deliberately unused. The CRT and the OS both produce it for abnormal termination, so leaving it free keeps "the
program decided to stop" distinct from "the program died."

The search is unbounded and runs until a palindrome appears, you interrupt it, a `--stop-*` limit is hit, or the machine
runs out of memory.

## License

MIT — see [LICENSE](LICENSE).

Copyright (c) 2026 Vinny Romano
