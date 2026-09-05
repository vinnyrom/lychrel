# Times the lychrel.exe at a fixed length by stamping the arrival of each progress line with a high-resolution clock,
# then reports seconds per iteration and the normalized cost k = seconds / digit / iteration.
#
# Any .isf works as input. Real checkpoints are written to the working directory the run was launched from; synthetic
# fixed-length probes can be regenerated at any size with bin\mkisf.exe <digits> <iter> <out>:
#
#   .\bin\mkisf.exe 400000000 1 probe\p400M.isf
#   .\tools\measure.ps1 -isf probe\p400M.isf -status 100 -samples 4
#
# -status is a number of ITERATIONS between progress lines, not seconds; it is passed straight through to
# --status-iterations. Size it so each interval lands around half a second: long enough to swamp timer noise, short
# enough that -samples lines arrive promptly. Indicative ms/iteration on the reference machine (an Azure
# Standard_D16ds_v5: 16 vCPUs on 8 physical Xeon Platinum 8370C cores, Ice Lake-SP, 64 GB):
#
#     2M digits  0.025 ms      25M digits  0.22 ms      100M digits  1.2 ms      400M digits  6.5 ms
#
# so -status 20000 suits 2M, 1600 suits 25M, 400 suits 100M, and 100 suits 400M. Cost per iteration is not quite
# linear in length: normalized k = s/digit/iteration dips while the working set is cache-resident and rises once the
# loop becomes DRAM-bound.
#
# SANITY-CHECK THE GB/s COLUMN BEFORE USING A RESULT. Too short a -status collapses the interval into timer
# granularity and the arithmetic then reports nonsense rather than failing - k=1.45e-13 at 90,646 GB/s is the kind of
# figure it produces. Nothing here can move data at those rates, which is what gives the error away. A plausible
# figure is tens of GB/s; anything in the thousands means the interval was too short, so raise -status until the
# column is believable and only then read k.
#
# TREAT THE ABSOLUTE NUMBERS ABOVE AS INDICATIVE ONLY, in that they are specific to the machine and the length. They
# are, however, stable across sessions for a fixed binary, and repeatable to within ~10% within one. Architecture is
# what moves them: a 32-bit build measures ~3.6x slower than a 64-bit one on the same workload. So if a k or GB/s
# figure has moved from one you recorded earlier by more than that spread, suspect the binary - -diag prints the
# architecture in its [config] banner - before suspecting the machine. Comparing ratios measured in one session is
# still the more robust habit.
#
# -diag switches to bin\lychrel-diag.exe, which is where the [config] banner and the tuning warnings live. Use it
# to check that the settings, architecture and pinning are what you think before a measurement round, and to read the
# [span]/[ser]/[var] breakdown. Do NOT compare its k or GB/s against a release figure: the diag build carries rdtsc
# and QueryThreadCycleTime calls inside the per-dispatch path, so its timings are instrumented and only comparable
# against other diag runs. The default remains the release binary so the common case stays honest.
#
# Either way the binary's stderr is echoed below the table rather than discarded, so a warning raised mid-run is not
# lost behind the redirected stdout the harness consumes.
param(
	[Parameter(Mandatory = $true)][string]$isf,
	[int]$status = 20000,
	[int]$samples = 4,
	[string]$exe = "",
	[int]$affinity = 0,
	[string]$extra = "",
	[switch]$diag
)

if ($exe -eq "") { $exe = if ($diag) { "bin\lychrel-diag.exe" } else { "bin\lychrel.exe" } }

# Bytes of DRAM traffic per limb per iteration, used only for the GB/s column. The parallel core touches each limb
# twice: its sweep reads every limb and writes every limb (the mirrored high half is produced in that same sweep and
# normalized out of an L1-resident stage buffer). The per-block carry fixup that follows re-touches only the blocks
# that actually take a carry, which is a vanishingly small fraction, so it is not modelled here.
#
# Revisit if the number of full sweeps over the buffer ever changes, or this column will silently misreport bandwidth.
# Note the serial path below the parallel threshold does make a second sweep, so this understates its traffic.
$BytesPerLimb = 2.0

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = (Resolve-Path $exe).Path
$psi.Arguments = "`"$((Resolve-Path $isf).Path)`" --status-iterations $status --save-digits 0 $extra"
$psi.WorkingDirectory = (Get-Location).Path
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true

$proc = [System.Diagnostics.Process]::Start($psi)
# Drained on a background handler rather than read inline: stderr is not read until after the sampling loop, and a
# blocked stderr pipe would eventually stall the run itself.
$errLines = New-Object System.Collections.Generic.List[string]
$null = Register-ObjectEvent -InputObject $proc -EventName ErrorDataReceived -Action {
	if ($EventArgs.Data) { $Event.MessageData.Add($EventArgs.Data) }
} -MessageData $errLines
$proc.BeginErrorReadLine()
$proc.PriorityClass = 'High'
if ($affinity -ne 0) { $proc.ProcessorAffinity = [IntPtr]$affinity }

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$rows = @()
$prev = $null

while (-not $proc.HasExited -and $rows.Count -lt ($samples + 2)) {
	$line = $proc.StandardOutput.ReadLine()
	if ($null -eq $line) { break }
	$t = $sw.Elapsed.TotalSeconds
	# "YYYY-MM-DD HH:MM:SS      <iteration>      <length>"
	if ($line -match '^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\s+(\d+)\s+(\d+)\s*$') {
		$rec = [pscustomobject]@{ T = $t; Iter = [uint64]$Matches[1]; Len = [uint64]$Matches[2] }
		if ($null -ne $prev) {
			$di = [double]($rec.Iter - $prev.Iter)
			$dt = $rec.T - $prev.T
			if ($di -gt 0) {
				$rows += [pscustomobject]@{
					Digits      = $rec.Len
					Iters       = $di
					Seconds     = [math]::Round($dt, 4)
					SecPerIter  = $dt / $di
					k           = $dt / $di / [double]$rec.Len
					GBps        = ([double]$rec.Len / 2.0) * $BytesPerLimb * $di / $dt / 1e9
				}
			}
		}
		$prev = $rec
	}
}

if (-not $proc.HasExited) { $proc.Kill() ; $proc.WaitForExit() }

# The first interval includes buffer touch / page-fault warmup, so it is reported but excluded from the summary.
$rows | Format-Table -AutoSize @{n='Digits';e={$_.Digits}}, Iters, Seconds,
	@{n='ms/iter';e={"{0:N4}" -f ($_.SecPerIter*1000)}},
	@{n='k (s/digit/iter)';e={"{0:E3}" -f $_.k}},
	@{n='GB/s';e={"{0:N1}" -f $_.GBps}}

$steady = $rows | Select-Object -Skip 1
if ($steady) {
	$km = ($steady | Measure-Object -Property k -Average).Average
	$bm = ($steady | Measure-Object -Property GBps -Average).Average
	"STEADY  digits={0}  k={1:E3}  GB/s={2:N1}" -f $rows[-1].Digits, $km, $bm
}

Start-Sleep -Milliseconds 200
Get-EventSubscriber | Where-Object { $_.SourceObject -eq $proc } | Unregister-Event
if ($errLines.Count -gt 0) {
	""
	"--- stderr from $exe ---"
	$errLines | ForEach-Object { $_ }
}
elseif ($diag) {
	""
	"NOTE: -diag produced no stderr. The [span]/[ser]/[var] report is emitted every 100000 dispatches, so a short"
	"      run can end before the first one. Raise -samples or -status to run longer."
}
