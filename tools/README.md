# tools/ - host-side utilities

Note: AI generated text below.

Everything executable lives here, C and Python. Besides `insrcv.c` and
`replay.c` that is `replay.py` (dataset replay with plots, KML and telemetry),
`inspostgui.py` (post-processing GUI), `replay_core.py` (the replay loop
those two share, so they cannot drift apart), `allan_variance.py`,
`crazyflie_reader.py` and `ublox_f9p_config.py` / `ublox_x20p_config.py`; they
are documented in `python/README.md` and in their own `--help`. The importable
package they are built on is `python/INSLIB/`.

The tools described below are hardware independent: the live receiver, the calibration
maths and the command line calibration, and the protocol library. The tools
that talk to the reference board itself (the serial port hub, its control
GUI, its configuration interface, the guided calibration window) are part of
the reference board delivery and not of this repository. What they speak is
documented in `inslib_protocol.md`, so a board of your own can feed `insrcv`
the same way.

## The shape of it

A serial port has exactly one owner. That single fact decides how these
tools fit together: a receiver that opens the device makes recording the
same session impossible, and two recorders cannot coexist at all.

So one tool owns the device and everything else speaks UDP.

```
 corrections (str2str) ──UDP:29797─►┐────────────────────.
                                    │                    ├──► inslib_<stamp>.ubx
 odometry ─────────────UDP:29798───►│   inslib_hub.py    ├──► _timesync.csv, _speed.csv
 (inslib_obd_speed.py)              │  owns the COM port │
                                    │                    ├──UDP:29800──► insrcv ──► PlotJuggler
                                    │                    │                     └──► MAVLink (GCS)
                       serial ◄────►│    e.g. COM4       │
                       (corr out)   └────────────────────┘
                       (UBX in)
```

`inslib_replay_udp.py` puts a recorded `.ubx` on the same UDP port at the
original pace, so `insrcv` cannot tell a replay from a live session.

## insrcv.c - live receiver (C)

Binds a UDP port, drives `nav_suite`, publishes to PlotJuggler over
UDP/JSON and, on request, to MAVLink. Also the reference for porting the
host-side glue onto an embedded target.

```sh
make insrcv
./build/insrcv                                   # listen on 0.0.0.0:29800
./build/insrcv --udp-port 29800 --udp-bind 127.0.0.1
./build/insrcv --raw-only                        # raw sensors, no filter
./build/insrcv --config config.yaml              # noise model + IMU calibration
./build/insrcv --mavlink                         # + MAVLink 2 on udp:14550
./build/insrcv --help                            # all options
```

Then point PlotJuggler at UDP/JSON on 127.0.0.1:9870 (the default).

It never opens a serial port. Recording and live processing have to be
possible at the same time, and one owner per port means the receiver
cannot be that owner. It consumes the protocol in `inslib_protocol.md`
and nothing else.

**Status line.** Repainted once a second (`--stats-sec`), on one line,
built on the same rule as the hub's -- rates and current state live,
totals in the closing summary:

```
  632s | imu 804 baro 25.0 gnss 4.0 odo  4.0@13.9m/s ppsT | |a|9.88 gz+12 alt43.0 | 3D 24sv FIX 0.15/0.24m 0.05m/s ZUPT
```

Sensor rates are in Hz and counted over the interval, so a stream that
stopped reads `0` (the IMU rate is deliberately *not* the device-clock
rate, which would freeze at its last value on a dead link). Then the
receiver's fix: type, satellites, RTK carrier solution (`float` / `FIX`)
and the hAcc/vAcc/sAcc that gate GNSS fusion (position accuracy in
metres, then speed accuracy) -- omitted without a position fix, where
they are the receiver's "invalid" sentinel rather than a number. `|a|`, `alt` and `|m|` are the sensor sanity checks; `gz` appears
once the unit is actually turning.

Counters only appear once they are not zero -- `drop`, `gnss-off`,
`baro-off`, `odo-stale`, `decl`, `crc`/`resync`/`badlen` -- so anything
visible there is worth reading. Session totals (samples per sensor, the
measured device-clock IMU rate, framing faults) are printed once on
Ctrl-C.

**Magnetometer.** `0x40/0x06` frames are decoded, counted and published
always, and fused only when `mag: enable` says so. It is the one sensor
here that is opt-in, because its raw output is not a heading until it has
been calibrated against the platform it is bolted into: an uncalibrated
hard iron of 15 µT against a 48 µT field is tens of degrees of yaw.
`mag: misalignment` / `fixed_bias` (soft iron, hard iron and the
alignment onto the IMU, all measured by `inslib_calib_gui.py`),
`mag: stddev_ut` and `mag: estimate_bias` are read from the same
`config.yaml` the replay harnesses use.

The stats line and the raw overlay both carry the magnetometer's own
sanity check, the exact counterpart of `|a|` against 9.81:

```
|m|48/49
```

left is the calibrated magnitude, right is what the World Magnetic Model
expects at the receiver's position. A calibrated magnetometer reads one
number whichever way the unit is turned, so a left-hand number that moves
as the platform turns is a calibration that is not done (in PlotJuggler:
`sensor/mag/norm_ut`, `norm_raw_ut` and `expected_ut`).

The stats line also carries `hdgmag123`, a tilt-compensated heading off
the magnetometer, after the `mag:` calibration (hard/soft iron) and so
the direction the filter is handed. It is relative to magnetic north, no
declination applied, so against a phone compass set to true north it is
off by the local declination. Levelled with the latest IMU sample rather
than ins/ahrs, so it is there from the first magnetometer frame, before
either filter has initialised. The raw overlay carries the
calibrated/uncalibrated pair the same way it does for magnitude:
`sensor/mag/heading_deg` and `heading_raw_deg`.

**The magnetic model needs an epoch**, and takes the first it can get:
`mag: wmm_year` if the config states one, else GPS time from the
`0x40/0x05` pulses, else the UTC date NAV-PVT carries once the receiver
has decoded one (`validDate`), else the host clock. The last one is why a
session indoors, with no fix, no time pulse and no dated epoch, still gets
declination instead of silently referencing magnetic north. Which source
was used is in the log line:

```
[insrcv] magnetic model at 48.137 11.575 (from GNSS fixType 3), epoch 2026.61 (from host clock)
```

A better source arriving later re-applies the model right away rather than
waiting for the 25 km refresh, and logs the line again -- a receiver needs
tens of seconds to decode the date, and without that the whole session
would stay on whatever the host clock said.

**Odometry.** `0x40/0x80` frames are fused as `ins`'s absolute-speed
aiding (REQ-NAV-068). A sample's age at its epoch is derived from its own
`t_us`, so the residual is anchored that far back in the filter's state
history. A small *negative* age is normal and is fused at delay 0: the
odometry delay is measured against the host clock while the IMU stream
arrives with a transport latency of its own, so the effective age lands
near zero with jitter on both sides. The `REVERSE` flag is decoded,
published and counted but **not** acted on -- see `inslib_protocol.md`.

### MAVLink output (`--mavlink`)

Off by default, because nothing listens on `udp:14550` unless you started
something. With it on, any ground station or log analyser pointed at that
port gets the message set `tools/replay.py --mavlink` produces, at the
same rates, plus `GPS_RAW_INT`:

| message | rate | carries |
|---|---|---|
| `HEARTBEAT` | 1 Hz | so a GCS keeps the system on screen |
| `ATTITUDE`, `ATTITUDE_QUATERNION` | 60 Hz | the arbitrated attitude, both spellings |
| `LOCAL_POSITION_NED` | 60 Hz | position and velocity in the filter's own NED frame |
| `GLOBAL_POSITION_INT` | 10 Hz | lat/lon/height |
| `GPS_RAW_INT` | 5 Hz | the receiver's own fix: satellite count, fix type, accuracies |
| `ALTITUDE` | 20 Hz | the vertical channel on its own |
| `HIGHRES_IMU` | 50 Hz | the nav-frame acceleration and the body rates |
| `EKF_STATUS_REPORT` | 10 Hz | real flags and real variances from the covariance |
| `NAMED_VALUE_FLOAT` | 5 Hz | the sub-filter breakdown and its 1-sigma |
| `STATUSTEXT` | on change | the blocked reason, in words |

```sh
./build/insrcv --mavlink                              # udp:14550 on this host
./build/insrcv --mavlink --mav-ip 192.168.1.20        # a GCS on another machine
./build/insrcv --mavlink --mav-subfilter-hz 1         # thin out NAMED_VALUE_FLOAT
./build/insrcv --mavlink --mav-pose-hz 30             # attitude and local position slower
```

`--mav-pose-hz` sets `ATTITUDE`, `ATTITUDE_QUATERNION` and
`LOCAL_POSITION_NED` (default 60 Hz) and lifts `--pub-hz` to match when that
is lower, since MAVLink is sent from the publish round.

`ALTITUDE` is the one that makes a baro-only run visible: the two
position messages need a horizontal solution, so in `ATTITUDE_ONLY` mode
they carry nothing while `baro_alt` is producing a perfectly good height.

`GPS_RAW_INT` is the receiver, not the filter: it is the last NAV-PVT
epoch as it arrived, so `satellites_visible` (NAV-PVT `numSV`) and
`fix_type` stay visible even while the filter refuses every fix, and they
keep flowing with `gnss.enable: 0`. `fix_type` is the u-blox `fixType`
mapped onto `MAV_GPS_FIX_TYPE`, with `carrSoln` folded in, so an RTK
float/fixed solution shows as one. An epoch with `gnssFixOK` clear goes
out as `NO_FIX` -- it carries a position, but not one anything should act
on, which is the verdict the filter's own gate reaches too. `fixType` 1
(dead reckoning only) and 5 (time only) have no MAVLink spelling and map
to `NO_FIX` as well. `eph`/`epv` are `hAcc`/`vAcc` in cm, saturating to
65535 ("unknown") where the receiver reports its own "no idea".

Whether the filter is actually **fusing** GNSS is a different question,
and it rides in the `NAMED_VALUE_FLOAT` block: `gnss_used` is 1 while
`ins` has accepted a fix within the last 3 s and 0 otherwise, so a run
whose fixes all fail the accuracy gate shows `GPS_RAW_INT.fix_type` 3 and
`gnss_used` 0. `gnss_nsat` repeats the satellite count there so it is
plottable without a `GPS_RAW_INT` decode. Both are live-only: the CSV
replay path carries no satellite count at all, which is why
`tools/replay.py --mavlink` has neither them nor `GPS_RAW_INT`.

Two things a consumer should know. `GLOBAL_POSITION_INT.alt`,
`GPS_RAW_INT.alt` and `ALTITUDE.altitude_amsl` are declared as mean sea
level but carry the **ellipsoidal** height, because there is no geoid
model in this library. And `EKF_STATUS_REPORT` is an *ardupilotmega*
message rather than a common one, so a decoder needs that dialect
(pymavlink's default has it).

The framing is `tools/mini_mavlink.h`, a MAVLink 2 encoder for exactly
these eleven messages. The generated MAVLink C library would be a
submodule and a build dependency for a fixed, small message set, and the
point of `insrcv.c` is to be portable host-side glue.

### Config file (`--config`)

`--config` reads a `config.yaml` subset shared with `tools/replay.py`.
The file is applied where it appears on the command line, later options
override it.

A key `insrcv` does not consume is **named on the console** and then
ignored -- a tuning value that silently has no effect is
indistinguishable from one that had no effect. It implements the filter
side of the shared schema in full: `imu:`, `gnss:` (including the
covariance conditioning), `baro: stddev_m` / `acc_bias_init_mps2`,
`init_stddev:`, `init_hint:`, `free_inertial_start:`, `mag:`, and the
top-level filter switches. Still replay-only, because this receiver has
no counterpart for them: `ahrs:` (its ARS/AHRS templates are derived from
`imu:`), the rest of `baro:`, `automotive_mode`, and the harness's own
bias/init windows. The keys that describe the replay harness rather than the filter
(`score:`, `inputs:`, and the `name`/`aiding`/`init` labels) pass
unremarked, so pointing `insrcv` at a dataset `config.yaml` does not
bury the real warnings. The count is repeated in the config summary line:

```
[insrcv] config: "imu.pos_pred_stddev_m_sqrts" not used by insrcv (typo, or a replay-only key)
[insrcv] config foo.yaml: gyr_psd=5e-07 acc_psd=1e-05 imu-calib=off, 1 key(s) ignored (see above)
```

### Starting the 3D filter without GNSS (`free_inertial_start`)

Normally the 3D solution waits for the first fix, because without one
there is no origin to navigate away from. This section lets you supply
that origin instead -- *"I am here, to within this much"* -- so the filter
comes up on the IMU alone and coasts on the auto-ZUPTs and the barometer:

```yaml
free_inertial_start:
  enable: 1
  lat_deg: 49.0069
  lon_deg: 8.4037
  height_m: 118.0
  stddev_m: 2.0     # how well you know that point, 1-sigma

gnss:
  enable: 0         # optional: exclude GNSS outright (see below)
  init_dwell_sec: 1.0
```

The position is **offered as a measurement only until the filter has
bootstrapped from it**, then the receiver goes quiet -- a source that kept
repeating the same point would pin the solution to it instead of dead
reckoning. The stats line counts the offers as `decl=N`, with a `*` while
they are still being made.

While the declaration stands, a GNSS fix **wider than it** is kept out of
the filter and counted as `decl=N(-M)`. Indoors the receiver keeps
producing fixes tens of metres wide, and such a fix does not just carry
less than the declaration -- it also **resets the 3D entry dwell** the
declaration is in the middle of earning, so letting it through means the
filter never starts at all. The first fix that is at least as good (judged
on NAV-COV, or NAV-PVT's `hAcc`) hands the job over for good:

```
free-inertial start: ignoring GNSS wider than the declared +-2.0 m while starting up
free-inertial start: 3D solution up, declared position withdrawn (3 offer(s))
   ...  decl=3(-1)
```

Those fixes are only held back *while starting up*. Once the filter is in
3D they reach it again and its own gates decide; a stream of wide fixes
will then keep it in `COASTING` rather than `FULL`.

**`enable: 1` is a developer mode and behaves like one: once the 3D
filter is up, it stays up.** Coasting without absolute aiding is not a
fault here, it *is* the mode, so both mechanisms that would otherwise end
the run are taken out of the loop and the receiver says so at startup:

- **`max_deadreckoning_sec` is ignored** (`allow_unlimited_deadreckoning`
  forced on). A budget would stop the filter mid-experiment for doing
  exactly what it was asked to do.
- **the 3D exit gate is off** (`gnss.stop_disable` forced on). A run of
  poor-but-valid fixes -- 20 m indoors, a canyon, a tunnel mouth -- makes
  `ins` conclude after `gnss.stop_dwell_sec` that its aiding degraded and
  re-arm (REQ-NAV-052), ending the run for no good reason.

What is *not* switched off: the 3D entry gate, and every fix is still
judged on its own accuracy before it may fuse. GNSS keeps its power to
**correct** the solution, it only loses its power to **end** it.

### Excluding GNSS outright (`gnss: enable: 0`)

Sometimes the point is to see what the IMU does *without* being corrected.
`gnss.enable: 0` keeps decoding, counting and publishing the receiver --
so the same recording stays comparable against a run with GNSS on -- but
nothing derived from it reaches the filter. Dropped fixes are counted as
`gnss-off=N` in the stats line.

That includes the magnetic model: with GNSS excluded it takes its anchor
from `free_inertial_start`'s declared position instead of from a fix. Its
*epoch* still comes from the receiver where the stream carries one -- the
`0x40/0x05` time pulse or NAV-PVT's UTC date -- because a date is a clock
reading and not a position measurement; `mag.wmm_year` and the host clock
remain the fallbacks.

### NAV-PVT fix quality: two consumers, two thresholds

`fixType` is read by two things that need very different amounts of it.

**The filter gets a 3D fix or nothing** (`fixType` 3, or 4 = GNSS+DR, and
`gnssFixOK` set). Everything else is counted as `nofix=N` in the stats
line and never reaches `ins`. Indoors an F9P keeps emitting NAV-PVT at the
full rate with `fixType 0`, `hAcc 0xFFFFFFFF` (4294967 m) and
`sAcc 999 m/s` -- that is the receiver saying *nothing*, not saying
*something imprecise* -- and a 2D fix has no height to give at all.
Handed to the filter as position measurements they used to trip the 3D
**exit** gate after `gnss.stop_dwell_sec`, so the filter concluded its
aiding had degraded and dropped out of 3D (REQ-NAV-052), ending a
dead-reckoning run that nothing had gone wrong with. The offline
converter has always filtered on `fixType` (`--min-fix-type`); this is
the live path catching up.

**The magnetic model takes any fix with a position** (`fixType` >= 2). It
only needs to know where on earth it is, to look up declination and the
NED reference field; those vary on the scale of degrees of latitude, so a
2D fix hundreds of metres out is worth exactly as much as an RTK one.
Refusing it would leave a magnetometer AHRS referencing *magnetic* north
-- a real heading error of a few degrees in Europe, far more further
north -- for no gain. The model is re-looked-up when the platform has
moved 25 km:

```
[insrcv] magnetic model at 49.007 8.404, epoch 2026.02 (fixType 2 is good enough for this)
```

Its epoch comes from the receiver too, because it knows the date better
than a config file does and a stale hand-typed year costs about 0.1 deg of
declination per year: GPS time from the stream (`0x40/0x05`) first, else
the UTC date in NAV-PVT itself, which is the only date a capture without
time pulses carries. `mag.wmm_year` overrides both for a session that gets
neither.

The blocked-reason line tells the two no-aiding cases apart:

```
3D filter off: no GNSS/position measurements at all             # nothing arrives - antenna? config?
3D filter off: the receiver has no 3D fix (NAV-PVT arrives without one)   # arrives, empty - go outside
```

Two limits are inherent, not incidental:

- **Heading is not observable.** Without GNSS course and without a
  magnetometer nothing measures yaw, and at standstill a MEMS gyro cannot
  find north either (earth rate is far below its bias stability). The
  track gets a shape, not a direction.
- **Horizontal position drifts** with nothing on board to correct it. The
  barometer holds the vertical channel and the ZUPTs hold velocity and the
  biases while the platform stands still -- that is what makes the run
  useful, and it is also all of it.

So `max_deadreckoning_sec` still applies: when the budget expires the
suite falls back to `ATTITUDE_ONLY`, and the first real fix re-acquires
normally. Combining this with `allow_unlimited_deadreckoning: 1` is
refused rather than silently resolved -- the two ask for opposite things.
Keep `stddev_m` at or below `gnss.start_max_horizontal_pos_stddev_m`,
otherwise the declaration cannot pass the 3D entry gate and the filter
never starts.

The same section works in `replay.c` and `replay.py` with identical
semantics, so a recording made this way replays the same way -- see the
design document's *Free-inertial mode* section.

### Building and debugging on Windows

The Makefile drives gcc, which typically Windows does not have on PATH.
Rather than failing with a bare "gcc: command not found" from inside a
recipe, `make` checks up front and prints how to put a MinGW-w64 / MSYS2
toolchain on PATH (many MinGW-w64 builds ship a `mingwvars.bat` /
`mingwvars.sh` for exactly that, MSYS2 has its "MinGW 64-bit" shell).

For VS Code: build tasks and the debugger inherit PATH when VS Code
starts, so add the gcc (MinGW) toolchain's `bin\` directory to your **user
PATH** once (Windows Settings > "Edit environment variables for your
account") and restart VS Code. Then:

- `.vscode/tasks.json` has **make insrcv**
- `.vscode/launch.json` has **Debug insrcv**, which runs that task first.
  Start `inslib_hub.py` (or `inslib_replay_udp.py`) alongside it to give
  it something to receive.
## inslib_speed_scale.py - speed.scale from GNSS

```sh
python inslib_speed_scale.py datasets/mydrive/csv
python inslib_speed_scale.py datasets/mydrive/csv --plot
python inslib_speed_scale.py --speed speed.csv --gnss gnss.csv --plot scale.png
```

config.yaml's own comment on `speed: scale` says what this does:
"Systematic and vehicle-specific, so calibrate it against GNSS". Reads a
dataset's `speed.csv` and `gnss.csv`, shifts each to its own true event
time using its `delay_ms` (odometry per-row, GNSS from `config.yaml`'s
`gnss: delay_ms`), and fits `gnss_speed = scale * odometry_speed` through
the origin - a weighted least squares plus a robust median-ratio
cross-check that a handful of outliers cannot move:

```
13213 samples used of 16332 odometry rows (81%), spanning 1566 s, odometry speed 3.1 - 38.6 m/s
gnss.delay_ms used: 200 ms (from config.yaml)

weighted LS through origin : scale = 1.0295 +/- 3.8e-05
robust median ratio        : scale = 1.0301 +/- 0.0048 (MAD-based)
speed RMS vs GNSS: 0.768 m/s raw -> 0.112 m/s scaled

config.yaml:
  speed:
    scale: 1.0295
```

Samples near a stop are dropped (`--min-speed-mps`, default 3): GNSS
Doppler noise and any driveline backlash dominate there and the scale is
not observable anyway. Samples where GNSS speed is changing fast are
dropped too (`--max-accel-mps2`, default 1.5): a timing error costs
nothing while speed is constant and grows with the rate of change, so
those are exactly the samples a leftover error in either `delay_ms` would
bias, not average out. `reverse=1` rows are dropped by default
(`--include-reverse` keeps them) - whether the vehicle scales the same way
running backwards is not something this script can tell.

The two reported numbers are a check on each other, not a choice: the
weighted fit is right if the noise model (odometry stddev plus propagated
GNSS velocity variance) is right, the median ratio is right regardless but
throws away information. Large disagreement between them means the fit is
leaning on a few samples - the script says so and points at `--plot`.

`--plot` (needs matplotlib) opens a window with a scatter of odometry vs.
GNSS speed and the fitted line, plus the two time series overlaid, so a
systematic bend (wrong model, not just a scale) or a stretch of bad
interpolation is visible instead of hiding inside the two numbers above.
`--plot FILE.png` also saves it.

## inslib_imu_calib.py - IMU calibration (no fixture)

Records one session in which the unit is set down in a number of
**arbitrary** static poses with rotations in between, then writes the
REQ-NAV-037 keys into a `config.yaml` that `insrcv --config` and
`tools/replay.py` consume directly.

```sh
python3 tools/inslib_imu_calib.py --port COM4
python3 tools/inslib_imu_calib.py --udp 29801        # via the hub fan-out
python3 tools/inslib_imu_calib.py --csv mysession/   # offline, any IMU
```

**No sensor board? Calibrate from CSV.** `--csv` runs the same solve and
the same config writer on a session recorded with any IMU, as
replay-format files (see `datasets/replay_format.py`):

```text
imu.csv   t_us, gyr_x, gyr_y, gyr_z [rad/s], acc_x, acc_y, acc_z [m/s^2][, temp_degC]
mag.csv   t_us, mag_x, mag_y, mag_z [uT]          (optional)
```

Body frame FRD, timestamps in integer microseconds, both files on the
same clock (the magnetometer is matched to the static poses by time, and
may run at a different rate). Lines starting with `#` are skipped. Point
`--csv` at a directory and its `imu.csv` and `mag.csv` are both used, or
at an `imu.csv` directly and name the magnetometer file with `--mag-csv`.
The recording has to follow the procedure below: the unit left alone for
the first `--init-sec` seconds, then set down in 20 or more different
attitudes, a few seconds each, with the rotations in between. `-y` writes
the result without asking.

```sh
python3 tools/inslib_imu_calib.py --csv mysession/ -o config.yaml
python3 tools/inslib_imu_calib.py --csv log/imu.csv --mag-csv log/mag.csv \
    --gravity 9.8093 --mag-field-ut 48.6 -y
```

`python/tests/test_calib_csv.py` runs exactly this path against a
synthetic session with known scale factors, misalignments and biases in
all three sensors.

**A board that corrects its own stream has to be switched off first**, or
the recording measures what is left over of the calibration it already
holds rather than the sensor:

On the reference board that is `CFG-IMU-APPLY_CAL=0` through its configuration
tool (live layer only). A board of your own needs the equivalent switch.

Nothing in the numbers says which it was, so the recording counts the per
sample `cal_applied` bit instead and says at the end if any of it arrived
corrected. Clearing the switch this way touches the live layer only: the
stored image stays as it is and the unit comes up correcting again at the
next power cycle.

**Why the poses can be arbitrary.** The method is IMU-TK's
(Tedaldi/Pretto/Menegatti, ICRA 2014; the numpy port is
`inslib_imu_tk.py`). Its accelerometer cost is

    residual = |g| - || M * (a_raw - bias) ||

a *magnitude*, so where a pose points never enters it. Hold the unit, put
it down somewhere, leave it a few seconds, pick it up, turn it, repeat.
The static periods are found from a rolling variance and the single
threshold involved is swept and resolved by residual, so there is nothing
to tune. What matters is that the poses point in *different* directions
and that the unit is really at rest in each.

This replaced a 6-position tumble test. That method estimates the bias as
`(up + down)/2` per axis pair, which is exact only when the two halves of
a pair are exactly opposite one another — an independent 1 deg placement
error on each position puts ~0.11 m/s² into the accelerometer bias, more
than the bias of a decent MEMS part. It also could not calibrate the
gyroscope beyond its bias, for want of a rate table.

**The gyroscope gets scale and full misalignment too**, from the same
recording and still without a turntable: between two static poses the
accelerometer knows both gravity directions, and the gyro integrated over
the rotation between them has to carry one onto the other.

Measured on synthetic data with known truth: accelerometer `M` to 1.2e-4
and bias to 7e-4 m/s², gyro `M` to 7e-5 and bias to 0.0007 deg/s. On
IMU-TK's own Xsens recording (`testdata/`): 38 poses found automatically,
|g| 1.1 mm/s² rms across all of them. Both are gated by
`python/tests/test_imu_tk.py`.

**What did it buy?** Both front ends print a before/after table over the
very samples the fit used, so the answer is not a matter of faith:

```
what the calibration bought (over the fitted samples):
                    |a| - |g| [m/s^2]                  gyro dir
                      rms        mean         max          rms
  raw                0.1060    -0.0359      0.2382       2.38 deg
  +bias              0.0713    -0.0454      0.1411       1.44 deg
  +scale/misalign    0.0101    -0.0000      0.0314       0.01 deg
```

The rows are cumulative, so reading down the column separates what the
zero offset fixed from what the scale and misalignment added.

**`--misalignment`** (checkbox in the GUI, **off by default**) turns the
axis misalignment on; without it the fit is scale and bias only and the
off-diagonal terms stay at identity. They are the weakly observable ones:
with few or poorly spread poses they will happily absorb scale and bias
error instead of measuring anything, and a diagonal model is then the more
honest fit, which is why the checkbox starts off. The table is how you tell — if the last
row barely improves on the middle one, they were not observable. On a
session that genuinely has misalignment, switching it off costs a factor
of 2.7 on |g| and 50 on the gyro direction; on imu_tk's Xsens recording,
4x and 6x.

**The initial rest period is the one part that needs care**, because the
gyro bias and the noise model come from it and nothing else can check
them. Two defences: only the genuinely static samples in it are used (the
fraction is reported, and warned about below 90 %), and the bias is a
*median*. A gyro measures a rate, so vibration that starts and ends at
rest integrates to zero and a mean handles it fine; what a mean cannot
survive is a knock that leaves the unit slightly **turned**. With 5 such
knocks in a 20 s window a plain mean is 0.016 deg/s off, the masked mean
0.0009; at 40 knocks the masked mean degrades to 0.020 while the median
holds 0.0023. On a clean window the median costs 0.0003 deg/s.

**Which input.** `--port` opens the device itself, which only works when
nothing else has it. While the process that owns the device (the hub) is running it does, so use
`--udp` instead and give the hub a second fan-out destination — the
capture and `insrcv` keep running right through the calibration:

For example fan-out destinations `127.0.0.1:29800,127.0.0.1:29801`.

A separate port, not a second listener on `insrcv`'s: two UDP sockets
bound to the same port do not both receive on Windows.

An existing `config.yaml` is **merged**, not overwritten: the calibration
keys are replaced and everything else is kept, including a noise model
somebody already tuned against a real dataset (a measured still window is
a decent estimate, but not better than that).

**Temperature.** MEMS biases drift with it, so the session's temperature
range goes into the generated file as a comment. Calibrate on a board
that has reached its working temperature; numbers taken while it is still
warming up describe a state it will not be in.

**The magnetometer rides along.** If the board streams one (`0x40/0x06`),
the same session calibrates it into the `mag:` keys, and nothing has to be
done differently for it: the turns between the poses are exactly the
coverage a magnetometer fit needs. `--no-mag` skips it, `--mag-field-ut`
scales the result to the local field strength instead of to whatever the
sensor reads on average. Details below and in `inslib_mag_calib.py`.

## Magnetometer: hard iron, soft iron, and where it is pointing

Turned through every attitude, a perfect magnetometer traces a sphere of
radius |F|, the local field strength. Two things spoil that. Permanently
magnetised material on the board adds a constant field in the body frame,
which moves the sphere off the origin (**hard iron**), and nearby ferrous
material bends the field, which turns it into an ellipsoid (**soft
iron**). Fitting the ellipsoid recovers both: its centre is `fixed_bias`,
and the matrix that maps it back onto a sphere is `misalignment`.

**What the ellipsoid fit cannot see is a rotation** — a sphere is a sphere
whichever way you turn it — so it pins down `M` only up to one, and the
fit here is deliberately the symmetric square root, the one that rotates
nothing. The rotation left over is the physical one: where the
magnetometer sits relative to the IMU. It is solved separately, from the
one quantity that a rigid pair of sensors cannot disagree about:

    the angle between the local field and gravity is a property of
    where on earth you are, not of how you are holding the unit

so it has to come out the same in every static pose, and the way it fails
to identifies the rotation (three well-spread poses are enough in
principle; a session has twenty). The final `mag: misalignment` is
`R_mag_to_imu * A_symmetric`, and the reported misalignment angles say how
far off the part was mounted. A magnetometer 3 deg off its board otherwise
contributes 3 deg of heading error to an otherwise perfect fusion.

```
magnetometer: 2000 samples, 48.5 uT field (WMM at 48.137 11.575, 2026.6)
hard iron   [+9.00, -5.51, +12.99] uT
soft iron   [-7.0, -0.8, +8.5] % (semi-axis spread)
|m| scatter 8.596 -> 0.151 uT, direction coverage 0.76
mag/IMU misalignment roll +1.43, pitch -2.55, yaw +3.48 deg (4.57 deg total)
measured dip 64.00 deg, scatter over 38 poses 2.26 -> 0.03 deg
```

Read the last line: the field/gravity angle wandered by 2.26 deg over the
poses before the alignment and by 0.03 deg after, which is what says the
rotation was real and not fitted noise. The measured dip can be held
against a WMM lookup for the same position, and `direction coverage` (1.0
is a cloud spread evenly over the sphere, 0 one confined to a plane) is
the number that catches a session where the unit was only ever yawed: the
hard iron across the thin axis of a flat cloud is guesswork, and no
residual will say so.

**Scale.** The fit forces `|corrected|` onto one number, which has to be
chosen. Give the tool a position (`--latlon`, or the Position field in the
window) and it looks the field strength up in INSLIB's own WMM, the same
model the filter arms its magnetometer fusion with. The same position also
fills in the local gravity, which is the accelerometer's half of exactly
this idea (below). Left at 0 the result
keeps whatever the sensor measures on average, which points the same way
but need not agree with the field strength the filter gates on. In the
window, the **GNSS** button next to Position fills the field in from the
receiver's own NAV-PVT fix instead of it being typed in by hand — it lights
up once a 3D fix is arriving on the link (the GNSS receiver's output passed
through by `inslib_hub.py`, above), and the first fix seen fills an empty
field in automatically.

**Watching it live.** Once a field strength is known, the Magnetometer tab
draws it as the line `|m|` has to sit on, and the panel reports how far
off the sensor is right now and over the last few seconds:

```
89 frames
raw +15.8 uT off, 15.6 rms over 6 s
calibrated +0.2 uT off, 0.3 rms
```

Same check the accelerometer gets against 9.81, and it has to be read the
same way: what matters is not one attitude but that the number stays put
while the unit is turned. A hard iron is an offset that *changes* with
attitude, so a magnetometer can look perfect lying on the desk and be 15
µT out upside down. The rms over the trace window is that, condensed.
`insrcv` shows the same pair on its stats line during a live run.

## Housing alignment: where the IMU sits in the box

The calibration above measures the sensor against itself. What it cannot
know is how the board was mounted, and a board 3 deg out of level makes
every attitude the filter reports 3 deg wrong.

So: put the unit on a level surface and press Capture (in
`inslib_calib_gui.py`). Any deviation from level is the mounting error,
and the rotation that removes it is folded into the same 3x3 matrices the
calibration already writes — for the accelerometer, the gyroscope and the
magnetometer alike, because they have to end up in one body frame.

**One placement gives roll and pitch only.** Gravity cannot see a rotation
about itself: a level box turned on the table still reads level. Capture a
second placement on a face that is **not parallel** to the first (level,
then on its side) and the yaw comes with it, because each placement is a
direction that is known in the housing frame and two non-parallel
directions determine a rotation completely (Wahba's problem, solved by
SVD). Note that two *opposite* faces, left and right, are parallel: they
pin the axis they turn about and say nothing about the rotation around it.
Level plus one side is the shortest recipe that gives all three angles.

```
housing alignment from 2 placement(s): bottom face down (normal, upright), left face down
IMU sits at roll +2.19, pitch -1.29, yaw +3.98 deg in the housing
total correction 4.74 deg
residual per placement [0.23, 0.23] deg
```

Two things this cannot do. It assumes the housing faces are square to one
another, and a housing where they are not shows up as the per-placement
residual — which is indistinguishable from a surface that was not level,
so a large residual says something is wrong without saying which. And it
aligns the IMU to the *housing*, not to the vehicle: if the box is then
bolted in crooked, that is a different rotation and only something that
knows the direction of travel (GNSS course over a straight run) can
measure it.

The capture is only as good as the accelerometer calibration under it: an
uncorrected bias of 0.1 m/s² is 0.6 deg of tilt, so calibrate the IMU
first, or point the window at a `config.yaml` that already carries one.
The window says so when there is none. Captures are averaged over the
whole window and rejected if the unit was not still, and the solve always
runs against the *current* calibration from scratch, so capturing one more
placement can only improve the answer and saving twice cannot apply the
rotation twice.

**On the board it is a parameter of its own.** `CFG-FRAME-HOUSING`, one
36-byte record for the whole device, which the board premultiplies onto
whatever its temperature table produces. So a housing pass alone is a
complete upload - **Upload to board** is offered for one, with no
recording behind it - and a temperature node uploaded next winter needs
no housing measurement beside it. Read it back with the board's configuration tool (`CFG-FRAME`), which prints it as roll/pitch/yaw, the three numbers a person can check
against the box, plus the matrix. In `config.yaml` it stays folded into
the 3x3 matrices: there is one of them per sensor there and nothing to
interpolate.

**It does not have to happen in the same session as the IMU calibration.**
Save is offered as soon as a housing alignment exists, with or without a
solve behind it: with no new solve the rotation is applied to the
matrices the `config.yaml` already carries - accelerometer, gyroscope and
the magnetometer section alike, because they all have to end up in the
same body frame. So the usual order works and so does the split one:

```sh
# in one go
record -> solve -> capture placements -> Save

# or separately, later, against the file the first pass wrote
inslib_calib_gui.py -o config.yaml     # point it at the existing file
capture placements -> Save
```

What matters is the order *within* one save, and the window handles it:
the rotation is folded into the file, and the copies still on screen move
with it. That is why the panel switches to "folded into the saved
calibration; capture again to check what is left" afterwards - the
captures stay valid measurements, and re-solving them against the new
calibration shows the leftover rather than the original error.

The one real ordering rule is the other one: **the capture is only as good
as the accelerometer calibration under it.** Doing the housing pass first,
against an uncalibrated unit, measures the accelerometer bias as if it
were a tilt. The panel says which of the three cases it is in, above the
face selector: measured through this session's calibration, through the
one in a `config.yaml`, or through the raw accelerometer.

**What the drop-down asks is which face is on the surface right now**, one
capture at a time - it is not a list of faces to measure. Picking the
wrong entry is the mistake the solve cannot see: it fits whatever it is
given and only the residual grows, which reads the same as a surface that
was not level. So the capture is compared against gravity first and
rejected when the unit is more than 25 deg away from the face it was filed
under, with the face it actually looks like named in the message. Captured
faces are ticked off in the list, and after the first one the selection
moves on to a face that is *not* parallel to it - the one that still has
something to add. The two steps and what each buys stand under the
buttons:

```
[x] 1. one face down      -> roll, pitch
[ ] 2. a non-parallel one -> yaw
```

**A board that corrects its own stream is the case to watch.** With
`CFG-IMU-APPLY_CAL` on and something stored to apply, the samples arriving
are already rotated by the housing key the board carries, so a capture
measures what is *left over* of the mounting error - while the upload
**replaces** that key rather than composing onto it. The panel says so and
offers to clear the switch in the live layer, which leaves the stored
image alone: the unit comes up correcting again at the next power cycle.
The same button is the answer to a `config.yaml` correcting on top of the
board, where every sample is corrected twice.
## inslib_imu_tk.py - the calibration maths (library, not a command)

numpy port of IMU-TK (BSD, Tedaldi/Pretto/Menegatti, ICRA 2014):
static-interval detection, the two least-squares problems, and a
Levenberg-Marquardt in place of Ceres — which for 9 and 12 parameters is
not a sacrifice worth the dependency. No I/O, so it runs against a
recording from anywhere.

The original C++ is not vendored here; what is kept is the reference
*recording* it ships with, in `testdata/`, because that is the only check
the port has against real hardware — see that directory's README and
`python/tests/test_imu_tk.py`.

## inslib_mag_calib.py, inslib_frame_align.py - the other two maths (libraries)

`inslib_mag_calib.py` is the ellipsoid fit (hard iron, soft iron, the
outlier rejection a magnetometer needs because it picks up whatever passed
close to it) plus the WMM lookup for the local field strength.

`inslib_frame_align.py` is the rotations: the magnetometer against the IMU
from the constancy of the dip angle, and the IMU against the housing from
placements on a level surface, together with the rotation toolbox they
share (Kabsch, the minimal rotation between two directions, the ZYX
extraction `ins_rotmat_to_rpy` uses). No I/O in either, and both are gated
by `python/tests/test_mag_calib.py`, which puts a known hard iron, soft
iron, magnetometer rotation and housing rotation into synthetic data and
requires them all back out.

`inslib_calib_report.py` is the calibration certificate and record the
GUI writes (see above): formatting plus the little statistics it prints
(parameter uncertainty from the fit's Jacobian, orientation coverage) and
the raw-data export. It computes no calibration of its own.

## testdata/ - reference recordings

Input data for the tests of the tools above, not INSLIB replay datasets
(those live in `datasets/`). Gzipped; `numpy.loadtxt` reads `.gz`
directly. Provenance and licences in `testdata/README.md`.

## inslib_ubx.py - shared framing (library, not a command)

UBX framing/resync/checksum, frame building, the class-0x40 payload
formats, the configuration interface (keys, the VALSET/VALGET/ACK
messages, the calibration record layouts and the SI-to-stream unit
conversions), the device-restart watch, and the datagram packing rule.
Imported by the tools above.

`python/tests/test_cfg_protocol.py` pins the configuration half against
the firmware's own `cfg_keys.h`, reading the record lengths and group ids
straight out of it. A record of the wrong length is refused loudly by the
board; a record of the right length with its fields in the wrong order is
not, which is why that agreement is checked rather than assumed.

Deliberately not pyubx2's stream reader for the *framing*: a capture must
be written before anything is parsed, and reading through a parser couples
the recording to it. The IMU payloads are raw floats, so a `0x24` (`$`)
turns up in them several times a second; pyubx2 read that as an NMEA
header, called `.decode()` on bytes that were not valid UTF-8, and raised
straight out of `read()` — taking the recording with it. pyubx2 is still
the right tool for the *standard* u-blox messages, handed one complete,
already checksum-verified frame at a time.
