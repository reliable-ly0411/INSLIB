# INSLIB

![INSLIB](https://raw.githubusercontent.com/jnz/INSLIB/main/doc/figures/inslib_logo_transparent_small.png)

Python binding of [INSLIB](https://github.com/jnz/INSLIB), a portable C11
library for 3D navigation state estimation: a 15/18-state error-state Kalman
filter fusing IMU, GNSS, magnetometer, barometer and local position, plus
standalone AHRS/ARS attitude filters and a baro/accel vertical channel, run
side by side with automatic mode arbitration.

The wheel ships the compiled C library, the Python side is a thin `ctypes`
layer over it, with no required dependencies.

## Install

```sh
pip install inslib
```

Wheels exist for Linux x86_64, aarch64 and armv7l (64- and 32-bit Raspberry
Pi OS included), Windows x64 and ARM64, and macOS Intel and Apple silicon.
Anywhere else (a Pi Zero/1, for example), install from a checkout of the
repository with `pip install -e python/`, which needs a C compiler.

The import name is `INSLIB`.

## Quick start

Roll and pitch from nothing but an IMU (gyroscope and accelerometer),
with the ARS attitude filter that runs inside every `Navigator`:

```python
import math
import random
from INSLIB import Navigator

D2R = math.pi / 180.0

# Stand-in for a real sensor: an IMU resting at roll 10 deg, pitch -5 deg.
# Body frame FRD (front/right/down),
# accelerometer (acc) in m/s^2 (lying flat would read 0, 0, -9.81),
# gyroscope (gyr) in rad/s.
# A little noise is added here, like a real sensor.
def read_imu():
    r, p, g = 10.0 * D2R, -5.0 * D2R, 9.81
    acc = (g * math.sin(p), -g * math.sin(r) * math.cos(p),
           -g * math.cos(r) * math.cos(p))
    acc = [a + random.gauss(0.0, 0.02) for a in acc]
    gyr = [random.gauss(0.0, 0.002) for _ in range(3)]
    return acc, gyr

random.seed(1)
nav = Navigator()

dt = 0.01                                  # 100 Hz simulated data
for k in range(1, 501):                    # Simulate for 5 s
    acc, gyr = read_imu()
    nav.imu(int(k * dt * 1e6), dt, acc=acc, gyr=gyr)
    nav.update()                           # Process data
    rpy = nav.rpy_ars()                    # Get [roll, pitch, yaw] in rad
    if rpy:
        if k % 100 == 0:
            print("t %.0f s  roll %6.2f deg  pitch %6.2f deg"
                  % (k * dt, rpy[0] / D2R, rpy[1] / D2R))
```

It prints:

```
t 1 s  roll   9.97 deg  pitch  -4.98 deg
t 2 s  roll  10.00 deg  pitch  -5.01 deg
t 3 s  roll  10.03 deg  pitch  -5.02 deg
t 4 s  roll  10.03 deg  pitch  -5.00 deg
t 5 s  roll  10.04 deg  pitch  -4.99 deg
```

For your own sensor, replace `read_imu()` and pass its real time stamps.
More sensors are one call each: GNSS for position and velocity, a
magnetometer for a north-referenced heading, a barometer for height. The
[Python tutorial](https://github.com/jnz/INSLIB/blob/main/tutorial/python_tutorial.md)
shows how.

Conventions: body frame FRD, navigation frame NED, Hamilton quaternion
`[w, x, y, z]`, time in int64 microseconds, angles in radians.

## Documentation

- [Python tutorial](https://github.com/jnz/INSLIB/blob/main/tutorial/python_tutorial.md)
- [API reference](https://jnz.github.io/INSLIB/): every class, method and
  `Config` field
- [Manual (PDF)](https://github.com/jnz/INSLIB/blob/main/doc/INSLIB_manual.pdf)
- [Changelog](https://github.com/jnz/INSLIB/blob/main/CHANGELOG.md)

The replay, post-processing GUI and calibration tools live in the
repository's `tools/` directory, not in this package.

## License

AGPL-3.0-or-later.
