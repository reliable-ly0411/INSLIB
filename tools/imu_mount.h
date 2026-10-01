/** @file imu_mount.h
 * @author Jan Zwiener (jan@zwiener.org)
 *
 * config.yaml `imu: mount_rpy_deg` (REQ-VER-036), shared by tools/replay.c
 * and tools/insrcv.c: the attitude of the sensor board's axes in the
 * vehicle body frame (ZYX roll/pitch/yaw, FRD) composed onto a REQ-NAV-037
 * calibration matrix, M := R(roll, pitch, yaw) * M, so the calibrated
 * samples M * (raw - fixed_bias) come out in vehicle axes. R is built like
 * R_b_to_n from a roll/pitch/yaw with the vehicle in place of NED.
 * python/replay.py contains the same calculations (mounted_calibration()).
 */
#ifndef IMU_MOUNT_H
#define IMU_MOUNT_H

#include <math.h>
#include <stdbool.h>

/* True if any of the three mounting angles is nonzero. */
static bool imu_mount_is_set(const float rpy_deg[3])
{
    return fabsf(rpy_deg[0]) > 0.0f || fabsf(rpy_deg[1]) > 0.0f || fabsf(rpy_deg[2]) > 0.0f;
}

/* M: column-major 3x3, all-zero read as identity. Returns false and leaves
 * M untouched on a non-finite angle. */
static bool imu_mount_compose(const float rpy_deg[3], float M[9])
{
    const float d2r = 3.14159265358979323846f / 180.0f;
    if (!isfinite(rpy_deg[0]) || !isfinite(rpy_deg[1]) || !isfinite(rpy_deg[2])) return false;

    const float sr = sinf(rpy_deg[0] * d2r);
    const float cr = cosf(rpy_deg[0] * d2r);
    const float sp = sinf(rpy_deg[1] * d2r);
    const float cp = cosf(rpy_deg[1] * d2r);
    const float sy = sinf(rpy_deg[2] * d2r);
    const float cy = cosf(rpy_deg[2] * d2r);
    /* column-major */
    const float R[9] = {cp * cy, cp * sy, -sp,
                        sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp,
                        cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp};

    bool  is_set = false;
    float in[9];
    int   i;
    for (i = 0; i < 9; ++i)
    {
        if (M[i] > 0.0f || M[i] < 0.0f) is_set = true;
    }
    for (i = 0; i < 9; ++i)
    {
        in[i] = is_set ? M[i] : ((i % 4 == 0) ? 1.0f : 0.0f);
    }
    int r;
    int c;
    for (c = 0; c < 3; ++c)
    {
        for (r = 0; r < 3; ++r)
        {
            M[r + 3 * c] = R[r] * in[3 * c] + R[r + 3] * in[1 + 3 * c] + R[r + 6] * in[2 + 3 * c];
        }
    }
    return true;
}

#endif /* IMU_MOUNT_H */
