The dataset in this directory comes from the KF-GINS repository:

    https://github.com/i2Nav-WHU/KF-GINS 

Thanks to the team of Prof. Xiaoji Niu of the Integrated and Intelligent
Navigation (i2Nav) group from GNSS Research Center of Wuhan University for
providing the dataset and the open-source KF-GINS software.

Contents:

* `imu.csv`, `gnss.csv`, `ref.csv`: the Leador A15 trial in the replay CSV
  format, `ref.csv` is `truth.nav`
* `config.yaml`: INSLIB setup of the INSLIB table in the top level readme.md,
  scored against `ref.csv`
* `gnss_vel.csv`, `gnss_vel_outage.csv`: `gnss.csv` with a SYNTHETIC GNSS
  velocity on every fix (truth velocity at the antenna plus 0.12 m/s white
  noise per axis), the second with the fixes of eleven 60 s outages removed
* `config_vel.yaml`, `config_vel_outage.yaml`, `config_vel_outage_nhc.yaml`:
  INSLIB setup of the INSLIB vs KF-GINS vs EKF3 table in readme.md, normal
  operation, coasting through the outages, and the same with the
  non-holonomic constraint and the IMU mounting yaw taken from the truth
* `gnss_fault_<type>.csv`, `gnss_spp.csv`, `gnss_spp_<type>.csv`:
  `gnss_vel.csv` with SYNTHETIC GNSS faults (spikes, a wrong RTK fix,
  multipath, honest degradation, velocity spikes, lasting steps, 500 m
  blunders), with every fix at L1 single point accuracy, and with faults
  on top of that. Written by `python3 tools/gnss_faults.py datasets/kfgins`
  with fixed seeds, its docstring has the fault model (REQ-VER-050)
* `config_fault_<type>.yaml`, `config_spp.yaml`, `config_spp_<type>.yaml`:
  INSLIB with its chi2 outlier gate on through these files, normal
  operation as `config_vel.yaml` otherwise (`make datasets-kfgins-faults`)
* `ADIS16460/`, `ADIS16465/`, `ICM20602/`, `i300/`: `config.yaml` and the
  matching KF-GINS `kf-gins.yaml` of the other four IMUs of the table. The
  data is not part of this repository, the header of each `config.yaml`
  describes the CSV bundle to convert it into.

The four other trials: https://github.com/i2Nav-WHU/awesome-gins-datasets
