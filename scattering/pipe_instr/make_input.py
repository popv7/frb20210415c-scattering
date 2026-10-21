#!/usr/bin/env python3
"""make_input.py -- build the arrival-time pipeline's input file,
Data/FRB20210415C.npz, from the public Catalog 2 dynamic spectrum: the
unmasked channels, their frequencies, the catalogue DM and the window length.

    python make_input.py      (run_toas.py calls it automatically)
"""
import os

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H5 = os.path.join(HERE, "..", "..", "data", "FRB20210415C_stokesi_dynamic_spectrum.h5")
OUT = os.path.join(HERE, "Data", "FRB20210415C.npz")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with h5py.File(H5, "r") as f:
        good = f["good_freq"][:].astype(bool)
        freqs = f["index_map/freqs"][:]
        np.savez(OUT, wfall=f["data"][:][good].astype(np.float32), dfs=freqs[good],
                 DM=np.float64(f.attrs["dm_incoherent"]),
                 duration=np.float64(f.attrs["res_time"] * f.attrs["num_time"]),
                 bandwidth=np.float64(400.0), telescope="CHIME")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
