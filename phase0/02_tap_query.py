from astroquery.ipac.irsa import Irsa
import time, collections

ra, dec = 269.452, 4.693
size_deg = 5/60.0  # 5 arcmin FOV

query = f"""
SELECT 'https://irsa.ipac.caltech.edu/' || a.uri || '?center={ra},{dec}d&size={size_deg}' AS uri,
       p.time_bounds_lower, p.energy_bandpassname
FROM spherex.artifact a JOIN spherex.plane p ON a.planeid = p.planeid
WHERE 1 = CONTAINS(POINT('ICRS', {ra}, {dec}), p.poly)
ORDER BY p.time_bounds_lower
"""
for i in range(2):
    t0 = time.time()
    res = Irsa.query_tap(query).to_table()
    dt = time.time() - t0
    print(f"Run {i}: {len(res)} rows in {dt:.2f}s")

print(res.colnames)
print(res[0]['uri'])
print(res[0]['time_bounds_lower'], res[0]['energy_bandpassname'])

bands = collections.Counter(res['energy_bandpassname'])
print("Band counts:", dict(bands))

releases = collections.Counter('dr1' if 'dr1' in u else ('qr3' if 'qr3' in u else 'other') for u in res['uri'])
print("Release counts:", dict(releases))

import numpy as np
t = np.array(res['time_bounds_lower'])
print("time range MJD:", t.min(), t.max(), "span days:", t.max()-t.min())
