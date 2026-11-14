from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord
import astropy.units as u
import time

coord = SkyCoord(269.452, 4.693, unit='deg', frame='icrs')
for i in range(2):
    t0 = time.time()
    try:
        res = Irsa.query_sia(pos=(coord, 2.5*u.arcmin), collection=['spherex_dr1','spherex_qr3'])
        dt = time.time()-t0
        print(f"SIA run {i}: {len(res)} rows in {dt:.2f}s")
    except Exception as e:
        dt = time.time()-t0
        print(f"SIA run {i} failed after {dt:.2f}s:", repr(e))

print(res.colnames)
print(res[0])
