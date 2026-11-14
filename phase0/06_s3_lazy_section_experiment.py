import time
import s3fs
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

bucket = "nasa-irsa-spherex"
key = "dr1/level2/2025W33_2C/l2b-v27-2026-228/6/level2_2025W33_2C_0321_1D6_spx_l2b-v27-2026-228.fits"
s3path = f"s3://{bucket}/{key}"

fs = s3fs.S3FileSystem(anon=True)

t0 = time.time()
with fs.open(s3path, 'rb') as f:
    hdul = fits.open(f, lazy_load_hdus=True, memmap=False)
    print(f"Opened (headers loaded) in {time.time()-t0:.2f}s")
    hdul.info()

    # Get WCS from IMAGE header only (doesn't load pixel data)
    t1 = time.time()
    header = hdul['IMAGE'].header
    wcs = WCS(header)
    print(f"Got header+WCS in {time.time()-t1:.2f}s, NAXIS1={header['NAXIS1']} NAXIS2={header['NAXIS2']}")

    coord = SkyCoord(269.452, 4.693, unit='deg', frame='icrs')
    x, y = wcs.world_to_pixel(coord)
    print(f"Target pixel: x={x:.1f} y={y:.1f}")

    # Define a small cutout window (5 arcmin at ~6"/px -> ~50x50 px)
    half = 25
    x0, x1 = int(x-half), int(x+half)
    y0, y1 = int(y-half), int(y+half)
    print(f"Window: x[{x0}:{x1}] y[{y0}:{y1}]")

    t2 = time.time()
    # .section does a lazy, ranged read of only the needed slice
    sub_image = hdul['IMAGE'].section[y0:y1, x0:x1]
    print(f"IMAGE section read in {time.time()-t2:.2f}s, shape={sub_image.shape}, bytes={sub_image.nbytes}")

    t3 = time.time()
    sub_flags = hdul['FLAGS'].section[y0:y1, x0:x1]
    sub_var = hdul['VARIANCE'].section[y0:y1, x0:x1]
    sub_zodi = hdul['ZODI'].section[y0:y1, x0:x1]
    print(f"FLAGS+VARIANCE+ZODI sections read in {time.time()-t3:.2f}s")

    print(f"TOTAL time: {time.time()-t0:.2f}s")
    print("Sample IMAGE values:", sub_image[:3,:3])

# Check how many bytes s3fs actually pulled over the wire
print("fs.cat count / transfer stats not directly exposed; checking instance cache info:")
