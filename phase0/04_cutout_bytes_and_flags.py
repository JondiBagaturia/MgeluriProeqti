import requests, io, time
from astropy.io import fits

base_url = "https://irsa.ipac.caltech.edu/ibe/data/spherex/dr1/level2/2025W33_2C/l2b-v27-2026-228/6/level2_2025W33_2C_0321_1D6_spx_l2b-v27-2026-228.fits"
ra, dec = 269.452, 4.693
size_deg = 2/60.0
url = f"{base_url}?center={ra},{dec}d&size={size_deg}"

r = None
for attempt in range(8):
    r = requests.get(url, timeout=30)
    if r.status_code == 200:
        break
    print(f"attempt {attempt}: {r.status_code} {r.content[:50]}, retrying...")
    time.sleep(5)

print("status", r.status_code, "bytes", len(r.content))
hdul = fits.open(io.BytesIO(r.content))
hdul.info()
for h in hdul:
    data_size = h.data.nbytes if h.data is not None else 0
    print(f"ext {h.name}: shape={getattr(h.data,'shape',None)} data_bytes={data_size} ({data_size/len(r.content)*100:.1f}%)")
print()
print("FLAGS header MSK keys:")
flags_hdr = hdul['FLAGS'].header
for k in flags_hdr:
    if k.startswith('MSKN') or k.startswith('MSKD'):
        print(" ", k, "=", flags_hdr[k])
