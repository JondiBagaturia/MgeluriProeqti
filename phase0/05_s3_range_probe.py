import requests, time

bucket = "nasa-irsa-spherex"
region = "us-east-1"
key = "dr1/level2/2025W33_2C/l2b-v27-2026-228/6/level2_2025W33_2C_0321_1D6_spx_l2b-v27-2026-228.fits"
url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"

# 1. HEAD request to check accessibility + size
t0=time.time()
r = requests.head(url, timeout=20)
print("HEAD:", r.status_code, dict(r.headers), f"{time.time()-t0:.2f}s")

# 2. Range request for first 20000 bytes (headers)
t0=time.time()
r = requests.get(url, headers={"Range": "bytes=0-20159"}, timeout=20)
print("RANGE first 20160B:", r.status_code, len(r.content), f"{time.time()-t0:.2f}s")
print("Accept-Ranges:", r.headers.get('Accept-Ranges'), "Content-Range:", r.headers.get('Content-Range'))
print(r.content[:80])
