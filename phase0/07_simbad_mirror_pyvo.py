import pyvo
import time

tap = pyvo.dal.TAPService("https://simbad.harvard.edu/simbad/sim-tap")

targets = {
    "Barnard's star": "NAME Barnard's star",
    "eps Ind B": "eps Ind B",
    "WISE J085510.83-071442.5": "WISE J085510.83-071442.5",
    "RW Aurigae": "RW Aur",
    "R Draconis": "R Dra",
}

for label, main_id in targets.items():
    q = f"""
    SELECT main_id, ra, dec, pmra, pmdec, plx_value
    FROM basic
    WHERE main_id = '{main_id}'
    """
    t0=time.time()
    try:
        res = tap.search(q).to_table()
        print(f"{label} ({main_id!r}): {time.time()-t0:.2f}s -> {len(res)} rows")
        print(res)
    except Exception as e:
        print(f"{label}: FAILED {e}")
    print()

# fallback: fuzzy search via ident/TAP join if exact main_id doesn't match
print("=== fallback: search via IDS table for RW Aur / R Dra ===")
for name in ["RW Aur", "R Dra"]:
    q = f"""
    SELECT b.main_id, b.ra, b.dec, b.pmra, b.pmdec, b.plx_value
    FROM ident AS i JOIN basic AS b ON i.oidref = b.oid
    WHERE i.id = '{name}'
    """
    try:
        res = tap.search(q).to_table()
        print(name, "->", res)
    except Exception as e:
        print(name, "FAILED", e)
