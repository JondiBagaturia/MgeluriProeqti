from astroquery.ipac.irsa import Irsa
import sys

print("Listing SIA collections matching 'spherex'...")
try:
    cols = Irsa.list_collections(servicetype='SIA')
    matches = [c for c in cols if 'spherex' in str(c).lower()] if cols is not None else []
    print("Matches:", matches)
    print(cols if len(str(cols)) < 3000 else str(cols)[:3000])
except Exception as e:
    print("list_collections failed:", repr(e))

print()
print("Listing TAP tables matching 'spherex'...")
try:
    tables = Irsa.get_tables()
    names = [t.name if hasattr(t,'name') else str(t) for t in tables]
    sx = [n for n in names if 'spherex' in n.lower()]
    print("Tables:", sx)
except Exception as e:
    print("get_tables failed:", repr(e))
