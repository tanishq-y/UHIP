import rasterio as rio
import numpy as np
from pathlib import Path

RAW = Path("data/raw/landsat")
PROC = Path("data/processed")

st_path = sorted(RAW.glob("*ST_B10.TIF"))[0]
qa_path = sorted(RAW.glob("*QA_PIXEL.TIF"))[0]

with rio.open(st_path) as st_src, rio.open(qa_path) as qa_src:
    st = st_src.read(1).astype("float32")
    qa = qa_src.read(1).astype("uint32")
    profile = st_src.profile

    bad = (st==0) | ((qa>>1)&1) | ((qa>>2)&1) | ((qa>>3)&1) | ((qa>>4)&1) | ((qa>>5)&1)
    st = np.where(bad, np.nan, st)
    lst_c = st * 0.00341802 + 149.0 - 273.15
    lst_c = np.where((lst_c < 0) | (lst_c > 62), np.nan, lst_c)

print(f"ST: {st_path.name}")
print(f"QA: {qa_path.name}")
print(f"Final LST: {np.nanmin(lst_c):.1f}°C to {np.nanmax(lst_c):.1f}°C")
print(f"Valid: {np.isfinite(lst_c).sum()/lst_c.size*100:.1f}%")

profile.update(dtype="float32", nodata=np.nan, compress="lzw")
with rio.open(PROC/"LST_Celsius.tif","w",**profile) as dst:
    dst.write(lst_c,1)