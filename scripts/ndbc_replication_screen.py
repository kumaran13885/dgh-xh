import gzip, io, math, os, re, sys, time
import numpy as np
import pandas as pd
import requests

NETWORKS = {
    "atlantic": ["44005", "44007", "44008", "44009", "44011", "44013",
                 "44014", "44017", "44025", "44027", "44065", "41025"],
    "westcoast": ["46011", "46012", "46013", "46014", "46022", "46025", "46026",
                  "46027", "46028", "46042", "46047", "46050", "46053", "46054",
                  "46069", "46086"],
}
CANDIDATES = NETWORKS["atlantic"]
YEARS = [2019, 2020, 2021]
SPLIT = pd.Timestamp("2021-06-01 00:00:00")
MIN_COVERAGE = 0.80
OUT = "ndbc_atlantic_screen"
MISSING = {99.0, 999.0, 9999.0}


def fetch(url, tries=3):
    for k in range(tries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 * (k + 1))
    return None


def parse_stdmet(raw_bytes):
    try:
        text = gzip.decompress(raw_bytes).decode("latin-1")
    except OSError:
        text = raw_bytes.decode("latin-1")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = lines[0].lstrip("#").split()
    body = [ln for ln in lines[1:] if not ln.startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(body)), sep=r"\s+", header=None,
                     names=header, engine="python")
    yy = "YYYY" if "YYYY" in df.columns else ("YY" if "YY" in df.columns else "#YY")
    mm = "mm" if "mm" in df.columns else None
    ts = pd.to_datetime(dict(year=df[yy], month=df["MM"], day=df["DD"], hour=df["hh"],
                             minute=df[mm] if mm else 0), errors="coerce")
    out = pd.DataFrame({"timestamp": ts})
    for c in ("WVHT", "WSPD", "WDIR"):
        v = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
        out[c] = v
        out.loc[out[c].isin(MISSING), c] = np.nan
    return out.dropna(subset=["timestamp"])


def coverage(df):
    hourly = df.set_index("timestamp").resample("1h").mean()
    full = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31 23:00", freq="h")
    hourly = hourly.reindex(full)
    res = {}
    for name, mask in (("train", full < SPLIT), ("test", full >= SPLIT)):
        for c in ("WVHT", "WSPD", "WDIR"):
            res[f"{c}_{name}"] = float(hourly.loc[mask, c].notna().mean())
    return res


def coords_from_metadata():
    raw = fetch("https://www.ndbc.noaa.gov/activestations.xml")
    if raw is None:
        return {}
    xml = raw.decode("utf-8", "ignore")
    out = {}
    for m in re.finditer(r'<station\s+([^>]*)>', xml):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        if "id" in attrs and "lat" in attrs and "lon" in attrs:
            out[attrs["id"].upper()] = (float(attrs["lat"]), float(attrs["lon"]))
    return out


def haversine_km(a, b):
    (la1, lo1), (la2, lo2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    dphi, dl = p2 - p1, math.radians(lo2 - lo1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def main():
    global CANDIDATES, OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", choices=sorted(NETWORKS), default="atlantic")
    args, _ = ap.parse_known_args()
    CANDIDATES = NETWORKS[args.network]
    OUT = f"ndbc_{args.network}_screen"
    print(f"network: {args.network}  candidates: {CANDIDATES}")
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for b in CANDIDATES:
        parts = []
        for y in YEARS:
            raw = fetch(f"https://www.ndbc.noaa.gov/data/historical/stdmet/{b.lower()}h{y}.txt.gz")
            if raw is not None:
                try:
                    parts.append(parse_stdmet(raw))
                except Exception as e:
                    print(f"  {b} {y}: parse failed ({e})")
        if not parts:
            rows.append({"buoy": b, **{f"{c}_{p}": 0.0 for c in ("WVHT", "WSPD", "WDIR")
                                       for p in ("train", "test")}})
            continue
        rows.append({"buoy": b, **coverage(pd.concat(parts))})

    cov = pd.DataFrame(rows).set_index("buoy")
    cov["included"] = (cov >= MIN_COVERAGE).all(axis=1)
    cov.round(3).to_csv(os.path.join(OUT, "coverage.csv"))
    print(cov.round(3).to_string())

    inc = cov.index[cov["included"]].tolist()
    print(f"\nincluded by rule (>= {MIN_COVERAGE:.0%} coverage): {inc}")
    if len(inc) < 6:
        sys.exit("Fewer than 6 buoys qualify")

    meta = coords_from_metadata()
    missing = [b for b in inc if b not in meta]
    if missing:
        print(f"\nNo coordinates in activestations.xml for {missing}")
    coords = {b: meta[b] for b in inc if b in meta}

    print("\nBUOY_COORDS = {")
    for b, (la, lo) in coords.items():
        print(f'    "{b}": ({la:.3f}, {lo:.3f}),')
    print("}")

    print("\nnearest-neighbour distances (km), k = 4:")
    for b, c in coords.items():
        d = sorted(haversine_km(c, coords[o]) for o in coords if o != b)[:4]
        print(f"  {b}: " + ", ".join(f"{x:.0f}" for x in d))


if __name__ == "__main__":
    main()
