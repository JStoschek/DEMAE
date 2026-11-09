import sys, csv, time, hashlib, math, statistics as st
from concurrent.futures import ThreadPoolExecutor
sys.argv = [sys.argv[0], "x", "/private/tmp/claude-501/-Users-jstoschek-Code-DEMMAE/7f3337a8-5547-4b0c-9abe-6ffa87f17433/scratchpad/p187/bench_tmp"]
sys.path.insert(0, sys.path[0])
import importlib.util
spec = importlib.util.spec_from_file_location("gp", __file__.replace("read_bench.py", "geomorphon_pass.py"))
gp = importlib.util.module_from_spec(spec); spec.loader.exec_module(gp)
from dataset.geometry import parse_cog_key, lat_band, COLS_PER_DEGREE
rows = [r["tile_id"] for f in ("head_train_tiles","val_tiles","test_tiles") for r in csv.DictReader(open(f"artifacts/task1/{f}.csv"))]
# anisotropy over all 27k: needed column buffer at L=50/100
need = {50: [], 100: []}
for t in rows:
    c = parse_cog_key(t.rsplit("_",2)[0]); W = COLS_PER_DEGREE[lat_band(c.origin_lat + .5)]
    r0 = int(t.rsplit("_",2)[1])
    lat = c.origin_lat + 1 - (r0 + 112) / 3600
    ratio = (1/3600) / (math.cos(math.radians(lat)) / W)
    for L in need: need[L].append(math.ceil(L * ratio))
for L, v in need.items():
    v = sorted(v); print(f"L={L}: column buffer px  median={v[len(v)//2]} p99={v[int(.99*len(v))]} max={v[-1]}  tiles needing >{L}: {sum(x>L for x in v)}")
rows.sort(key=lambda t: hashlib.blake2b(t.encode(), digest_size=8).digest(), reverse=True)
sample = rows[:240]
def one(t):
    t0 = time.perf_counter(); gp.read_buffered(t, 50); return time.perf_counter() - t0
seq = [one(t) for t in sample[:40]]
print("sequential cold read s: median %.3f mean %.3f p90 %.3f" % (st.median(seq), st.mean(seq), sorted(seq)[36]))
for k in (16, 32):
    chunk = sample[40:140] if k == 16 else sample[140:240]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(k) as ex: list(ex.map(one, chunk))
    dt = time.perf_counter() - t0
    print(f"{k} threads: {len(chunk)} windows in {dt:.1f}s -> {len(chunk)/dt:.1f} windows/s")
