"""Stage 5b — corpus storage.

Persist normalized tiles for the Phase-1 dataloader. Format (webdataset / zarr /
npy / GeoTIFF crops) is open, judged on streaming-read throughput:
resolved by "Decide storage format + tile-manifest schema".
"""
