import sys, time, json, torch, torch.nn as nn, torch.nn.functional as F
from terratorch.models.decoders import UperNetDecoder
dev = "mps"; out = {}
class Upn(nn.Module):
    def __init__(s, ch):
        super().__init__()
        s.neck = nn.ModuleList([nn.Sequential(nn.ConvTranspose2d(768,768,2,2), nn.BatchNorm2d(768), nn.GELU(), nn.ConvTranspose2d(768,768,2,2)),
                                nn.ConvTranspose2d(768,768,2,2), nn.Identity(), nn.MaxPool2d(2)])
        s.dec = UperNetDecoder([768]*4, channels=ch, scale_modules=False, pool_scales=(1, 7)); s.cls = nn.Conv2d(ch, 10, 1)
    def forward(s, f):
        return F.interpolate(s.cls(s.dec([m(x).contiguous() for m, x in zip(s.neck, f)])), size=224, mode="bilinear")
for ch in (512, 256):
    for b in (8, 16):
        m = Upn(ch).to(dev); opt = torch.optim.AdamW(m.parameters(), 6e-4)
        feats = [torch.randn(b, 768, 14, 14, device=dev) for _ in range(4)]
        lab = torch.randint(0, 10, (b, 224, 224), device=dev)
        def step(train=True):
            if train:
                with torch.autocast("mps", dtype=torch.bfloat16): lg = m(feats)
                l = F.cross_entropy(lg.float(), lab); opt.zero_grad(); l.backward(); opt.step()
            else:
                with torch.no_grad(), torch.autocast("mps", dtype=torch.bfloat16): m(feats)
        for train in (True, False):
            step(train); torch.mps.synchronize(); t = time.perf_counter()
            for _ in range(6): step(train)
            torch.mps.synchronize()
            out[f"ch{ch}_b{b}_{'train' if train else 'fwd'}_ms"] = (time.perf_counter()-t)/(6*b)*1e3
        out[f"ch{ch}_params"] = sum(p.numel() for p in m.parameters())
        print(json.dumps(out), flush=True); del m, opt; torch.mps.empty_cache()
