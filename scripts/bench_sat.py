"""Benchmark stable-audio-tools (PyTorch reference), mirroring aria's --uncond
computation exactly: build the real DiT + autoencoder, run the pingpong loop
(8 steps, zero cross/global cond, local_add=None) and decode latent->audio.
Reports warm wall-time + peak memory (VRAM on cuda, RSS on cpu)."""
import sys, json, time, resource
import torch
from safetensors.torch import load_file
from stable_audio_tools.models.dit import DiffusionTransformer
from stable_audio_tools.models.autoencoders import create_autoencoder_from_config
from stable_audio_tools.inference.sampling import build_schedule, LogSNRShift

device  = sys.argv[1]                                   # cuda | cpu
mdir    = sys.argv[2]
seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
steps   = int(sys.argv[4]) if len(sys.argv) > 4 else 8
N       = int(sys.argv[5]) if len(sys.argv) > 5 else 3
half    = (device == "cuda")
dt      = torch.float16 if half else torch.float32

cfg = json.load(open(f"{mdir}/model_config.json"))
sr  = cfg["sample_rate"]
dcfg = cfg["model"]["diffusion"]["config"]
obj  = cfg["model"]["diffusion"].get("diffusion_objective", "rf_denoiser")
downsample = cfg["model"]["pretransform"]["config"]["downsampling_ratio"]
T = round(seconds * sr / downsample)

sd = load_file(f"{mdir}/model.safetensors")
dit = DiffusionTransformer(diffusion_objective=obj, **dcfg).eval()
dit.load_state_dict({k[len("model.model."):]: v for k, v in sd.items()
                     if k.startswith("model.model.")}, strict=False)
ae = create_autoencoder_from_config({"model": cfg["model"]["pretransform"]["config"],
                                     "sample_rate": sr}).eval()
ae.load_state_dict({k[len("pretransform.model."):]: v for k, v in sd.items()
                    if k.startswith("pretransform.model.")}, strict=False)
for m in ae.modules():
    if hasattr(m, "noise_regularize"): m.noise_regularize = False
    if hasattr(m, "mask_noise"):       m.mask_noise = 0.0

if half: dit = dit.half(); ae = ae.half()   # halve on CPU first to avoid an fp32-on-GPU transient
dit = dit.to(device); ae = ae.to(device)

cross = torch.zeros(1, 257, 768, device=device, dtype=dt)
glob  = torch.zeros(1, 768, device=device, dtype=dt)
shift = LogSNRShift(anchor_length=2000, anchor_logsnr=-6.2, rate=1.0, logsnr_end=2.0)
sched = build_schedule(steps=steps, sigma_max=1.0, dist_shift=shift, effective_seq_len=T)

def run():
    with torch.no_grad():
        x = torch.randn(1, 256, T, device=device, dtype=dt)
        for i in range(steps):
            tc = sched[i].item(); tn = sched[i + 1].item()
            v = dit._forward(x, torch.tensor([tc], device=device, dtype=dt),
                             cross_attn_cond=cross, global_embed=glob)
            x = (1 - tn) * (x - tc * v) + tn * torch.randn_like(x)
        z = ae.bottleneck.decode(x)
        dec = ae.decoder(z)
        ae.pretransform.decode(dec)
    if device == "cuda": torch.cuda.synchronize()

run()                                                   # warm
if device == "cuda": torch.cuda.reset_peak_memory_stats()
t0 = time.time()
for _ in range(N): run()
sec = (time.time() - t0) / N
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
if device == "cuda":
    alloc = torch.cuda.max_memory_allocated() / 1048576.0
    resv  = torch.cuda.max_memory_reserved() / 1048576.0
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1048576.0     # actual process VRAM (ctx+weights+act), comparable to aria
    print(f"RESULT sat {device} T={T} dur={seconds}s time={sec:.3f}s vram_used={used:.0f}MB "
          f"vram_alloc={alloc:.0f}MB vram_reserved={resv:.0f}MB rss={rss:.0f}MB")
else:
    print(f"RESULT sat {device} T={T} dur={seconds}s time={sec:.3f}s rss={rss:.0f}MB")
