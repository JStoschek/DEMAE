# DEMAE Phase 1: Ops Runbook

**Scope:** everything between "I have a plan and a budget" and "the box is gone and the run survived it." The recipe itself lives in [`phase1-pretraining-plan.md`](phase1-pretraining-plan.md); this is the operational wrapper around it.
**Hardware:** single rented GPU, vast.ai. **Store:** Cloudflare R2, account `79083f36be71e95fad1b0b06d2bb62a1`.
**Scripts:** [`push_code.sh`](../scripts/push_code.sh), [`provision_box.sh`](../scripts/provision_box.sh), [`sync_checkpoints.sh`](../scripts/sync_checkpoints.sh), [`teardown_box.sh`](../scripts/teardown_box.sh), [`check_corpus.py`](../scripts/check_corpus.py).
**Status:** v2, 2026-08-03 — the four gaps R0 hit on a live box are closed in the scripts (#107); v1 was 2026-07-31. Steps marked **[verified]** were run end-to-end on this laptop; steps marked **[untested]** need a live box.

---

## 0. At a glance

| # | Step | Where | Script |
|---|---|---|---|
| 1 | Verify the corpus in R2 | laptop | `check_corpus.py --bucket` |
| 2 | Rent the instance | vast console | — |
| 3 | Add the SSH alias | laptop | — |
| 4 | Push code, then provision | laptop → box | `push_code.sh --provision` |
| 5 | Read the provision footer | — | — |
| 6 | Start training and the sync loop | box, tmux | `sync_checkpoints.sh` |
| 7 | Let it run | — | — |
| 8 | Verify, then destroy | laptop | `teardown_box.sh` |
| 9 | Sync the hauled W&B runs | laptop | `wandb sync` |

Nothing here rents or destroys automatically. Both spend money or delete data and stay a human's to run — see [`provision_box.sh:26`](../scripts/provision_box.sh:26).

---

## 1. One-time laptop setup

Done once, then never again.

| Item | State |
|---|---|
| `vastai` CLI on PATH | done — `uv tool install vastai`, 1.5.1 |
| `vastai set api-key` | done |
| SSH public key at [cloud.vast.ai/manage-keys](https://cloud.vast.ai/manage-keys/) | done |
| `boto3` for the laptop-side bucket checks | already a core dependency, present in `.venv` |
| `[r2]` profile in `~/.aws/credentials` | done — **[verified]** authenticates against `demae-corpus` |
| W&B account and project `demae-phase1` | done — **[verified] 2026-08-03**, entity `jonathan-stoschek-purdue-university`, zero runs. The key belongs in the §11 env block, not here: it is needed on the box, not the laptop |
| Bucket lock rule on `demae-checkpoints` | **required — see §11** |

**Adding the SSH key only affects instances created afterwards.** Add it before renting or you are locked out of a box you are paying for.

**Run every laptop-side Python script through `uv run`.** Both `check_corpus.py` and `teardown_box.sh` call bare `python3`, resolved from `PATH` at runtime. With the venv active that finds boto3; from a shell with only conda `base` it does not, and teardown dies at step 5/7 — *after* step 3/7 has already run a sync. `uv run` makes it deterministic.

---

## 2. Before you rent

Cheap insurance: prove the bucket is whole from the laptop, where the cost is seconds rather than idle GPU time.

```bash
uv run python3 scripts/check_corpus.py \
  --bucket demae-corpus --prefix corpus_500k/ --profile r2 \
  --endpoint-url https://79083f36be71e95fad1b0b06d2bb62a1.r2.cloudflarestorage.com
```

**[verified] 2026-07-31** — 37 objects (34 shards), 50.17 GB payload, `dataset_version bcbd6484d8116199c5c1160e21ea0284`, byte-identical to local `data/corpus_500k`. That hash equals `EXPECT_VERSION` at [`provision_box.sh:40`](../scripts/provision_box.sh:40), so provisioning steps 3 and 4 will pass.

Why this check is not redundant with rclone: `rclone copy --checksum` proves the *transfer* was faithful. It cannot prove the *source* was whole. A bucket missing `shard_00007` copies cleanly, reports success, and leaves the box training on less corpus than the manifest describes — with `dataset_version` still reading correctly, because that hash is over the manifest, not over what is on disk.

---

## 3. Rent the instance

In the vast console. Requirements come from provision's preflight, which is deliberately fatal and deliberately first — a wrong host should cost 30 seconds, not the 20 minutes it takes to discover the corpus does not fit after it has half-downloaded.

| Setting | Value | Enforced at |
|---|---|---|
| Image | `vastai/pytorch` (any recent py311–py313 tag) | — |
| GPU | 1 (multi-GPU warns, does not block) | [`:203`](../scripts/provision_box.sh:203) |
| RAM | ≥ 60 GB | [`:206`](../scripts/provision_box.sh:206) |
| Free disk at `/workspace` | ≥ what still has to land — 70 GB on a fresh box | [`:226`](../scripts/provision_box.sh:226) |
| Onstart | `env >> /etc/environment` | — |
| Env vars | see §11 | — |

The image's bundled torch is unused — `uv.lock` pins torch and its whole CUDA userspace as wheels, and that lock is the reproducibility anchor. The image supplies drivers, sshd, and a Python to bootstrap `uv` from. Its own pytorch is dead weight but cached on vast hosts, so it mostly costs nothing to pull.

**The onstart line is not optional.** SSH sessions load a fresh environment from `/etc/environment`, so anything passed via `--env` is invisible to your shell without it ([Cloudflare-independent; see vast's Docker environment docs](https://docs.vast.ai/guides/instances/docker-environment)). Skip it and provisioning dies at step 3/8 with `no corpus remote`.

---

## 4. Add the SSH alias

The instance's IP and port do not exist until now, which is why no script can pre-create this.

```bash
printf 'Host demae-box\n  HostName <IP>\n  User root\n  Port <PORT>\n  IdentityFile ~/.ssh/id_ed25519\n' >> ~/.ssh/config
```

**Required for teardown, optional for provisioning.** `teardown_box.sh` calls `ssh "$HOST"` with `$HOST` quoted as a single argument ([`:112`](../scripts/teardown_box.sh:112)), so a port cannot be passed inline — and vast never gives you port 22. Provisioning would accept a raw `ssh -p <port> root@<ip>` spec instead.

---

## 5. Push the code, then provision

`provision_box.sh` runs **on the box**, streamed over the SSH connection so it never lands on that disk as a file. It cannot copy your code itself — it has no access to your laptop, and giving a rented machine inbound credentials to your Mac would be strictly worse than pushing. So the push is a real prerequisite, and [`push_code.sh`](../scripts/push_code.sh) is it:

```bash
scripts/push_code.sh --host demae-box --provision
```

That is one command because R0 proved it had to be. This step used to be a three-line `&&` chain
typed from this document, whose first line generated `.git_sha`; the code went up without it and
a whole run recorded `git sha unknown` (#107). The sha is now regenerated and shipped by the same
command that ships the code, so there is no ordering left to get wrong. Pass provisioning's own
flags after `--`:

```bash
scripts/push_code.sh --host demae-box --provision -- --skip-corpus --skip-env
```

Use `--dry-run` to see the exact `rsync` and `ssh` it would run without touching the box.

**`.git_sha` is what puts a code version on every run.** `.git` is deliberately not in the
payload — at 128 MB it is the largest thing that would move, and shipping it would make the
answer *wrong* rather than merely expensive: `git status --porcelain` on the box sees every
unshipped path (`docs/`, `notebooks/`, `CONTEXT.md`) as deleted, so every run would stamp
`-dirty` from a clean commit. So the laptop computes the SHA where the history actually is and
ships the one-line answer. [`dataset.provenance.git_sha`](../src/dataset/provenance.py) asks git
first and reads `.git_sha` only when git cannot answer, which is why a stale copy in a real
checkout is inert.

It is generated through the project's own `git_sha()` rather than `git rev-parse` — that is what
carries the `-dirty` suffix, and a run from uncommitted code should say so. It is recomputed on
every push and never reused: a `.git_sha` from three pushes ago is a confident wrong answer,
which is worse than the `unknown` it replaced. For the same reason the script refuses to ship the
literal `unknown` — writing it would leave a *non-empty* file on the box, silencing provisioning's
step-6 warning while recording nothing. Provisioning step 6 prints the value it found and warns if
the file is missing.

A failed rsync never reaches provisioning. Three separate `die` paths exist for missing code — [`:241`](../scripts/provision_box.sh:241) (no `uv.lock`), [`:368`](../scripts/provision_box.sh:368) (no `check_corpus.py`), [`:385`](../scripts/provision_box.sh:385) (no `artifacts/`).

**`--delete` reaches only the directories being pushed.** rsync scopes deletion to each
transferred source, so `data/`, `runs/`, `.venv/` and `.box_identity` at the destination root
survive — which is what makes the push safe to re-run against a box that already holds a corpus.
Re-check that property before adding anything to the payload list.

Idempotent throughout: `rclone copy` re-transfers only what disagrees and every other step checks whether it is already done, so a run that dies on a flaky download resumes by re-running the same command rather than from zero. **This is now actually reachable** — until #107, preflight rejected any box that had already downloaded a corpus, which took the resume path and `--skip-corpus` with it. See §6.

Useful flags: `--skip-corpus` (verify what is already on disk), `--skip-env` (skip `uv sync`), `--probe-steps 0` (skip the throughput probe), `--recheck` (re-hash after the copy; moves no bytes), `--rclone-version` (which rclone to install; see §6).

---

## 6. Read the provision footer

The eight steps are: preflight → Python env → rclone and corpus copy → corpus completeness → committed artifacts → config/fingerprints → throughput probe → box identity.

**Step 1 asks what still has to land, not what the finished article occupies.** The old check
compared a flat 70 GB against total free space, and 70 GB is exactly what corpus + venv + a run's
own writes *add up to* — so the moment the corpus was down, the disk holding it failed the check
reserving room for it. R0's box (102 GB disk, 53 GB of it the corpus and venv provisioning had
itself put there, 50 GB free) could not be re-provisioned at all. The budget is now its three
parts, and only the remainder of each is required: 47 GB corpus minus whatever `du` finds on
disk, 8 GB venv (2 GB if `.venv` already exists), and a **15 GB reserve that is never subtracted**
— §11 retention keeps the last 3 checkpoints plus the 4 milestones, which is 7 files at ~1.2 GB
each for ViT-B, and those writes are always still ahead. On a fresh box the sum is 70 GB, i.e.
the constant it replaces: this loosened nothing for the case the old check was written for.

**Step 3 settles rclone before it looks at `--skip-corpus`**, because the thing an
already-provisioned box still needs rclone for is shipping checkpoints for the next 43 hours. The
vastai image ships `v1.60.1-DEV`, a 2022 build; against R2 every checkpoint upload failed
`NotImplemented (501)` on attempt 1 and succeeded on attempt 2, three for three in R0. The bytes
were never in doubt — a re-downloaded checkpoint hashed identically — but a healthy pass logging
three ERROR lines is how a real failure gets ignored.

Both of the issue's levers are pulled, and they are not equally well founded:

- **The pin.** Anything below the **1.65.0 floor** is replaced with a pinned **v1.69.1**, and the
  version lands in `.box_identity`. Note what is *not* claimed: no release was bisected, so this
  is a move off the 2022 series rather than a fix. What it buys regardless is that R1–R5 all ship
  through the same client and the run record says which.
- **`--s3-no-check-bucket`**, on every upload path (`sync_checkpoints.sh`, and teardown's default
  catch-up command). This one has a mechanism: rclone probes its destination bucket and will try
  to *create* it before the first upload, caching the result per process. The box holds an Object
  Read & Write token, which cannot create a bucket, and R2 may answer that with 501. Because the
  sync loop runs a fresh `rclone copy` per pass, that predicts exactly one failure per pass on
  attempt 1 — which is what R0 saw, three for three. Still a hypothesis: it was never confirmed
  on the box. It is safe either way, since the buckets are created by hand long beforehand.

**An old rclone warns; it does not block provisioning.** Absence is fatal — the corpus arrives
through rclone and the checkpoints leave through it — but age is not, because R0 proved an old
one ships a run, noisily, for 43 hours. Dying there would hand back what the step-1 fix above
just stopped doing: refusing to re-provision a box that has already paid for its corpus, over
something cosmetic. The hard gate is `sync_checkpoints.sh --once`, which a human runs before
committing GPU hours (§7).

**[verified locally]** steps 4, 5, and 6 pass against the real corpus on this laptop — `check_corpus.py` reports complete, all four `artifacts/phase1/` files are present, and `pretrain.launch --dry-run` resolves cleanly (92,400 steps, 231/epoch, containment assertion passes).

Step 7 is the one that does **not** transfer between hosts. The loader ≥2× gate is a fact about *this box's* NVMe, cores, and free RAM, not about the recipe. Before starting a real run, check the probe footer for:

- **loader wait > ~10%** — the ADR-0003 headroom argument does not hold on this host
- **recompiles** — §7.3 expects zero; any is a moving shape costing the whole run
- **attention** — must be the sdpa path the config asked for, not a silent fallback

The probe deliberately uses the ViT-S default rather than `--vit-base`: §12 plans ViT-S at ~3,000 img/s against ViT-B's ~1,300, so the small model is the *stricter* test. A box that feeds ViT-S feeds ViT-B; the converse does not follow.

Step 8 writes `.box_identity` — paste those values into the W&B run tags. When a throughput number looks wrong six weeks from now, the host needs to be attributable rather than reconstructed. It now carries `rclone=` alongside the GPU and RAM, for the same reason.

---

## 7. Start the run

Two tmux windows. Training first:

```bash
cd /workspace/DEMAE && tmux new -s r1
uv run python -m pretrain.launch --vit-base --set 'monitoring.wandb_mode="online"'
```

That one command is the whole run: §11's checkpoints on both triggers, §10's val pass, alarms,
per-patch distribution, reconstruction grids and kNN probe, and the W&B run they all go to. It
did not used to be — [#106](https://github.com/JStoschek/DEMAE/issues/106) composed it, and
before that this command called the plain training loop and produced none of them.

**Checkpoints go to `/workspace/DEMAE/runs/R1`** — `runs/<run_id>`, and `run_id` defaults to
`R1`. `--checkpoint-dir` moves it; `--set run_id=R2` moves it with the arm, which is what keeps
an ablation from resuming the baseline's weights. Read the `checkpoints` line the launcher
prints in its first ten lines rather than trusting this paragraph: it names the directory, both
triggers and the retention rule, and the sync loop below has to point at the same place.

Then the checkpoint sync, in its own window:

```bash
tmux new-window -t r1 -n sync
scripts/sync_checkpoints.sh --checkpoint-dir /workspace/DEMAE/runs/R1 \
  --remote bucket:demae-checkpoints/r1/
```

Prove the credentials first, before committing GPU hours — `--once` probes the remote even when no checkpoint exists yet:

```bash
scripts/sync_checkpoints.sh --checkpoint-dir /workspace/DEMAE/runs/R1 \
  --remote bucket:demae-checkpoints/r1/ --once
```

`--once` also **refuses to run** on an rclone below the 1.65.0 floor, and the loop only warns
about the same thing. That asymmetry is deliberate: `--once` is this pre-run probe, where nothing
is training and the fix is to re-run provisioning; the loop is started *during* a run, and exiting
to avoid some ERROR lines would end checkpoint shipping for the next 40 hours — the exact trade
the script exists to refuse.

**This loop is not optional.** §11 requires checkpoints off-instance every epoch and every 30–60 minutes, because vast instances vanish permanently rather than pause. Nothing in `src/pretrain` pushes off-instance — the launcher writes §11's checkpoints to local disk (since #106; before it, to nowhere) and `checkpoint.py` mentions syncing only in a comment, while teardown calls its own pass "the safety net, not the mechanism." Without this window, a box that disappears at hour 40 of a 43-hour run takes every checkpoint with it, not merely the last interval's worth.

**Check the first pass reports a file rather than `ok` over nothing.** The loop reports `ok` whether or not the directory holds anything, which is correct behaviour for a sync and was how the empty-directory case survived unnoticed until #106. One epoch in, `ls /workspace/DEMAE/runs/R1` should show a `checkpoint_e*.pt`.

The loop exits by itself once training has started and then stopped, after a final pass.

---

## 8. During the run

- Checkpoints land every epoch and every 30–60 min; retention keeps last 3 plus the 50/100/200/400 milestones.
- Every epoch prints one line — LR, train loss, val loss, samples/s, step time, loader wait — and §10.1's three alarms print `[STOP]` lines inline where the person watching will see them. Every 10 epochs the reconstruction grids render; every 20, the kNN probe.
- The sync loop prints a timestamped `ok` per pass. A persistent failure prints a rising consecutive-failure count rather than dying — exiting would guarantee the outcome it exists to prevent.
- **If the box vanishes, run the exact same command on the replacement.** Resume is the default: the launcher picks up the newest checkpoint in `runs/R1` and prints which one before spending anything. Pull the directory back from the bucket first — the box that vanished took its copy with it.
- `wandb_mode="online"` avoids offline run directories entirely. If you run offline instead, those directories are unique to the box until `wandb sync` ships them, and teardown will warn about them.

---

## 9. Teardown

Runs on the **laptop**, not the box. A teardown script that destroys the machine it is running on cannot report what it found, and would verify the bucket copy by asking the same box that wrote it. The check that means something is independent: this laptop, its own credentials, its own network path, re-reading the bytes the box claims it uploaded.

**Stop training first.** Step 1/7 refuses while a `pretrain.launch|train` process is alive, and says to come back later rather than to fix something — a checkpoint written 20 seconds after the inventory is one the script never verified and would then have destroyed.

```bash
uv run ./scripts/teardown_box.sh --host demae-box \
  --checkpoint-dir /workspace/DEMAE/runs/R1 \
  --bucket demae-checkpoints --prefix r1/ --profile r2 \
  --endpoint-url https://79083f36be71e95fad1b0b06d2bb62a1.r2.cloudflarestorage.com
```

Seven steps: alive check → inventory (sha256 on the box) → catch-up sync → haul small unique things → bucket presence (`head_object` from here) → deep verify (re-download and re-hash) → verdict.

**Report-only by default.** Read the verdict, then re-run with `--destroy`, which still makes you type the instance id back. A failed verification refuses to destroy no matter what flags are passed. Destroying is permanent — vast container storage is deleted with the instance, not archived.

There is no pause anywhere in this workflow. Teardown is the destroy path.

---

## 10. After

```bash
wandb sync ./teardown-<instance>/wandb/offline-run-*
```

`teardown-<instance>/` also holds `box_identity` — the host axis your W&B run tags want.

---

## 11. Reference: the rent-time env block

Fourteen variables. Two rclone remotes, because provisioning reads the corpus through `demae:` and teardown's default sync command writes checkpoints through `bucket:` ([`teardown_box.sh:155`](../scripts/teardown_box.sh:155)) — a remote name documented nowhere else.

```
RCLONE_CONFIG_DEMAE_TYPE=s3
RCLONE_CONFIG_DEMAE_PROVIDER=Cloudflare
RCLONE_CONFIG_DEMAE_REGION=auto
RCLONE_CONFIG_DEMAE_ENDPOINT=https://79083f36be71e95fad1b0b06d2bb62a1.r2.cloudflarestorage.com
RCLONE_CONFIG_DEMAE_ACCESS_KEY_ID=<R2 key id>
RCLONE_CONFIG_DEMAE_SECRET_ACCESS_KEY=<R2 secret>
RCLONE_CONFIG_BUCKET_TYPE=s3
RCLONE_CONFIG_BUCKET_PROVIDER=Cloudflare
RCLONE_CONFIG_BUCKET_REGION=auto
RCLONE_CONFIG_BUCKET_ENDPOINT=https://79083f36be71e95fad1b0b06d2bb62a1.r2.cloudflarestorage.com
RCLONE_CONFIG_BUCKET_ACCESS_KEY_ID=<R2 key id>
RCLONE_CONFIG_BUCKET_SECRET_ACCESS_KEY=<R2 secret>
DEMAE_CORPUS_REMOTE=demae:demae-corpus/corpus_500k
WANDB_API_KEY=<W&B key, from wandb.ai/authorize>
```

### W&B

The fourteenth variable is the only one that is not about storage, and it is here for the same
reason as the rest: **it cannot be added to a box that is already running.** vast passes `--env`
at instance creation, so a box rented without it needs `uv run wandb login` inside the container
instead — which writes `/root/.netrc`, which dies with the instance. Put the key in the env block
and every future box logs itself in.

Project **`demae-phase1`**, matching the default at [`config.py:603`](../src/pretrain/config.py:603), so no run passes a project flag. **[verified] 2026-08-03** — entity `jonathan-stoschek-purdue-university`, project reachable, zero runs; R0 is the first.

`monitoring.wandb_entity` is **pinned to `jonathan-stoschek-purdue-university`** at
[`config.py:612`](../src/pretrain/config.py:612), so no run passes an entity flag either. It used
to be `None`, meaning "the key's current default entity" — a mutable account setting rather than a
fact about this project. R0 pinned it through a config file for its own run; #107 moved that into
the repo. The whole of §10's decision rule reads R0–R5 against each other, and a default that
shifts mid-phase files the later arms under a different entity with nothing in the run record
saying so. It is an `ENVIRONMENT` field, so the pin cost no `config_hash` change and cannot make a
§11 resume look like a different experiment — which is also why it was safe to change *after* R0
had already run. A fork of this repo wants `None` back.

Offline mode needs none of this — it is the default and the tested path, which is why #97 never
waited on this section.

### R2 specifics

**No script changes are needed for R2.** All three bucket-touching code paths auto-detect `r2.cloudflarestorage.com` and swap the SigV4 signing region to the literal `auto` — [`check_corpus.py:130`](../scripts/check_corpus.py:130), [`teardown_box.sh:196`](../scripts/teardown_box.sh:196), [`teardown_box.sh:263`](../scripts/teardown_box.sh:263). Get it wrong and R2 raises `SignatureDoesNotMatch`, which reads like a wrong secret key.

**Never export `AWS_REGION` to a real AWS region.** The swap only fires when the region is still the default `us-east-1`. Setting it to, say, `us-west-2` silently disables the R2 handling.

**R2 has no object versioning — use a bucket lock instead.** The script headers say the checkpoint key's safety "only holds with versioning enabled." That is a B2/S3 statement with no R2 equivalent: R2 does not implement versioning, and [deletion is irreversible](https://developers.cloudflare.com/r2/objects/delete-objects/). Do not go looking for the setting; it is not there.

R2's mechanism is [bucket locks](https://developers.cloudflare.com/r2/buckets/bucket-locks/) — retention rules that prevent **both deletion and overwriting**, for a period or indefinitely. **R2 → `demae-checkpoints` → Settings → Bucket lock rules → Add rule**, or `npx wrangler r2 bucket lock add demae-checkpoints`. This is a closer fit than versioning would have been: it refuses the destructive operation rather than keeping a copy behind it, which is the "cannot remove anything" property the headers actually wanted.

Set a **time-based** rule, prefix `r1/`, retention outliving the run plus the analysis window — 90 days is a reasonable default. Avoid indefinite: checkpoints are large and you could never clean them up. Up to 1,000 rules per bucket; where several apply, the longest retention wins.

**Two subsystems, opposite granularity — do not conflate them.** API *tokens* cannot scope below a bucket. Lock *rules* can take a key prefix. The token limitation is why the corpus and checkpoint keys cannot be separated within one bucket; the lock capability is why protection can still be aimed at `r1/` alone.

**The box cannot lift the lock.** Editing bucket configuration is an Admin-tier operation, and the box holds an Object Read & Write token, which reaches objects only. A leaked box key therefore cannot remove the rule and then delete — which is the threat this exists for.

**Interaction to watch:** a lock blocks overwrites as well as deletes, so re-uploading an already-shipped checkpoint *filename* would fail. Normal operation never does this — names carry the step number so syncing only adds keys, and `rclone copy --checksum` skips files whose hashes already match. It bites only if a resumed run rewrites an existing filename with different bytes.

**Accepted risk:** a single Object Read & Write token covers both buckets, so a compromised box can overwrite the corpus. Judged acceptable — the corpus is byte-identical on local disk, so the downside is a 50 GB re-upload rather than data loss. The irreplaceable asset is the checkpoints, and the lock covers those.

---

## 12. Known gaps

| Gap | Status |
|---|---|
| The documented start command wrote no checkpoints and opened no W&B run | **closed 2026-08-03 (#106).** `pretrain.launch` called the plain training loop, so a 400-epoch R1 launched from §7 would have produced no checkpoints for `sync_checkpoints.sh` to ship, no W&B run for §15.7's tags to hang off, and none of §10's instruments — while every pass of the sync loop reported `ok`. The launcher now composes #94's resumable loop with #97's monitor; `python -m pretrain.monitor` is gone as a second training command, and with it a `--set` that was parsed and discarded |
| No continuous checkpoint sync in `src/pretrain` | closed by `sync_checkpoints.sh`; the plan's §11 wording still implies the trainer does it |
| `bucket:` rclone remote undocumented in provision's `--help` | documented in §11 above; the script's own help is still incomplete |
| Both script headers say the checkpoint key is safe "with versioning enabled" on R2 | **factually wrong — R2 has no object versioning.** [`provision_box.sh:23`](../scripts/provision_box.sh:23) and [`teardown_box.sh:22`](../scripts/teardown_box.sh:22) still carry it. §11 records the real mechanism (bucket locks); the headers need correcting |
| R2 cannot express the two-scoped-key design in the script headers | headers describe B2 semantics; §11 above records the R2 reality |
| Runs recorded `git sha unknown` — the box has no `.git` and §5's rsync did not ship one | **closed 2026-08-03 (#107).** The mechanism landed with #80 and the file still did not reach R0's box, because generating it was a line in this document. `push_code.sh` now regenerates and ships it in the same command as the code, and refuses to write the literal `unknown`. Step 6 still prints the value and warns rather than dies |
| Preflight's `MIN_FREE_GB=70` is checked against total free space, so it fails on any already-provisioned box | **closed 2026-08-03 (#107).** Step 1/8 now sums what still has to land — corpus remainder by `du`, venv (or a re-sync's worth), and a 15 GB reserve that is never subtracted. Reduces to 70 GB on a fresh box, so §5's resume path and `--skip-corpus` are reachable without loosening the original check. Arithmetic pinned in `tests/test_ops_scripts.py` |
| rclone 501s on the first upload attempt of every checkpoint | **closed 2026-08-03 (#107)** at the level the evidence supports. Step 3 replaces anything below 1.65.0 with a pinned v1.69.1 and records it in `.box_identity`; every upload path now passes `--s3-no-check-bucket`; `sync_checkpoints.sh` names the symptom when it sees an old client, fatally under `--once`. **Neither lever is confirmed on the box** — the flag has a mechanism that fits R0's evidence, the pin does not, and R5's ~1.2 GB checkpoints (4× what R0 tested) are still unexercised at any version. The first R1 sync pass is what settles it: zero ERROR lines means done |
| `sync_checkpoints.sh` tested only against a local rclone remote | its logic is verified, including the prune-does-not-propagate property; behavior against R2 is not. The `--once` probe on the box closes this in seconds |
