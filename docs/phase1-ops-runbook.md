# DEMAE Phase 1: Ops Runbook

**Scope:** everything between "I have a plan and a budget" and "the box is gone and the run survived it." The recipe itself lives in [`phase1-pretraining-plan.md`](phase1-pretraining-plan.md); this is the operational wrapper around it.
**Hardware:** single rented GPU, vast.ai. **Store:** Cloudflare R2, account `79083f36be71e95fad1b0b06d2bb62a1`.
**Scripts:** [`push_code.sh`](../scripts/push_code.sh), [`provision_box.sh`](../scripts/provision_box.sh), [`run.sh`](../scripts/run.sh), [`sync_checkpoints.sh`](../scripts/sync_checkpoints.sh), [`watch_run.sh`](../scripts/watch_run.sh), [`teardown_box.sh`](../scripts/teardown_box.sh), [`check_corpus.py`](../scripts/check_corpus.py).
**Status:** v4, 2026-08-04 — §7's start command is [`run.sh`](../scripts/run.sh) rather than two commands that only worked when typed by hand (#108), and the sync loop now leaves evidence that it started. v3 (2026-08-04) closed the idle-billing gap R1 exposed with `watch_run.sh` (#82); v2 was 2026-08-03 (#107's four R0 gaps), v1 2026-07-31. Steps marked **[verified]** were run end-to-end on this laptop; steps marked **[untested]** need a live box.

---

## 0. At a glance

| # | Step | Where | Script |
|---|---|---|---|
| 1 | Verify the corpus in R2 | laptop | `check_corpus.py --bucket` |
| 2 | Rent the instance | vast console | — |
| 3 | Add the SSH alias | laptop | — |
| 4 | Push code, then provision | laptop → box | `push_code.sh --provision` |
| 5 | Read the provision footer | — | — |
| 6 | Start the sync loop and training | box, tmux | `run.sh --run-id R1` |
| 7 | Let it run, and be told when it stops | laptop | `watch_run.sh` |
| 8 | Verify, then destroy | laptop | `teardown_box.sh` |
| 9 | Sync the hauled W&B runs | laptop | `wandb sync` |

Nothing here rents or destroys automatically. Both spend money or delete data and stay a human's to run — see [`provision_box.sh:26`](../scripts/provision_box.sh:26).

---

## 1. One-time laptop setup

Done once, then never again.

| Item | State |
|---|---|
| `vastai` CLI on PATH | done — `uv tool install vastai`, 1.5.1. **It lands in `~/.local/bin`, which a non-interactive shell does not have on `PATH`** — teardown looks there explicitly since #111, but anything else you run by hand may not |
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

One command, on the box, **by absolute path**:

```bash
/workspace/DEMAE/scripts/run.sh --run-id R1
```

Each arm passes its own recipe after `--`, and nothing else changes:

```bash
/workspace/DEMAE/scripts/run.sh --run-id R5 -- --vit-base
```

**R2 and R3 are done** ([#83](https://github.com/JStoschek/DEMAE/issues/83), [#84](https://github.com/JStoschek/DEMAE/issues/84)) and their examples are retired. Keep the trap they exposed, because the next `--set` arm will hit it: they read `--set masking.ratio=0.9` and `--set loss.norm_pix_loss=false` and **carried the knob but not the schedule**, so followed literally they would each have run 400 epochs against §12's planned 200. R2 caught it at launch and set `schedule.epochs=200` with `warmup_epochs=10` (§6's warmup rule is *5% of the schedule*, never the literal 20). **An arm changes a knob and a schedule, or it is not the arm the plan costed.**

**R4 has no one-liner yet, and that is [#86](https://github.com/JStoschek/DEMAE/issues/86)'s to settle.** `RunConfig.vit_base_continual()` carries the arm's four knobs — including `pos_encoding='sincos2d_rowfirst'`, which [P102](research/p102-mmselfsup-mae-layout.md) showed is silently catastrophic to get wrong — but it is **not reachable from the CLI**: [`header.py:153`](../src/pretrain/header.py#L153) offers only the default preset and `--vit-base`. R4 must therefore go through `--config`, or the preset needs a flag. The loader refuses a model whose positional buffer disagrees with the checkpoint's own `pos_embed`, so a wrong path raises rather than trains — but a documented command should exist before the box is rented, not after.

`--dry-run` prints the whole plan — every command, in order — and touches nothing.

**[verified] 2026-08-04, on a live RTX 5090 box** (instance 46808051), start to finish. The bug
reproduced first: a pane created with `tmux new-session -c /workspace/DEMAE` reported
`pane_current_path` `/workspace`, and `scripts/sync_checkpoints.sh` from it answered `No such
file or directory`. Then, through `run.sh` from a pane sitting in `/workspace`: the probe
authenticated, the sync loop came up and wrote its heartbeat, training ran 250 steps at ~5,840
samples/s (loader wait 1.1% steady), wrote `checkpoint_e0001_s000000231.pt`, and the loop
shipped it to the bucket, saw training exit, made its final pass and closed itself. The
laptop-side watcher armed against the same run and reported `R9 FINISHED cleanly` off the log
`run.sh` tees.

Both failure paths were exercised on the same box, not just the working one. With
`sync_checkpoints.sh` moved aside, the run stopped at the heartbeat gate **in one second** — the
dead pane's `status 127` printed, the session killed, no GPU time spent. And with the sync window
killed *under* a healthy run, the watcher reported `SYNC STALLED — no heartbeat for 1 min` from
the laptop, with the command to restart that window, while training carried on normally.

**Why this is a script and not the two commands this section used to print.** Both of them were
correct, and neither ran ([#108](https://github.com/JStoschek/DEMAE/issues/108)). The version
here through v3 was `cd /workspace/DEMAE && tmux new -s r1`, and the vastai image's shell rc
`cd`s every new pane to `/workspace` — overriding even `tmux new-session -c /workspace/DEMAE`,
confirmed on R1's box where `tmux display-message -p '#{pane_current_path}'` answered
`/workspace` for a pane created with `-c`. Typed by hand into an already-attached pane it
happened to work. Driven any other way — `send-keys`, a detached session, a script, an agent —
the two halves broke differently, and the asymmetry is the whole point:

- **Training failed loudly.** `uv run` found no `pyproject.toml`, fell back to the image's
  interpreter, and died with `ModuleNotFoundError: No module named 'pretrain'`.
- **The sync loop failed silently.** `scripts/sync_checkpoints.sh: No such file or directory` —
  one line, in a window nobody is watching, with no retry and no rising failure count, because
  the loop that prints those never started.

R1 escaped it only because the loud half failed at the same instant and forced a look. Had the
cwd been right for `uv run` and wrong for the relative path — or had the two windows been started
separately, which is what this section told you to do — R1 would have trained 7.2 hours with no
checkpoint ever leaving the box. This is [#106](https://github.com/JStoschek/DEMAE/issues/106)'s
shape on a third axis: #106 was the *launcher* composing no sync, this was the *documented
command* not starting one. A paragraph cannot be tested, and this whole class of bug is
"documented correctly, executed differently" — so the fix is a committed script, and every
property that failed is pinned in `tests/test_ops_scripts.py`.

What [`run.sh`](../scripts/run.sh) does, in order:

1. **Resolves every path from its own location**, never from the caller's cwd. That is a fact
   about the checkout; a cwd is a fact about somebody's shell.
2. **Probes the remote** — `sync_checkpoints.sh --once`, the gate below, which costs seconds.
   `--skip-probe` if it has just been run.
3. **Starts the sync loop first**, and waits for its heartbeat. The old ordering was the other
   way round, so the cheap half's failure arrived after the expensive half was already spending.
4. **Refuses to start training** if that heartbeat does not appear (120 s default,
   `--heartbeat-wait`). It prints the sync window's output, kills the session, and stops.
5. **Starts training**, tee'd to `/workspace/DEMAE/r1_train.log` — which is exactly where
   `watch_run.sh` looks (§8). That default has always been there; until #108 nothing wrote it.

**One run id, threaded everywhere:** the config's `run_id`, `runs/R1`, the bucket prefix `r1/`,
the log, the tmux session. This section used to name the run four times across two commands and
got one of them wrong — it passed `--vit-base` in the R1/R2/R3 position, and
[`RunConfig.vit_base()`](../src/pretrain/config.py:753) sets `run_id="R5"`, so following it
literally for R1 would have trained a ViT-B into `runs/R5` while the sync window shipped
`runs/R1`. `--run-id` now moves all of them together, and a `--set run_id=` after `--` is
refused rather than allowed to move the config alone.

Read the `checkpoints` line the launcher prints in its first ten lines rather than trusting this
paragraph: it names the directory, both triggers and the retention rule, and it has to be the
directory the sync window was pointed at.

**The sync loop is not optional.** §11 requires checkpoints off-instance every epoch and every
30–60 minutes, because vast instances vanish permanently rather than pause. Nothing in
`src/pretrain` pushes off-instance — the launcher writes §11's checkpoints to local disk (since
#106; before it, to nowhere) and `checkpoint.py` mentions syncing only in a comment, while
teardown calls its own pass "the safety net, not the mechanism." Without that window, a box that
disappears at hour 40 of a 43-hour run takes every checkpoint with it, not merely the last
interval's worth.

**The heartbeat is what makes "never started" a different observation from "started and found
nothing".** The loop writes `runs/R1/.sync_heartbeat` — one line, a UTC stamp, the pass number,
the remote — as soon as it is up and again after every pass, including the passes that find an
empty directory and the ones that fail. An empty run directory early in a run is the *normal*
state, so silence there could never be read as health. `--once` deliberately writes none: the
probe proving the credentials is not the loop proving it started, and R1 ran the probe
successfully while the loop it was meant to bless never came up. The file is a dotfile and the
include pattern is `checkpoint_*.pt`, so it never reaches the bucket.

**Check the first pass reports a file rather than `ok` over nothing.** The loop reports `ok`
whether or not the directory holds anything, which is correct behaviour for a sync and was how
the empty-directory case survived unnoticed until #106. One epoch in, `ls
/workspace/DEMAE/runs/R1` should show a `checkpoint_e*.pt`.

The loop exits by itself once training has started and then stopped, after a final pass.

**If `run.sh` is not there** — an old push, or a box whose `scripts/` predates it — the two
halves by hand, absolute throughout, are:

```bash
tmux new-session -d -s r1 -n sync /workspace/DEMAE/scripts/sync_checkpoints.sh --checkpoint-dir /workspace/DEMAE/runs/R1 --remote bucket:demae-checkpoints/r1/
tmux new-window -t r1: -n train "cd /workspace/DEMAE && uv run python -m pretrain.launch --set run_id=R1 --set monitoring.wandb_mode=online 2>&1 | tee -a /workspace/DEMAE/r1_train.log"
```

Two long lines rather than four wrapped ones, on purpose: a `\`-continued command pasted into a
pane is one more thing that behaves differently depending on how it is delivered, which is the
bug above. Nothing relative, nothing that assumes a pane's cwd, the run id said explicitly, and
the `cd` inside the quotes rather than before them. Check `runs/R1/.sync_heartbeat` exists before
walking away — started this way, nothing else will. Fix the push instead if you can: `scripts/`
is in `push_code.sh`'s payload, so `run.sh` ships with every push.

`--once` **refuses to run** on an rclone below the 1.65.0 floor, and the loop only warns about
the same thing. That asymmetry is deliberate: `--once` is the pre-run probe, where nothing is
training and the fix is to re-run provisioning; the loop is started *during* a run, and exiting
to avoid some ERROR lines would end checkpoint shipping for the next 40 hours — the exact trade
the script exists to refuse.

---

## 8. During the run

- Checkpoints land every epoch and every 30–60 min; retention keeps last 3 plus the 50/100/200/400 milestones.
- Every epoch prints one line — LR, train loss, val loss, samples/s, step time, loader wait — and §10.1's three alarms print `[STOP]` lines inline where the person watching will see them. Every 10 epochs the reconstruction grids render; every 20, the kNN probe.
- The sync loop prints a timestamped `ok` per pass. A persistent failure prints a rising consecutive-failure count rather than dying — exiting would guarantee the outcome it exists to prevent.
- **If the box vanishes, run the exact same command on the replacement.** Resume is the default: the launcher picks up the newest checkpoint in `runs/R1` and prints which one before spending anything. Pull the directory back from the bucket first — the box that vanished took its copy with it.
- `wandb_mode="online"` avoids offline run directories entirely. If you run offline instead, those directories are unique to the box until `wandb sync` ships them, and teardown will warn about them.

**Be told when it ends, from the laptop:**

```bash
scripts/watch_run.sh --host demae-box --run-id R1
```

It reads `/workspace/DEMAE/r1_train.log` by default — `<root>/<run-id lowercased>_train.log`, which is where [`run.sh`](../scripts/run.sh) tees the trainer. That default has always been here; until #108 nothing wrote the file, so a watcher pointed at a perfectly healthy run read an empty log and could never see the completion line it keys `finished` on. `--log` still overrides it.

One line per event on stdout, so it drives a notifier or a `| while read` loop; silence means healthy. It emits a milestone line every 50 epochs and exactly one terminal event — `finished` (0), `died` (1), or `unreachable` (2) — and those three are deliberately **not** interchangeable. A vanished box, a crashed trainer and a clean finish all look identical from the process table, so `finished` is keyed on the summary line `pretrain.launch` prints at [`launch.py:339`](../src/pretrain/launch.py:339) rather than on the *absence* of a process. An unreachable box is never reported as a finish: on vast that state means hurry, not done.

**Prove it on the box first, before trusting its silence** — seconds, no GPU risk, exactly like `sync_checkpoints.sh --once`:

```bash
scripts/watch_run.sh --host demae-box --run-id R1 --once
```

That runs the self-test and one poll and exits, so a watcher that cannot see the run says so while you are still at the keyboard, rather than by staying quiet for eight hours. **Run this every time; the plumbing is per-box.**

**It also watches the sync loop, not just the trainer.** `run.sh` gates on the heartbeat once, at
startup, which closes #108's failure — the loop that never came up. A loop that dies at hour 3
looks identical from the training side, and this is the only process observing the box from
somewhere the box cannot take with it. So when `runs/R1/.sync_heartbeat` goes older than
`--sync-stale-minutes` (default 90 — two of the loop's own 30-minute passes plus room for a slow
upload; 0 disables), it prints `SYNC STALLED` with the command to restart that window, and keeps
watching. It is deliberately **not** a terminal state: the run is worth finishing and the
checkpoints are still on the box. It says so once, and says so again when the heartbeat recovers.

`--max-hours` (default 24) is the backstop for the case where every other check is wrong: it fires on the clock alone, having observed nothing, so a run that *measures* as healthy forever still ends the watch. R5 at 400–800 epochs needs an explicit larger value — and note it reports `OVERDUE`, never a completion.

**It refuses to arm until it has proven its own liveness check works, in both directions** — that a sentinel process which cannot exist reads as absent, and that the run you pointed it at reads as present. R1 (#82) is why. Its watcher asked the box `pgrep -fc "pretrain.launch"`, and `pgrep -f` matches full command lines *including ssh's own*, so the count could never reach zero, the completion branch never fired, and the box billed for 7.6 idle hours (~$3.05, more than R1's own $2.90 of GPU) before a human noticed. Nothing was lost but the money. The generalisable part is why the self-test is fatal rather than a warning: **a check whose failure mode is silence needs a positive test that it can fire at all**, because a watcher that cannot observe completion is indistinguishable from a run that has not completed — for as long as you keep paying.

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

Seven steps: alive check → inventory (sha256 on the box) → catch-up sync → haul small unique things → bucket presence (`head_object` from here) → deep verify (re-download and re-hash) → verdict. Under `--destroy` an eighth follows the verdict: destroy, then prove the instance is gone.

**Report-only by default.** Read the verdict, then re-run with `--destroy`, which still makes you type the instance id back. A failed verification refuses to destroy no matter what flags are passed. Destroying is permanent — vast container storage is deleted with the instance, not archived.

There is no pause anywhere in this workflow. Teardown is the destroy path.

### The destroy is verified, because it once was not (#111)

**After R2/R3, `teardown_box.sh --destroy --yes` printed `instance 46808051 destroyed.` while the instance was still running and still billing.** Both lines appeared together, which is the only reason it was caught:

```
Aborted.
instance 46808051 destroyed. ./teardown-46808051 holds everything that was unique to it.
```

**There are two confirmations and `--yes` only ever skipped one.** `--yes` bypasses *this script's* "type the instance id" question; the vastai CLI has a prompt of its own, and with no tty on stdin it read EOF, printed `Aborted.`, and **exited 0**. Nothing checked the status, so the success line printed unconditionally.

That is the dangerous shape rather than a cosmetic bug: every other failure in the script refuses to proceed and says nothing was destroyed, while this one **claimed the irreversible action had succeeded when it had not** — so the natural next move, closing the terminal and no longer watching the spend, was exactly wrong.

What changed:

- **`--yes` now covers both confirmations** — the script passes `vastai destroy instance -y`.
- **The exit status is not the verdict.** After destroying, the script re-reads the account's instance list and asserts the id is gone. Same principle as step 5's independent bucket check: the actor's own report of its work is not evidence of the work.
- **The list is proved able to see the instance *before* the destroy**, so that "absent afterwards" means something. A listing that came back empty for any other reason — wrong API key, changed output format — would otherwise read as "gone".
- **`destroyed.` is printed on one path only.** If the id is still listed the script fails loudly and prints the command to run by hand; if the list cannot be read at all it says so and claims nothing either way.
- **The `vastai` CLI is now checked at startup**, not after the deep verify has re-downloaded ~900 MB — and it is looked for in `~/.local/bin` as well as on `PATH`, because that is where the `uv tool install vastai` in §1 puts it and a non-interactive shell does not have that directory. This fired on a correctly provisioned laptop.

**Still true, and worth keeping in mind:** the verdict line is a report, not proof. Confirm with `vastai show instances` — or the [console](https://console.vast.ai/instances/) — before you stop watching the spend.

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
| Nothing told the operator a run had ended, so a finished box kept billing | **closed 2026-08-04 (#82).** R1 finished at ~07:17 UTC and was destroyed at ~14:53 — 7.6 idle hours, ~$3.05, more than the run's own $2.90. The watcher's liveness check was `pgrep -fc "pretrain.launch"` run over ssh, which matched the ssh command line carrying the pattern and so could never return zero. [`watch_run.sh`](../scripts/watch_run.sh) replaces it: the bracket form `[p]retrain.launch` cannot match itself, `finished` is keyed on the launcher's completion line rather than process absence, `unreachable` is its own state, and it refuses to arm until it has proven both directions of its own check on the live box. Pinned in `tests/test_ops_scripts.py`, including that the naive pattern *does* self-match |
| §7's start command landed in the wrong directory, and its sync half failed silently | **closed 2026-08-04 (#108).** The image's rc `cd`s every new pane to `/workspace`, overriding `tmux new-session -c /workspace/DEMAE`, so §7's two commands worked only when typed by hand into an attached pane: `uv run` died loudly with `ModuleNotFoundError`, and `scripts/sync_checkpoints.sh` — a relative path — printed one "No such file or directory" and never started, with no retry and no rising failure count. [`run.sh`](../scripts/run.sh) replaces both: absolute throughout, sync started *first* and required to prove itself alive before any GPU time, one `--run-id` threaded through the config, the directory, the bucket prefix, the log and the session. The loop now writes `.sync_heartbeat` so "never started" and "started and found nothing" are different observations, and `--once` deliberately writes none. Pinned in `tests/test_ops_scripts.py`, including that §7's own fenced commands hold no relative script path — and **verified on a live box**, both the working path and the refusal |
| §7 documented `--vit-base` in the R1/R2/R3 position | **closed 2026-08-04 (#108).** `RunConfig.vit_base()` sets `run_id="R5"`, so following §7 literally for R1 would have trained a ViT-B into `runs/R5` while the sync window shipped `runs/R1`. R1 used the bare default and escaped it. `run.sh --run-id` now sets the config's run id explicitly on every run, refuses a second `run_id=` in the passthrough args, and warns when `--vit-base` is attached to another arm; a test refuses any `--vit-base` example in this document that does not name R5 |
| `watch_run.sh` could never arm: its self-test read every zero count as two lines | **closed 2026-08-04**, found on a live box while verifying #108. `pgrep -fc` and `grep -c` both print their count *and* exit non-zero when it is zero, so the `\|\| echo 0` guard — there for the different failure of the command being absent — fired on the ordinary empty case too and the answer arrived as `0\n0`. Every comparison against `"0"` then read false, in both directions: the self-test decided its impossible sentinel was running and refused to arm as if #82 were back, and `completed_marker` decided the completion line was present, which reports a **crashed** run as `finished`. One parse function now, with anything non-numeric still meaning "unknown" rather than zero. Pinned in `tests/test_ops_scripts.py`; the armed watcher then reported a real run's finish from the laptop |
| `teardown_box.sh --destroy --yes` reported an instance destroyed while it was still running and billing | **closed 2026-08-05 (#111).** Two confirmations, and `--yes` only ever skipped one: the vastai CLI has a prompt of its own, which with no tty read EOF, printed `Aborted.`, and **exited 0**. Unchecked, so `instance 46808051 destroyed.` printed one line below `Aborted.` — the only reason it was caught. The dangerous shape, not a cosmetic one: every other failure in the script refuses to proceed and says nothing was destroyed, while this one claimed the irreversible action had succeeded, making "stop watching the spend" the natural next move. `--yes` now passes `-y` to the CLI, **and the exit status is no longer the verdict** — the account's instance list is re-read and the id asserted gone, having first been proved visible *before* the destroy so that absence afterwards means something. `destroyed.` is reachable on that one path; a still-listed id fails loudly with the by-hand command, and an unreadable list claims nothing either way. Pinned in `tests/test_ops_scripts.py`, including the original bug: a CLI that exits 0 having destroyed nothing must not produce `destroyed.` |
| `--destroy` checked for the `vastai` CLI after the deep verify, and missed `~/.local/bin` | **closed 2026-08-05 (#111).** A one-line precondition ran after all seven steps including a ~900 MB re-download, and fired on a *correctly provisioned* laptop, because `uv tool install vastai` lands the binary in `~/.local/bin` — a directory a non-interactive shell does not have on `PATH`. The check moved to startup and now looks there explicitly |
| `sync_checkpoints.sh` tested only against a local rclone remote | its logic is verified, including the prune-does-not-propagate property; behavior against R2 is not. The `--once` probe on the box closes this in seconds |
