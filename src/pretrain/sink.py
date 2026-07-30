"""The metrics sink — where every number, histogram and grid a run produces goes (#97).

Plan §11 asks for "W&B (or equivalent) for all metrics and the recon grids, **so the box holds
nothing unique except the latest shard of checkpoints**", and §15.7 asks for the run to be
tagged with four content hashes. This module is both, plus the seam the rest of the monitoring
harness writes through.

## Why this is a payload-typed interface and not `log(dict[str, float])`

Three tickets write into it, and only one of them has scalars:

- :mod:`pretrain.monitor` (#97) — per-epoch losses, the throughput panel, and §4's per-patch
  loss **distribution**, which is a histogram and not a mean.
- #99's kNN probe — scalar trends, on a 20-epoch cadence.
- #98's reconstruction grids — 40 rendered **images** every 10 epochs, plus hillshades.

A seam shaped for scalars would send #98 around it, and then the grids would reach W&B by a
second path with its own step axis, its own tags and its own idea of what a run is. So the unit
of logging here is a mapping of name to *payload*, where a payload is a number, a
:class:`Histogram` or an :class:`Image`, and the W&B-specific translation of each lives in one
place — :meth:`WandbLogger._translate` — rather than at three call sites.

## The step axis is the epoch

W&B has one global step counter per run, so #98's 10-epoch cadence and #99's 20-epoch cadence
have to agree on what a step *is*. It is the epoch, everywhere, and going backwards raises here
rather than being dropped silently the way W&B drops it. Two callers logging different keys at
the same epoch is the normal case and merges into one point.

## Offline is the default, and it is the test target

Plan §10's whole logging path is built against W&B's **offline** mode: a local run directory,
no account, no network, `wandb sync` later. That is a deliberate decoupling from #80 (the ops
ticket that provisions the project and the credentials) — making that an edge would serialize a
code ticket behind an account setup — and it is also the better test target, because the
translation, the tags and the step axis are all exercised without anyone logging in. Online is
the same code with :attr:`Monitoring.wandb_mode <pretrain.config.Monitoring>` moved, and R0
(#81) is where a real project first receives a run.

``disabled`` returns a :class:`MemoryLogger` and never imports wandb at all, so a box with the
package missing still trains and still has a readable in-process record of what it logged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Self

import numpy as np

from .config import RunConfig
from .fingerprint import DataFingerprint, fingerprint_for
from .header import run_record

#: W&B's own ceiling on a tag. The four §15.7 tags are a 40-hex git SHA (plus a possible
#: ``-dirty``), a 16-hex config hash and two 32-hex md5s, all prefixed — comfortably under it.
#: Written down rather than left implicit because a tag W&B rejects is a tag that is simply not
#: there, and `test_a_tag_fits_in_what_wandb_will_accept` is what holds the margin: the check
#: belongs in a test rather than in :func:`run_tags`, which must not refuse to open a run over
#: a cosmetic limit.
MAX_TAG_CHARS: Final = 64


# --- The payloads -------------------------------------------------------------


@dataclass(frozen=True)
class Histogram:
    """A binned distribution — §4's per-patch loss log, and anything else shaped like it.

    Deliberately plain data rather than an array: the accumulation that produces it
    (:class:`pretrain.monitor.LossHistogram`) runs over a whole val pass and has to be mergeable
    across batches, and what reaches the sink is the finished counts. Passing a raw sample here
    instead would mean either holding every per-patch value of a 10k-tile pass in memory (~2M
    floats) or letting the sink decide the bins, and the bins are a measurement decision.
    """

    counts: tuple[int, ...]
    edges: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.edges) != len(self.counts) + 1:
            raise ValueError(
                f"{len(self.counts)} counts need {len(self.counts) + 1} edges to sit between, "
                f"got {len(self.edges)}"
            )
        if len(self.counts) == 0:
            raise ValueError("a histogram with no bins is not a distribution")


@dataclass(frozen=True)
class Image:
    """One rendered image — #98's masked/predicted/truth triptychs and their hillshades.

    ``data`` is ``(H, W)``, ``(H, W, 3)`` or ``(H, W, 4)``, either ``uint8`` or a float array
    already in ``[0, 1]``. That is **renderable data, not elevation**: #98 owns inverting a
    reconstruction to metres and mapping metres onto a colour ramp, and the range check here is
    what keeps that mapping from being done twice or not at all.

    Non-finite values are refused rather than clamped, and the reason is specific to #98's own
    acceptance criterion: the grids have to render on a *randomly initialized* model, whose
    predictions can be anything. A NaN that reaches W&B renders as a black pixel and reads as a
    reconstruction failure — a model problem, in a panel built to diagnose model problems —
    when it is a renderer bug. Loud at the seam is the only place it is cheap to find.
    """

    data: np.ndarray
    caption: str | None = None

    def __post_init__(self) -> None:
        data = np.asarray(self.data)
        if data.ndim not in (2, 3) or (data.ndim == 3 and data.shape[2] not in (3, 4)):
            raise ValueError(
                f"an image has shape (H, W), (H, W, 3) or (H, W, 4), got {data.shape}"
            )
        if data.dtype == np.uint8:
            return
        if not np.issubdtype(data.dtype, np.floating):
            raise ValueError(f"an image is uint8 or float, got {data.dtype}")
        if not np.isfinite(data).all():
            raise ValueError(
                "image holds non-finite values: a NaN or inf here renders as a black pixel and "
                "reads as a reconstruction failure. The renderer (#98) owns mapping an "
                "untrained model's predictions onto a finite colour range"
            )
        if data.size and (data.min() < 0.0 or data.max() > 1.0):
            raise ValueError(
                f"a float image is already normalized to [0, 1], got "
                f"[{data.min():g}, {data.max():g}] — the metres-to-colour mapping is the "
                "renderer's (#98), and doing it twice is how a grid ends up rescaled per epoch"
            )


#: What a caller may put in a log payload. Numbers and strings are the ordinary case; the two
#: dataclasses above are what make this seam serve #98 and #99 rather than scalars alone.
MetricValue = float | int | bool | str | Histogram | Image | Sequence[Image]


# --- The interface ------------------------------------------------------------


class RunLogger(ABC):
    """The seam. Three methods, and the step axis is the epoch (see the module docstring)."""

    def __init__(self) -> None:
        self._last_step: int | None = None
        self.closed = False

    def _check_step(self, step: int) -> None:
        """Refuse a step that goes backwards, which W&B would drop without saying so."""
        if self._last_step is not None and step < self._last_step:
            raise ValueError(
                f"step went backwards: {step} after {self._last_step}. The sink's step axis is "
                "the epoch, shared by every caller (#97's scalars, #98's grids, #99's probe), "
                "and W&B drops an out-of-order point silently"
            )
        self._last_step = step

    @abstractmethod
    def log(self, values: Mapping[str, MetricValue], *, step: int) -> None:
        """Record ``values`` at epoch ``step``. Several callers may log at one step."""

    @abstractmethod
    def summary(self, values: Mapping[str, MetricValue]) -> None:
        """The run's last word rather than a trend — a best loss, a final alarm count."""

    @abstractmethod
    def close(self) -> None:
        """Finish the run. Idempotent."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class MemoryLogger(RunLogger):
    """Everything logged, kept in this process. The test double, and ``disabled`` mode.

    Not a null logger, deliberately: a run with the sink switched off still wants its numbers
    readable — by a test asserting what an epoch emitted, by #81's smoke run on a laptop — and
    a logger that drops them makes "did the harness log anything?" unanswerable without W&B.
    """

    def __init__(self) -> None:
        super().__init__()
        #: ``(step, values)`` in call order, one entry per :meth:`log` call rather than per
        #: step, so two callers at one epoch stay distinguishable.
        self.history: list[tuple[int, dict[str, MetricValue]]] = []
        self.summaries: dict[str, MetricValue] = {}

    def log(self, values: Mapping[str, MetricValue], *, step: int) -> None:
        self._check_step(step)
        self.history.append((step, dict(values)))

    def at(self, step: int) -> dict[str, MetricValue]:
        """Everything logged at one epoch, merged across callers."""
        merged: dict[str, MetricValue] = {}
        for logged_step, values in self.history:
            if logged_step == step:
                merged.update(values)
        return merged

    def trend(self, key: str) -> list[tuple[int, MetricValue]]:
        """``(step, value)`` for one key, in order — what a trend assertion reads."""
        return [(s, v[key]) for s, v in self.history if key in v]

    def summary(self, values: Mapping[str, MetricValue]) -> None:
        self.summaries.update(values)

    def close(self) -> None:
        self.closed = True


# --- §15.7's tags and the run record ------------------------------------------


def run_tags(config: RunConfig, fingerprint: DataFingerprint) -> list[str]:
    """Plan §15.7's four tags: git SHA, config hash, selection-list md5, split md5.

    Prefixed, because a W&B tag list is a flat set of strings and four bare hex blobs are
    unreadable in the UI and unfilterable by kind. Derived from the config and the fingerprint
    rather than typed at a call site: these are the same four values the checkpoint payload
    carries (:func:`pretrain.header.run_record`), and a run whose tags and whose recorded config
    disagreed would be worse than one with no tags.

    **An uncommitted split tags as ``split:absent``** rather than being left out. The two are
    indistinguishable in the UI once the run exists, and the absent case is the dangerous one —
    it means the manifest column was not filled, which is what plan §15.2 gates R1 on.
    """
    return [
        f"sha:{fingerprint.git_sha}",
        f"config:{config.config_hash}",
        f"selection:{fingerprint.selection_list_md5}",
        f"split:{fingerprint.split_md5 or 'absent'}",
    ]


# --- W&B ----------------------------------------------------------------------


class WandbLogger(RunLogger):
    """A W&B run, online or offline, behind :class:`RunLogger`.

    Built by :func:`open_run_logger`. The wandb import is inside :meth:`open` so that importing
    this module — which #94's checkpoints and #93's loop both do transitively — never depends on
    the package being installed; only opening an actual W&B run does.
    """

    def __init__(self, run: Any) -> None:
        super().__init__()
        self.run = run

    @classmethod
    def open(
        cls, config: RunConfig, fingerprint: DataFingerprint, *, mode: str | None = None
    ) -> WandbLogger:
        """`wandb.init` with §15.7's tags and the full run record attached.

        ``dir`` is created first: wandb refuses a directory that does not exist, and the
        default is the run's own ``monitoring.wandb_dir``, which on a rented box points at the
        persistent volume rather than the container's ephemeral cwd.
        """
        import wandb

        m = config.monitoring
        directory = None
        if m.wandb_dir is not None:
            directory = Path(m.wandb_dir)
            directory.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            project=m.wandb_project,
            entity=m.wandb_entity,
            dir=None if directory is None else str(directory),
            mode=mode or m.wandb_mode,
            name=config.run_id,
            tags=run_tags(config, fingerprint),
            # #89's provenance block verbatim — the same dict a checkpoint carries. The tags
            # are for filtering; this is for reading, and it is what §11's "the box holds
            # nothing unique" actually means: a reader with the run page and neither the code
            # nor the corpus can still say which recipe produced a released encoder.
            config=run_record(config, fingerprint),
            reinit=True,
        )
        return cls(run)

    @property
    def run_dir(self) -> Path:
        """The run's own directory — ``…/offline-run-<stamp>-<id>`` in offline mode.

        `wandb`'s ``run.dir`` points at the ``files/`` subdirectory; the transaction log that
        makes an offline run syncable sits one level up, which is what a caller checking that
        the run landed on disk actually wants.
        """
        return Path(self.run.dir).parent

    @staticmethod
    def _translate(value: MetricValue) -> Any:
        """One payload into what wandb wants — the only place that mapping is written."""
        import wandb

        if isinstance(value, Histogram):
            return wandb.Histogram(
                np_histogram=(list(value.counts), list(value.edges))
            )
        if isinstance(value, Image):
            return wandb.Image(value.data, caption=value.caption)
        if isinstance(value, (list, tuple)):
            return [WandbLogger._translate(v) for v in value]
        return value

    def log(self, values: Mapping[str, MetricValue], *, step: int) -> None:
        self._check_step(step)
        self.run.log({k: self._translate(v) for k, v in values.items()}, step=step)

    def summary(self, values: Mapping[str, MetricValue]) -> None:
        for key, value in values.items():
            self.run.summary[key] = self._translate(value)

    def close(self) -> None:
        if not self.closed:
            self.run.finish()
            self.closed = True


def open_run_logger(
    config: RunConfig, fingerprint: DataFingerprint | None = None
) -> RunLogger:
    """The run's sink, off its config: a W&B run, or an in-process record at ``disabled``.

    The fingerprint is collected from the corpus when not supplied, so a caller that already
    has one (every entrypoint prints the header from it) does not read the manifest twice.
    """
    if config.monitoring.wandb_mode == "disabled":
        return MemoryLogger()
    return WandbLogger.open(config, fingerprint or fingerprint_for(config))
