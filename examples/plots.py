"""
Renders the figures and markdown tables behind report.md from sweep_results.json.

Figures go to plots/*.png, generated tables to plots/tables.md (pasted into
report.md so the reported numbers are never hand-transcribed).

Run examples/sweep.py first to produce sweep_results.json.

Chart conventions: two series only (ours vs faiss) in a validated
categorical pair, one measure per axis (never a second y-scale), solid hairline
grid, thin marks, legend plus selective endpoint labels rather than a number on
every point. Light-mode only on purpose -- these are embedded in a print PDF,
not a theme-switching web page. Every figure has a table twin in report.md.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "sweep_results.json"
PLOTS_DIR = ROOT / "plots"

# validated categorical slots 1-3 (blue / orange / aqua): worst all-pairs CVD
# dE 9.2, normal-vision dE 24.0. Aqua sits at 2.74:1 on the light surface, so it
# carries the relief rule -- it is always direct-labelled and every figure has a
# table twin in report.md.
C_OURS = "#2a78d6"
C_FAISS = "#eb6834"
C_FAISS_IP = "#1baf7a"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

LABEL = {
    "ours": "ours (numpy)",
    "faiss": "faiss",
    "faiss-ip": "faiss (inner-product ADC)",
}
COLOR = {"ours": C_OURS, "faiss": C_FAISS, "faiss-ip": C_FAISS_IP}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": MUTED,
            "ytick.labelcolor": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",  # never dashed -- dashing reads as "threshold"
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "figure.dpi": 200,
        }
    )


def grid(ax, axis="both") -> None:
    ax.grid(True, axis=axis, zorder=0)
    ax.set_axisbelow(True)


def log_ticks_at(ax, values) -> None:
    """Put ticks exactly on the swept values. A log axis otherwise labels decades
    or powers of two, which don't line up with values like 10/20/50/100/200/400."""
    ax.set_xticks(list(values))
    ax.set_xticklabels([str(v) for v in values])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def line(ax, xs, ys, impl, **kw):
    """Thin 2px line, >=8px markers with a surface ring so overlaps stay readable."""
    return ax.plot(
        xs, ys,
        color=COLOR[impl], label=LABEL[impl],
        linewidth=2, marker="o", markersize=6.5,
        markeredgecolor=SURFACE, markeredgewidth=1.4,
        zorder=3, **kw,
    )


def label_ends(ax, xs, ys, key, fmt="{key}={val}"):
    """Label just the two endpoints of one curve, pushed horizontally away from
    the line so they can't collide with the other series or clip on the axis.
    Only ever called for the `ours` series -- labelling both curves' endpoints
    overlaps, and faiss sweeps the identical parameter values anyway."""
    # push each label outward, away from the curve, into the axis margin --
    # a rising curve runs straight through an inward-placed label
    for i, (dx, ha) in ((0, (-8, "right")), (len(xs) - 1, (8, "left"))):
        ax.annotate(
            fmt.format(key=key, val=xs[i]),
            (ys[i][0], ys[i][1]),
            textcoords="offset points", xytext=(dx, -3),
            ha=ha, fontsize=7, color=MUTED,
        )


def sel(records, sweep=None, family=None, impl=None, **params):
    out = []
    for r in records:
        if family and r["family"] != family:
            continue
        if impl and r["impl"] != impl:
            continue
        if sweep and r["sweep"] != sweep:
            continue
        if any(r["params"].get(k) != v for k, v in params.items()):
            continue
        out.append(r)
    return out


def series(records, key, **kw):
    """Return (xs, ys_by_metric) for one impl's sweep, sorted by the swept param."""
    rows = sorted(sel(records, **kw), key=lambda r: r["params"][key])
    return (
        [r["params"][key] for r in rows],
        {m: [r[m] for r in rows] for m in ("recall", "search_ms", "build_s", "memory_mb")},
    )


# ---------------------------------------------------------------- figures
def fig_recall_vs_latency(records, meta):
    """The headline trade-off: what recall does each index buy per millisecond?"""
    panels = [
        ("ivf", "nprobe", "nprobe", "IVF — sweeping nprobe"),
        ("pq", "Ks", "Ks", "PQ — sweeping Ks"),
        ("hnsw", "ef_search", "ef_search", "HNSW — sweeping ef_search"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))

    for ax, (family, sweep, key, title) in zip(axes, panels):
        for impl in ("ours", "faiss"):
            xs, ys = series(records, key, sweep=sweep, family=family, impl=impl)
            line(ax, ys["search_ms"], ys["recall"], impl)
            if impl == "ours":
                label_ends(ax, xs, list(zip(ys["search_ms"], ys["recall"])), key)
        ax.set_xscale("log")
        ax.set_xlabel("search latency per query (ms, log)")
        ax.set_title(title)
        ax.set_ylim(0, 1.08)
        ax.margins(x=0.22)  # room for the endpoint labels instead of clipping them
        grid(ax)

    axes[0].set_ylabel(f"recall@{meta['k']}")
    axes[0].legend(loc="lower right")
    fig.suptitle(
        "Recall vs. search latency — up and to the left is better"
        "   (endpoints labelled on the ours curve; faiss sweeps the same values)",
        fontsize=10.5, fontweight="bold", color=INK, y=1.03,
    )
    fig.tight_layout()
    save(fig, "fig1_recall_vs_latency.png")


def fig_build_and_memory(records, meta):
    """Where the from-scratch implementation actually pays: build time and bytes."""
    d = meta["defaults"]
    picks = [
        ("Flat", dict(family="flat", sweep="default")),
        ("IVF", dict(family="ivf", sweep="nprobe", nprobe=d["ivf"]["nprobe"])),
        ("PQ", dict(family="pq", sweep="Ks", Ks=d["pq"]["Ks"], M=d["pq"]["M"])),
        ("HNSW", dict(family="hnsw", sweep="ef_search", ef_search=d["hnsw"]["ef_search"])),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    width = 0.36

    # Flat is excluded from the build panel: it has no training or graph-building
    # phase at all (add() just stores the array), so its "build time" is a memcpy
    # that reads as 0.00s and only adds a misleading bar on a log axis.
    for metric, ax, ylabel, logy, items in (
        ("build_s", ax1, "build time (s, log)", True, picks[1:]),
        ("memory_mb", ax2, "index memory (MB)", False, picks),
    ):
        xs = range(len(items))
        for j, impl in enumerate(("ours", "faiss")):
            vals = [sel(records, impl=impl, **q)[0][metric] for _, q in items]
            # 2px surface gap between adjacent bars, no borders around marks
            bars = ax.bar(
                [x + (j - 0.5) * (width + 0.02) for x in xs], vals, width,
                color=COLOR[impl], label=LABEL[impl], zorder=3,
                edgecolor=SURFACE, linewidth=1.2,
            )
            ax.bar_label(bars, fmt="%.3g", padding=2, fontsize=7, color=INK_2)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([n for n, _ in items])
        ax.set_ylabel(ylabel)
        ax.margins(y=0.18)
        grid(ax, axis="y")

    ax1.set_title("Build time at default hyperparameters\n(Flat omitted — it has no build step)")
    ax2.set_title("Memory footprint at default hyperparameters\n(linear scale: PQ really is that small)")
    ax1.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig2_build_and_memory.png")


def fig_pq(records, meta):
    """PQ is the weak one -- these two panels show why, and where its knee is."""
    n_train = meta["n_indexed"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))

    for impl in ("ours", "faiss", "faiss-ip"):
        xs, ys = series(records, "Ks", sweep="Ks", family="pq", impl=impl)
        line(ax1, xs, ys["recall"], impl)
        if impl == "faiss-ip":  # relief rule: aqua is always direct-labelled
            mid = len(xs) // 2
            ax1.annotate(
                "inner-product ADC",
                (xs[mid], ys["recall"][mid]), textcoords="offset points",
                xytext=(0, -17), ha="center", fontsize=7, color=INK_2,
            )
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Ks — centroids per subspace (log2)")
    ax1.set_ylabel(f"recall@{meta['k']}")
    ax1.set_title(f"PQ recall vs. codebook size (M={meta['defaults']['pq']['M']})")
    ax1.legend(loc="upper left")
    ax1.margins(y=0.16)  # headroom so the annotations below the curves stay visible
    grid(ax1)

    # faiss warns below ~39 training points per centroid; mark where this corpus
    # crosses it. Anchored to the axes bottom in axes-fraction coords so the label
    # can never fall outside the data range and vanish.
    ks_at_39 = n_train / 39
    ax1.axvline(ks_at_39, color=MUTED, linewidth=0.9, zorder=1)
    ax1.annotate(
        f"Ks > {ks_at_39:.0f}: under 39 training\npoints per centroid",
        xy=(ks_at_39, 0), xycoords=("data", "axes fraction"),
        textcoords="offset points", xytext=(-7, 9),
        ha="right", fontsize=7, color=MUTED,
    )

    for impl in ("ours", "faiss"):
        xs, ys = series(records, "M", sweep="M", family="pq", impl=impl)
        line(ax2, xs, ys["recall"], impl)
        if impl == "ours":  # ours-only labels: both curves' labels collide
            for i, (dx, ha) in ((0, (-8, "right")), (len(xs) - 1, (8, "left"))):
                ax2.annotate(
                    f"{ys['memory_mb'][i]:.2f} MB",
                    (xs[i], ys["recall"][i]), textcoords="offset points",
                    xytext=(dx, -11), ha=ha, fontsize=7, color=MUTED,
                )
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("M — subspaces (log2)")
    ax2.set_ylabel(f"recall@{meta['k']}")
    ax2.set_title(f"PQ recall vs. code length (Ks={meta['defaults']['pq']['Ks']})")
    ax2.margins(x=0.2)
    grid(ax2)

    fig.tight_layout()
    save(fig, "fig3_pq.png")


def fig_ivf(records, meta):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    nlist = meta["defaults"]["ivf"]["nlist"]

    for impl in ("ours", "faiss"):
        xs, ys = series(records, "nprobe", sweep="nprobe", family="ivf", impl=impl)
        line(ax1, [x / nlist * 100 for x in xs], ys["recall"], impl)
    ax1.set_xscale("log")
    ax1.set_xlabel(f"% of clusters probed (nlist={nlist}, log)")
    ax1.set_ylabel(f"recall@{meta['k']}")
    ax1.set_title("IVF recall vs. fraction of index scanned")
    ax1.legend(loc="lower right")
    grid(ax1)

    for impl in ("ours", "faiss"):
        xs, ys = series(records, "nlist", sweep="nlist", family="ivf", impl=impl)
        line(ax2, xs, ys["recall"], impl)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("nlist — number of clusters (log2)")
    ax2.set_ylabel(f"recall@{meta['k']}")
    ax2.set_title(
        f"IVF recall vs. cluster count (nprobe={meta['defaults']['ivf']['nprobe']} fixed)"
    )
    grid(ax2)

    fig.tight_layout()
    save(fig, "fig4_ivf.png")


def fig_hnsw(records, meta):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))

    for impl in ("ours", "faiss"):
        xs, ys = series(records, "ef_search", sweep="ef_search", family="hnsw", impl=impl)
        line(ax1, xs, ys["recall"], impl)
    ax1.set_xscale("log", base=2)
    log_ticks_at(ax1, xs)
    ax1.set_xlabel("ef_search — beam width at query time (log)")
    ax1.set_ylabel(f"recall@{meta['k']}")
    ax1.set_title("HNSW recall vs. search beam width")
    ax1.legend(loc="lower right")
    grid(ax1)

    for impl in ("ours", "faiss"):
        xs, ys = series(records, "ef_construction", sweep="ef_construction",
                        family="hnsw", impl=impl)
        line(ax2, xs, ys["build_s"], impl)
        # recall barely moves across this sweep -- label both ends of each curve
        # to make that the visible point, offset horizontally to avoid collisions
        for i, (dx, ha) in ((0, (-8, "right")), (len(xs) - 1, (8, "left"))):
            ax2.annotate(
                f"recall {ys['recall'][i]:.3f}",
                (xs[i], ys["build_s"][i]), textcoords="offset points",
                xytext=(dx, -11), ha=ha, fontsize=7, color=MUTED,
            )
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    log_ticks_at(ax2, xs)
    ax2.set_xlabel("ef_construction — beam width at build time (log)")
    ax2.set_ylabel("build time (s, log)")
    ax2.set_title("HNSW build cost vs. ef_construction")
    ax2.margins(x=0.28, y=0.3)
    grid(ax2)

    fig.tight_layout()
    save(fig, "fig5_hnsw.png")


def fig_pareto(records, meta):
    """The whole design space on one plot: memory vs recall, all four families."""
    d = meta["defaults"]
    # Flat and HNSW land at almost the same point, so their labels need opposite
    # vertical offsets to stay legible
    picks = [
        ("Flat", "o", (9, 6), dict(family="flat", sweep="default")),
        ("IVF", "s", (9, -3), dict(family="ivf", sweep="nprobe", nprobe=d["ivf"]["nprobe"])),
        ("PQ", "^", (9, -3), dict(family="pq", sweep="Ks", Ks=d["pq"]["Ks"], M=d["pq"]["M"])),
        ("HNSW", "D", (9, -11),
         dict(family="hnsw", sweep="ef_search", ef_search=d["hnsw"]["ef_search"])),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    # composite encoding: hue = implementation, shape = index family
    for impl in ("ours", "faiss"):
        for name, marker, offset, q in picks:
            r = sel(records, impl=impl, **q)[0]
            ax.scatter(
                r["memory_mb"], r["recall"],
                s=95, marker=marker, color=COLOR[impl],
                edgecolor=SURFACE, linewidth=1.4, zorder=3,
                label=LABEL[impl] if name == "Flat" else None,
            )
            ax.annotate(
                name, (r["memory_mb"], r["recall"]),
                textcoords="offset points", xytext=offset,
                fontsize=8, color=INK_2,
            )

    ax.set_xscale("log")
    ax.set_xlabel("index memory (MB, log)")
    ax.set_ylabel(f"recall@{meta['k']}")
    ax.set_title("Recall vs. memory at default hyperparameters\n(shape = index, colour = implementation)")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower right")
    grid(ax)
    fig.tight_layout()
    save(fig, "fig6_recall_vs_memory.png")


def save(fig, name) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    path = PLOTS_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


# ---------------------------------------------------------------- tables
def md_tables(records, meta) -> str:
    """Emit every sweep as a markdown table so report.md never hand-copies a number."""
    out: list[str] = []
    d = meta["defaults"]

    def table(title, note, rows, param_cols):
        out.append(f"### {title}\n")
        if note:
            out.append(f"{note}\n")
        head = param_cols + [
            "recall@10 (ours)", "recall@10 (faiss)",
            "build s (ours)", "build s (faiss)",
            "search ms (ours)", "search ms (faiss)",
            "memory MB (ours)", "memory MB (faiss)",
        ]
        out.append("| " + " | ".join(head) + " |")
        out.append("|" + "---|" * len(head))
        for r in rows:
            out.append("| " + " | ".join(r) + " |")
        out.append("")

    def pair_rows(sweep, family, key, param_cols_fn):
        rows = []
        ours = sorted(sel(records, sweep=sweep, family=family, impl="ours"),
                      key=lambda r: r["params"][key])
        for o in ours:
            f = sel(records, sweep=sweep, family=family, impl="faiss", **o["params"])[0]
            rows.append(
                param_cols_fn(o["params"])
                + [f"{o['recall']:.3f}", f"{f['recall']:.3f}",
                   f"{o['build_s']:.2f}", f"{f['build_s']:.2f}",
                   f"{o['search_ms']:.3f}", f"{f['search_ms']:.3f}",
                   f"{o['memory_mb']:.2f}", f"{f['memory_mb']:.2f}"]
            )
        return rows

    # headline
    picks = [
        ("Flat (exact)", dict(family="flat", sweep="default")),
        (f"IVF (nlist={d['ivf']['nlist']}, nprobe={d['ivf']['nprobe']})",
         dict(family="ivf", sweep="nprobe", nprobe=d["ivf"]["nprobe"])),
        (f"PQ (M={d['pq']['M']}, Ks={d['pq']['Ks']})",
         dict(family="pq", sweep="Ks", Ks=d["pq"]["Ks"], M=d["pq"]["M"])),
        (f"HNSW (M={d['hnsw']['M']}, efC={d['hnsw']['ef_construction']}, "
         f"efS={d['hnsw']['ef_search']})",
         dict(family="hnsw", sweep="ef_search", ef_search=d["hnsw"]["ef_search"])),
    ]
    rows = []
    for name, q in picks:
        o = sel(records, impl="ours", **q)[0]
        f = sel(records, impl="faiss", **q)[0]
        rows.append(
            [name, f"{o['recall']:.3f}", f"{f['recall']:.3f}",
             f"{o['build_s']:.2f}", f"{f['build_s']:.2f}",
             f"{o['search_ms']:.3f}", f"{f['search_ms']:.3f}",
             f"{o['memory_mb']:.2f}", f"{f['memory_mb']:.2f}"]
        )
    table("Headline: all four indexes at default hyperparameters", "", rows, ["Index"])

    table(f"IVF — nprobe sweep (nlist={d['ivf']['nlist']})",
          "Only search changes, so build time is the one shared build.",
          pair_rows("nprobe", "ivf", "nprobe",
                    lambda p: [str(p["nprobe"]), f"{p['nprobe'] / p['nlist'] * 100:.1f}%"]),
          ["nprobe", "% scanned"])

    table("IVF — nlist sweep",
          f"nprobe fixed at {d['ivf']['nprobe']}; each row is a fresh k-means training run.",
          pair_rows("nlist", "ivf", "nlist", lambda p: [str(p["nlist"])]), ["nlist"])

    table(f"PQ — Ks sweep (M={d['pq']['M']})",
          f"{meta['n_indexed']} training vectors, so training points per centroid "
          f"= {meta['n_indexed']}/Ks.",
          pair_rows("Ks", "pq", "Ks",
                    lambda p: [str(p["Ks"]), f"{meta['n_indexed'] / p['Ks']:.0f}"]),
          ["Ks", "train pts / centroid"])

    table(f"PQ — M sweep (Ks={d['pq']['Ks']})",
          "Each vector is stored as M codes; more subspaces = finer approximation, more bytes.",
          pair_rows("M", "pq", "M",
                    lambda p: [str(p["M"]), str(meta["dim"] // p["M"])]),
          ["M", "dims / subspace"])

    # the ADC-metric finding: same codebooks, different distance table
    out.append("### PQ — L2 vs inner-product ADC table (faiss)\n")
    out.append(
        "Identical codebooks and identical normalized vectors; only the distance "
        "table used at search time differs. Our PQIndex normalizes and then uses "
        "L2 per subspace, so the L2 column is the apples-to-apples comparison.\n"
    )
    head = ["M", "Ks", "recall (ours, L2)", "recall (faiss, L2)",
            "recall (faiss, inner product)", "IP penalty"]
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))
    for sweep, key in (("Ks", "Ks"), ("M", "M")):
        for o in sorted(sel(records, sweep=sweep, family="pq", impl="ours"),
                        key=lambda r: r["params"][key]):
            f_l2 = sel(records, sweep=sweep, family="pq", impl="faiss", **o["params"])[0]
            f_ip = sel(records, sweep=sweep, family="pq", impl="faiss-ip", **o["params"])[0]
            out.append(
                f"| {o['params']['M']} | {o['params']['Ks']} | {o['recall']:.3f} | "
                f"{f_l2['recall']:.3f} | {f_ip['recall']:.3f} | "
                f"{f_l2['recall'] - f_ip['recall']:+.3f} |"
            )
    out.append("")

    table(f"HNSW — ef_search sweep (M={d['hnsw']['M']}, "
          f"ef_construction={d['hnsw']['ef_construction']})",
          "Search-time only, so build time is the one shared build.",
          pair_rows("ef_search", "hnsw", "ef_search", lambda p: [str(p["ef_search"])]),
          ["ef_search"])

    table("HNSW — M sweep", f"ef_construction={d['hnsw']['ef_construction']}, "
          f"ef_search={d['hnsw']['ef_search']}.",
          pair_rows("M", "hnsw", "M", lambda p: [str(p["M"])]), ["M"])

    # column header abbreviated to efC: the full name is wide enough to push this
    # 10-column table past the page margin in the PDF build
    table("HNSW — ef_construction sweep", f"M={d['hnsw']['M']}, "
          f"ef_search={d['hnsw']['ef_search']}. efC = ef_construction.",
          pair_rows("ef_construction", "hnsw", "ef_construction",
                   lambda p: [str(p["ef_construction"])]), ["efC"])

    return "\n".join(out)


def main() -> None:
    data = json.loads(RESULTS_PATH.read_text())
    records, meta = data["records"], data["meta"]
    style()

    fig_recall_vs_latency(records, meta)
    fig_build_and_memory(records, meta)
    fig_pq(records, meta)
    fig_ivf(records, meta)
    fig_hnsw(records, meta)
    fig_pareto(records, meta)

    PLOTS_DIR.mkdir(exist_ok=True)
    (PLOTS_DIR / "tables.md").write_text(md_tables(records, meta))
    print(f"wrote plots/tables.md")


if __name__ == "__main__":
    main()
