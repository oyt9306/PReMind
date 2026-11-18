"""CPU-only: split_overview.{pdf,png} for the PReMind-Bench Benchmark section (D+london split + provisional b-lite).

Hazard types: by default a fixed per-image map (HAZARD_BY_IMAGE) read from each acute image's verdict text
(evaluator.acute_visual_hazard.image_verdict.what); --keyword uses the resplit_v2_scene keyword heuristic instead.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
FC = HERE.parents[1]
RS = FC / "resplit_v2_scene"
sys.path.insert(0, str(RS))
sys.path.insert(0, str(FC))
from common import hazard_types, identity_kinds, is_acute, load_rows, targets  # noqa: E402

DL = RS / "out_more" / "D_plus_london"
PREVIEW = RS / "preview_images"
BLITE_NEW = {"911bdd4b44e9": "aug-person-002", "355ca68b8429": "aug-person-003",
             "bcd113879364": "aug-person-003", "2a1d58e36dbc": "aug-person-032"}
TRAIN_SCENES = ("berlin", "bonsai", "bicycle", "kitchen", "alameda", "garden")
TEST_SCENES = ("room", "counter", "london", "nyc")
# one thumbnail per scene; all targets are objects/pets (no person), non-acute; checked on a contact sheet
THUMBS = {"berlin": "890d8a2b9506", "bonsai": "a6859b6a0bfa", "bicycle": "b6a3b96d729c", "kitchen": "94e75d654e78",
          "alameda": "f1a09515f291", "garden": "19aaf7af133c", "room": "1cee831a7c08", "nyc": "c3246cb64bc6",
          "counter": "12fa8c4e0d58", "london": "fc4c0889397c"}
# acute image (sha256[:6]) -> hazard types, from the image verdict text (quoted in NOTES.md)
HAZARD_BY_IMAGE = {
    # train
    "319ae1": ["person_down"], "561f0a": ["person_down"], "7132f0": ["person_down"], "e444c3": ["person_down"],
    "f727ba": ["person_down"], "463cd0": ["fall"], "4bebde": ["fall"], "5ef2ca": ["fall"], "cdcab9": ["fall"],
    "3c9a7c": ["fire_heat"], "91e336": ["fire_heat"], "0e3011": ["fire_heat"], "d4b205": ["fire_heat"],
    "407078": ["electrical"], "fccc8b": ["chemical"],
    # dev
    "1edd4d": ["electrical"], "786b2d": ["fall"], "541323": ["person_down"],
    # test
    "021773": ["person_down"], "53fe21": ["person_down"], "59e19d": ["person_down"], "97b405": ["person_down"],
    "a28ba7": ["person_down"], "b8d7d1": ["person_down"], "bf47eb": ["person_down"], "c11e45": ["person_down"],
    "c7d280": ["person_down"], "f4a716": ["person_down"], "4ebe07": ["fire_heat", "person_down"], "e5d9cc": ["fire_heat"],
    "843612": ["fire_heat"], "b55ad6": ["fire_heat"], "6c5abd": ["electrical"], "b25cc1": ["electrical"],
    "fe3302": ["electrical"], "355ca6": ["chemical"], "92381f": ["chemical"], "2a1d58": ["machine"],
    "911bdd": ["machine"], "bcd113": ["machine"], "828496": ["fall", "sharp"],
}
TYPE_ORDER = ["person_down", "fire_heat", "electrical", "chemical", "fall", "machine", "sharp"]
TYPE_LABEL = {"person_down": "Person down / injured", "fire_heat": "Fire, smoke, heat", "electrical": "Electrical",
              "chemical": "Chemical / fumes", "fall": "Fall", "machine": "Powered machine", "sharp": "Sharp object"}
KW_ORDER = ["collapse", "fire_smoke", "electrical", "chemical_fumes", "fall", "water", "machine", "sharp", "other"]
C = {"train": "#4C72B0", "dev": "#8FAADC", "test": "#DD8452"}


def load_split():
    recs = {r.uid: r for r in load_rows()}
    split, scene = {}, {}
    for s in ("train", "dev", "test"):
        for x in json.load(open(DL / f"{s}_ids.json")):
            split[x["row_id"]], scene[x["row_id"]] = s, x["scene"]
    dropped = json.load(open(DL / "dropped_ids.json"))
    blite = {}
    for im in json.load(open(RS / "blite" / "BLITE_MANIFEST.json"))["images"]:
        for row in im["rows"]:
            u = row["row_id"]
            split[u], scene[u] = "test", dropped[u]["scene"]
            blite[u] = (im["release_target_id"], BLITE_NEW[im["sha12"]])
    return recs, split, scene, blite


def htypes(r, keyword: bool) -> list[str]:
    if not is_acute(r):
        return []
    if keyword:
        return hazard_types(r)
    return HAZARD_BY_IMAGE[r.ev["source_image_sha256"][:6]]


def compute(keyword: bool) -> dict:
    recs, split, scene, blite = load_split()
    kinds = identity_kinds(list(recs.values()))
    tgt = {}
    for u in split:
        t = targets(recs[u])
        if u in blite:
            t = [blite[u][1] if x == blite[u][0] else x for x in t]
        tgt[u] = t
    out = {"hazard_mode": "keyword" if keyword else "image_map", "splits": {}}
    for s in ("train", "dev", "test"):
        ids = [u for u in split if split[u] == s]
        bl = [u for u in ids if u in blite]
        out["splits"][s] = {
            "rows": len(ids), "rows_provisional_blite": len(bl),
            "scenes": dict(Counter(scene[u] for u in ids)),
            "scenes_provisional_blite": dict(Counter(scene[u] for u in bl)),
            "unsafe_rows": sum(recs[u].main["risk"] == "unsafe" for u in ids),
            "acute_rows": sum(is_acute(recs[u]) for u in ids),
            "gold_action": dict(Counter(recs[u].main["gold_action"] for u in ids)),
            "hazard_rows": dict(Counter(h for u in ids for h in htypes(recs[u], keyword))),
            "hazard_rows_provisional_blite": dict(Counter(h for u in bl for h in htypes(recs[u], keyword))),
            "hazard_images": dict(Counter(h for sha in {recs[u].ev["source_image_sha256"] for u in ids
                                                        if is_acute(recs[u])}
                                          for h in htypes(next(recs[v] for v in ids if
                                                               recs[v].ev["source_image_sha256"] == sha), keyword))),
        }
    tr = [u for u in split if split[u] == "train"]
    te = [u for u in split if split[u] == "test"]
    tr_ids = {t for u in tr for t in tgt[u]}
    tr_h = {h for u in tr for h in htypes(recs[u], keyword)}
    un = {"scene": {"rows": len(te), "unseen_rows": len(te)}}
    for k in ("person", "pet", "object"):
        rk = [u for u in te if any(kinds.get(t) == k for t in tgt[u])]
        idk = {t for u in rk for t in tgt[u] if kinds.get(t) == k}
        un[k] = {"rows": len(rk), "unseen_rows": sum(any(kinds.get(t) == k and t not in tr_ids for t in tgt[u])
                                                     for u in rk),
                 "ids": len(idk), "unseen_ids": len(idk - tr_ids)}
    ac = [u for u in te if is_acute(recs[u])]
    te_h = {h for u in ac for h in htypes(recs[u], keyword)}
    un["hazard"] = {"rows": len(ac), "unseen_rows": sum(any(h not in tr_h for h in htypes(recs[u], keyword))
                                                       for u in ac),
                    "types": len(te_h), "unseen_types": sorted(te_h - tr_h)}
    out["unseen_vs_train"] = un
    dv_h = set(out["splits"]["dev"]["hazard_rows"])
    out["hazard_overlap"] = {"train_and_test": sorted(tr_h & te_h), "test_only": sorted(te_h - tr_h - dv_h),
                             "test_and_dev_not_train": sorted((te_h & dv_h) - tr_h),
                             "train_only": sorted(tr_h - te_h)}
    return out


def thumb(sha: str):
    f = glob.glob(str(PREVIEW / "*" / "*" / f"{sha}*.png"))[0]
    im = Image.open(f).convert("RGB")
    im.thumbnail((480, 320))
    return im


def draw(st: dict, path_stem: Path, keyword: bool) -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "STIXGeneral"],
                         "mathtext.fontset": "stix", "font.size": 7, "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5, "pdf.fonttype": 42})
    W, H = 7.0, 2.3
    fig = plt.figure(figsize=(W, H))
    tr, dv, te = (st["splits"][s] for s in ("train", "dev", "test"))
    # (a) scenes: 3x2 train/dev thumbnails, gap, 2x2 test thumbnails (all same size)
    tw, gx, gap, x0 = 0.0755, 0.006, 0.022, 0.006
    th = tw * W / H * 2 / 3
    ys = (0.47, 0.115)

    def put(sc, x, y, col, label):
        ax = fig.add_axes([x, y, tw, th]); ax.imshow(thumb(THUMBS[sc]), aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(1.2)
        ax.set_title(label, fontsize=6.3, pad=1.5)

    for k, sc in enumerate(TRAIN_SCENES):
        put(sc, x0 + (k % 3) * (tw + gx), ys[k // 3], C["train"],
            f"{sc} {tr['scenes'].get(sc, 0)}/{dv['scenes'].get(sc, 0)}")
    xt = x0 + 3 * (tw + gx) + gap
    for k, sc in enumerate(TEST_SCENES):
        prov = "$^\\dagger$" if te["scenes_provisional_blite"].get(sc) else ""
        put(sc, xt + (k % 2) * (tw + gx), ys[k // 2], C["test"], f"{sc} {te['scenes'].get(sc, 0)}{prov}")
    fig.text(0.004, 0.985, "(a) Capture scenes", fontsize=7.5, weight="bold", va="top")
    fig.text(x0 + 1.5 * tw + gx, 0.83, f"train/dev scenes ({tr['rows']}/{dv['rows']} rows)", fontsize=6.6,
             ha="center", color=C["train"], va="top", weight="bold")
    fig.text(xt + tw + gx / 2, 0.83, f"test scenes, unseen ({te['rows']}$^\\dagger$)", fontsize=6.6,
             ha="center", color=C["test"], va="top", weight="bold")
    # (b) hazard types
    axb = fig.add_axes([0.545, 0.15, 0.17, 0.70])
    order = KW_ORDER if keyword else TYPE_ORDER
    order = [t for t in order if any(st["splits"][s]["hazard_rows"].get(t) for s in ("train", "dev", "test"))]
    h = 0.27
    tr_h = set(tr["hazard_rows"])
    for k, s in enumerate(("train", "dev", "test")):
        vals = [st["splits"][s]["hazard_rows"].get(t, 0) for t in order]
        yy = [i + (k - 1) * h for i in range(len(order))]
        axb.barh(yy, vals, height=h, color=C[s], label=s, edgecolor="none")
        if s == "test":
            pv = [st["splits"][s]["hazard_rows_provisional_blite"].get(t, 0) for t in order]
            axb.barh(yy, pv, left=[v - p for v, p in zip(vals, pv)], height=h, color="none", edgecolor="white",
                     hatch="//////", linewidth=0)
        for i, (y, v) in enumerate(zip(yy, vals)):
            if v:
                unseen = s == "test" and order[i] not in tr_h
                axb.text(v + 0.8, y, f"{v}" + ("  unseen in training" if unseen else ""), va="center",
                         fontsize=5.6, color=C["test"] if unseen else "black", style="italic" if unseen else "normal")
    axb.set_yticks(range(len(order))); axb.set_yticklabels([TYPE_LABEL.get(t, t) for t in order], fontsize=6.3)
    axb.invert_yaxis()
    for i, t in enumerate(order):
        if t not in tr_h:
            axb.get_yticklabels()[i].set_color(C["test"])
    xmax = max(max(st["splits"][s]["hazard_rows"].values()) for s in ("train", "test"))
    axb.set_xlim(0, xmax * 1.12)
    axb.set_xlabel("acute-hazard rows (multi-label)", fontsize=6.5, labelpad=1)
    axb.tick_params(axis="x", labelsize=6, length=2, pad=1); axb.tick_params(axis="y", length=0, pad=2)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    axb.legend(fontsize=6, frameon=False, loc="center right", handlelength=1.0, borderaxespad=0.1)
    fig.text(0.452, 0.985, "(b) Acute hazard types per split", fontsize=7.5, weight="bold", va="top")
    # (c) unseen axes on test
    axc = fig.add_axes([0.835, 0.15, 0.095, 0.70])
    u = st["unseen_vs_train"]
    rows = [("Scene", u["scene"]["unseen_rows"], u["scene"]["rows"], "all rows"),
            ("Target person", u["person"]["unseen_rows"], u["person"]["rows"],
             f"{u['person']['unseen_ids']}/{u['person']['ids']} ids"),
            ("Object identity", u["object"]["unseen_rows"], u["object"]["rows"],
             f"{u['object']['unseen_ids']}/{u['object']['ids']} ids"),
            ("Pet identity", u["pet"]["unseen_rows"], u["pet"]["rows"], f"{u['pet']['unseen_ids']}/{u['pet']['ids']} ids"),
            ("Hazard type", u["hazard"]["unseen_rows"], u["hazard"]["rows"], "acute rows")]
    for i, (nm, a, b, note) in enumerate(rows):
        axc.barh(i, 100, height=0.62, color="#E6E6E6", edgecolor="none")
        axc.barh(i, 100 * a / b, height=0.62, color=C["test"], edgecolor="none")
        axc.text(103, i - 0.14, f"{a}/{b}", va="center", fontsize=5.8, clip_on=False)
        axc.text(103, i + 0.24, note, va="center", fontsize=5.0, color="#555555", clip_on=False)
    axc.set_yticks(range(len(rows))); axc.set_yticklabels([r[0] for r in rows], fontsize=6.3)
    axc.invert_yaxis(); axc.set_xlim(0, 100); axc.set_xticks([0, 50, 100])
    axc.set_xlabel("% test rows unseen in train", fontsize=6.5, labelpad=1)
    axc.tick_params(axis="x", labelsize=6, length=2, pad=1); axc.tick_params(axis="y", length=0, pad=2)
    for sp in ("top", "right"):
        axc.spines[sp].set_visible(False)
    fig.text(0.752, 0.985, "(c) Unseen axes on test", fontsize=7.5, weight="bold", va="top")
    fig.text(0.004, 0.01, f"$^\\dagger$includes {te['rows_provisional_blite']} provisional test rows "
             "(hatched in b). Scene labels: rows per split.", fontsize=5.5, ha="left", va="bottom", color="#555555")
    fig.savefig(path_stem.with_suffix(".pdf"))
    fig.savefig(path_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyword", action="store_true", help="keyword hazard types (resplit_v2_scene/common.py)")
    a = ap.parse_args()
    st = compute(a.keyword)
    stem = HERE / ("split_overview_keyword_v1" if a.keyword else "split_overview_v1")
    json.dump(st, open(stem.with_name(stem.name + "_stats.json"), "w"), indent=1)
    draw(st, stem, a.keyword)
    print(json.dumps({k: v for k, v in st.items()}, indent=1)[:6000])


if __name__ == "__main__":
    main()
