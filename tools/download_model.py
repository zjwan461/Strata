"""tools/download_model.py - the model files of one setup choice, downloaded from the command line.

setup.py downloads the model it installs from Hugging Face at the revision pinned in setup.py.  This tool downloads the
same files for the family and size you name, from ModelScope by default (usually the faster host here, no proxy needed),
and puts them exactly where setup.py looks for them - <data>/models/<tag>/<file>, each with its .done mark - so a later
START-HERE.bat --setup finds them and does not download them again.

The choices and the file names come from setup.py itself (FAMILIES, MODELS, model_file), so this stays in step with
setup's options; nothing here is hard-coded to one model.

    python tools/download_model.py --list                      # the families and the sizes setup.py offers
    python tools/download_model.py                             # qwen IQ3_XXS (setup's default size)
    python tools/download_model.py --model IQ2_XS
    python tools/download_model.py --family coder --model IQ1_M
    python tools/download_model.py --family unsloth --model UD-IQ4_XS --vision
    python tools/download_model.py --model IQ3_S --models-dir E:\\Strata-data\\models
    python tools/download_model.py --model IQ3_S --check        # what is already there (offline)
    python tools/download_model.py --model IQ3_S --dry-run      # the plan, nothing is downloaded

--source (default auto):
    modelscope   modelscope.cn through the modelscope package - snapshot_download(repo, revision, local_dir).
                 --repo / --revision / --endpoint name another mirror, branch or commit; the default repository id is
                 the one the Hugging Face repository has (mirrors usually keep the same id)
    huggingface  setup.py's own downloader: resumable, HF_ENDPOINT honoured, the pinned revision, and the Unsloth
                 sizes checked against their pinned SHA-256
    auto         modelscope when the modelscope package is importable, else huggingface

--shard N (repeatable) fetches only some shards: the original's -00002-of-00002 is the same file for all its sizes and
the Coder, so a second size usually needs its first shard only.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup as S  # noqa: E402  the choices (FAMILIES, MODELS) and setup's own downloader


def families_of(model: str) -> tuple:
    """The families a size exists for, as setup.py reads it (gguf_choice)."""
    return tuple(S.MODELS[model].get("families", ("qwen", "swift")))


def default_model(family: str) -> str:
    """setup's default size when this family has it (IQ3_XXS), else the family's smallest choice."""
    allowed = sorted(m for m in S.MODELS if family in families_of(m))
    return "IQ3_XXS" if "IQ3_XXS" in allowed else allowed[0]


def default_models_dir() -> Path:
    """Where setup.py keeps the model files: the data folder it remembers, else Strata-data next to this checkout."""
    data = S.load_settings().get("data_dir")
    return (Path(data) if data else ROOT.parent / "Strata-data") / "models"


def hf_repo(url: str) -> str:
    """"https://huggingface.co/<repo>/resolve/<sha>/..." -> "<repo>"."""
    m = re.match(r"https?://[^/]+/(.+?)/resolve/", url)
    return m.group(1) if m else url


def repo_subdir(fam: dict, model: str) -> str:
    """The repository folder a family's shards live in: the size's name, or "" when they are at the repository's root
    (Swift 1.5's are; the GSQ-RCO and Unsloth repositories keep one folder per size)."""
    return model if "{q}" in fam["hf"] else ""


def ms_snapshot(repo: str, revision: str, endpoint: str | None, dest: Path, patterns: list, dry: bool = False) -> None:
    """ModelScope's snapshot_download into dest (the repository's own folder layout), or the call it would make."""
    if endpoint:
        os.environ["MODELSCOPE_ENDPOINT"] = endpoint       # modelscope reads it when it is imported, so set it first
    if dry:
        S.say(f"  would run: snapshot_download({repo!r}, revision={revision!r}, local_dir={str(dest)!r},"
              f" allow_file_pattern={patterns!r})")
        return
    try:
        from modelscope import snapshot_download
    except ImportError:
        S.fail("the modelscope package is not installed",
               "pip install modelscope, or run this tool with --source huggingface")
    S.say(f"  Downloading from ModelScope ({os.environ.get('MODELSCOPE_ENDPOINT', 'https://modelscope.cn')}), "
          f"{repo} @ {revision} ...")
    try:
        snapshot_download(repo, revision=revision, local_dir=str(dest), allow_file_pattern=patterns)
    except TypeError:                                      # an older modelscope named the pattern parameter differently
        snapshot_download(repo, revision=revision, local_dir=str(dest), allow_patterns=patterns)
    except Exception as e:
        S.fail(f"ModelScope could not serve {repo} ({e})",
               "that repository or revision may not be mirrored there: try --repo <id>, --revision <branch-or-commit>, "
               "or --source huggingface")


def place(root: Path, sub: str, dest: Path) -> None:
    """ModelScope writes a file under its repository path (<root>/<size>/<file>): move it to where setup.py expects it
    (<root>/<tag>/<file>); the same path (the original sizes) stays where it is."""
    if not sub:
        return
    src = root / sub / dest.name
    if src == dest or not src.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dest)
    try:
        (root / sub).rmdir()                               # the now-empty folder ModelScope made
    except OSError:
        pass


def finish(fam: dict, s: Path) -> None:
    """The .done mark setup.py looks for - written only after the file checked out: the Unsloth sizes against their
    pinned SHA-256 (setup's verify_sha256, which marks it too), everything else against its own GGUF tensor directory."""
    if s.name in fam.get("sha256", {}):
        S.verify_sha256(s, *fam["sha256"][s.name])
    elif S.whole_shard(s):
        S.mark(s, "downloaded with tools/download_model.py")
    else:
        S.fail(f"{s} is not whole: it is not as long as its own tensor directory says",
               "delete the file and run this tool again")


def downloads(fam: dict, model: str, root: Path, tag: str, targets: list, mmproj: bool, a) -> None:
    """Get `targets` (and the image encoder) from the chosen source and leave them where setup.py expects them.  With
    --dry-run it only says what it would fetch (from either source: nothing here opens a connection then)."""
    root.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        for s in targets:
            S.say(f"  would download: {s}")
        if mmproj:
            S.say(f"  would download: {root / fam['mmproj']} (vision encoder)")
    if a.source == "huggingface" or (a.source == "auto" and not _have_modelscope()):
        if a.dry_run:
            for s in targets:
                S.say(f"  from Hugging Face: {fam['hf'].format(q=model) + s.name}")
            if mmproj:
                S.say(f"  from Hugging Face: {fam['mmproj_hf'] + fam['mmproj']}")
            return
        for s in targets:
            S.download(fam["hf"].format(q=model) + s.name, s)
        if mmproj:
            S.download(fam["mmproj_hf"] + fam["mmproj"], root / fam["mmproj"], "vision encoder")
        return
    sub = repo_subdir(fam, model)
    repo = a.repo or hf_repo(fam["hf"])
    ms_snapshot(repo, a.revision, a.endpoint, root, [(f"{sub}/{s.name}" if sub else s.name) for s in targets], a.dry_run)
    for s in targets:
        place(root, sub, s)
        if not a.dry_run:
            finish(fam, s)
    if mmproj:
        mm = root / fam["mmproj"]
        here = fam["mmproj_hf"]
        ms_snapshot(a.repo or hf_repo(here), a.revision, a.endpoint, root, [fam["mmproj"]], a.dry_run)
        if not a.dry_run:
            finish(fam, mm)


def _have_modelscope() -> bool:
    try:
        import modelscope  # noqa: F401
        return True
    except ImportError:
        return False


def listing() -> None:
    S.say("Families (--family):")
    for f, d in sorted(S.FAMILIES.items()):
        S.say(f"  {f:<9} {d['title']} - {d['about']}")
    S.say("")
    S.say("Sizes (--model), as setup.py has them:")
    for m, d in S.MODELS.items():
        fams = ", ".join(families_of(m))
        n = d.get("shards", S.FAMILIES[families_of(m)[0]].get("shards", 2))
        S.say(f"  {m:<12} {d['download_gb']:5.1f} GB  {n} shard(s)  families: {fams}")
        S.say(f"               {d['about']}")
        if d.get("experimental"):
            S.say("               EXPERIMENTAL in setup.py (docs/UNSLOTH_Q4.md)")


def status(fam: dict, model: str, shards: list, models_dir: Path, mmproj: Path) -> bool:
    """What is already there: every requested shard whole and marked (offline).  True when nothing is missing."""
    S.say(f"  {fam['title']} {model} -> {models_dir}")
    if not shards:
        return False
    for s in shards:
        if not s.exists():
            S.warn(f"missing: {s.name}")
        elif not S.done(s):
            S.warn(f"no finish mark: {s.name} (setup downloads it again)")
        elif not S.whole_shard(s):
            S.warn(f"short: {s.name} (delete it and download it again)")
        else:
            S.ok(f"{s.name} ({s.stat().st_size / 1e9:.2f} GB, whole)")
    if mmproj is not None:
        S.ok(f"{mmproj.name} there") if mmproj.exists() else S.warn(f"missing: {mmproj.name} (only needed with images)")
    return all(s.exists() and S.done(s) and S.whole_shard(s) for s in shards)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=sorted(S.FAMILIES), default="qwen",
                    help="which model family (setup's --family); coder|unsloth hold one or two sizes")
    ap.add_argument("--model", choices=sorted(S.MODELS),
                    help="which size (setup's --model); the default is the family's IQ3_XXS, or its only size")
    ap.add_argument("--models-dir", dest="models_dir",
                    help="where the files go: the data folder setup.py remembers, else <this checkout>/../Strata-data, "
                         "both ending in /models (setup's --models-dir)")
    ap.add_argument("--source", choices=("auto", "modelscope", "huggingface"), default="auto",
                    help="where to download from (default auto: modelscope when it is installed, else huggingface)")
    ap.add_argument("--repo", help="the ModelScope repository id (default: the Hugging Face repository's own id)")
    ap.add_argument("--revision", default="master", help="the ModelScope revision: a branch, tag or commit")
    ap.add_argument("--endpoint", help="another ModelScope endpoint or mirror (MODELSCOPE_ENDPOINT)")
    ap.add_argument("--vision", action="store_true", help="also the image encoder (~1 GB), for setup's --vision yes")
    ap.add_argument("--shard", type=int, action="append", help="only these shards, 1..N (repeatable)")
    ap.add_argument("--list", action="store_true", help="the families and sizes setup.py offers, then exit")
    ap.add_argument("--check", action="store_true", help="only say what is already there (reads no network)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; nothing is downloaded")
    a = ap.parse_args()
    if a.list:
        listing()
        return 0

    family = a.family
    fam = S.FAMILIES[family]
    model = a.model or default_model(family)
    allowed = sorted(m for m in S.MODELS if family in families_of(m))
    if model not in allowed:
        S.fail(f"{fam['title']} has no {model}",
               f"{family} holds: {', '.join(allowed)} (run with --list for what each size is)")
    if a.vision and not S.MODELS[model].get("vision", fam.get("vision", True)):
        S.fail(f"{fam['title']} {model} has no image support in setup.py: the encoder is never used with it",
               "leave --vision out (the sizes that support images: " +
               ", ".join(m for m in S.MODELS if S.MODELS[m].get("vision", S.FAMILIES[families_of(m)[0]]
                                                               .get("vision", True))) + ")")

    tag = fam["tag"] + model                                  # setup.py's folder for this choice
    root = Path(a.models_dir).expanduser() if a.models_dir else default_models_dir()
    models_dir = root / tag
    total = S.model_shards(fam, model)
    picks = sorted(set(a.shard or range(1, total + 1)))
    if picks[0] < 1 or picks[-1] > total:
        S.fail(f"--shard takes 1..{total} for {model} ({total} shards)", "or leave --shard out for all of them")
    names = [S.model_file(fam, model, i) for i in picks]
    shards = [models_dir / n for n in names]
    mmproj = (root / fam["mmproj"]) if a.vision else None

    S.say()
    S.say(f"=== {fam['title']} {model} - {S.MODELS[model]['download_gb']:.1f} GB, {len(names)} of {total} shard(s) ===")
    S.say(f"  {S.MODELS[model]['about']}")
    S.say(f"  source: {'ModelScope' if a.source == 'modelscope' or (a.source == 'auto' and _have_modelscope())
                           else 'Hugging Face'} | files: {models_dir}")

    if a.check:
        return 0 if status(fam, model, shards, models_dir, mmproj) else 1

    todo = [s for s in shards if not (s.exists() and S.done(s))]
    want_mm = bool(mmproj) and not (mmproj.exists() and S.done(mmproj))
    for s in shards:
        if s not in todo:
            S.ok(f"{s.name} already downloaded")
    if a.dry_run:
        downloads(fam, model, root, tag, todo, want_mm, a)    # prints the plan; nothing is downloaded
        return 0
    if not todo and not want_mm:
        S.ok(f"nothing to do: {fam['title']} {model} is already in {models_dir}")
    else:
        downloads(fam, model, root, tag, todo, want_mm, a)
        S.check_shards(shards)                                # each file as long as its own tensor directory says
        for s in shards:
            S.ok(f"{s.name} ({s.stat().st_size / 1e9:.2f} GB)")
        if a.source != "huggingface":
            S.say("")
            S.say("  Note: ModelScope serves its own revision of these files (not the Hugging Face commit setup.py "
                  "pins). The GGUF headers are checked, and the Unsloth sizes against their pinned SHA-256; for the "
                  "other sizes the bytes are as ModelScope serves them - setup.py finds them by their .done marks and "
                  "prepares them without downloading them again.")
    S.say("")
    S.say(f"  next: {'START-HERE.bat' if S.WIN else './setup.sh'} --setup --family {family} --model {model}"
          + (" --vision yes" if a.vision else ""))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        S.say("\nstopped.")
        sys.exit(1)
