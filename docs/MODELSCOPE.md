# Downloading from ModelScope

> 中文版: [MODELSCOPE.zh-CN.md](MODELSCOPE.zh-CN.md)

setup.py gets everything it installs from Hugging Face: the model files (step 5) and the MTP draft layer (step 6, from
the original BF16 checkpoint).  From China that is often a few hundred KB/s, and the MTP tensors alone are ~5 GB.

Two tools read the same files from [ModelScope](https://modelscope.cn) (modelscope.cn) instead, which is usually
several MB/s there with no proxy at all:

| tool | what it gets | where from |
|---|---|---|
| `tools/download_model.py` | the model files of any family and size setup.py offers (`--family`, `--model`, `--vision`) | ModelScope (default), or Hugging Face |
| `tools/mtp_fetch_ms.py` | the 31 `mtp.*` tensors of the BF16 checkpoint (the draft layer) | ModelScope |

Both are drop-in replacements for the downloads setup.py makes: they write exactly what setup.py, `tools/mtp_pack.py`
and `tools/mtp_rt.py` already expect, and the `.done` marks they leave behind make a later setup run skip those
downloads instead of repeating them.

- The plain Hugging Face tools stay as they are: `tools/mtp_fetch.py` (range requests at the pinned commit) and
  setup.py's own `download()`.
- `docs/ORCA.md` documents the hand-built draft layer with the plain tool; `docs/UNSLOTH_Q4.md` mentions both.

## 0. What to install

- `tools/download_model.py` needs the **modelscope** package for `--source modelscope`:
  `pip install modelscope`.  Its default `--source auto` uses ModelScope when that package is importable and
  Hugging Face (setup.py's own downloader) when it is not, so the tool always works.
- `tools/mtp_fetch_ms.py` needs **nothing but Python**: it speaks ModelScope's HTTP API itself.

The interpreter matters: on the PC this was written on, modelscope is installed in the conda environment `strata`, not
in the `.venv` that `START-HERE.bat` builds, so there `--source auto` prints `source: Hugging Face`.  The command lines
below use `python` - run them with whichever interpreter has modelscope (or pass `--source modelscope` and read the
message when it is missing).

## 1. The model files: tools/download_model.py

The choices come from setup.py itself (`FAMILIES`, `MODELS`, `model_file`), so nothing is hard-coded: `--family qwen`
(the original), `swift`, `coder`, `unsloth`; `--model Q2_0`, `IQ2_XS`, `IQ3_XXS`, `IQ3_S`, `IQ1_M`, `UD-Q4_K_XL`,
`UD-IQ4_XS`.

```sh
python tools/download_model.py --list                    # every family and size, with setup.py's own notes
python tools/download_model.py                           # qwen IQ3_XXS (setup's default size)
python tools/download_model.py --model IQ2_XS
python tools/download_model.py --family coder --model IQ1_M
python tools/download_model.py --family unsloth --model UD-IQ4_XS --vision
python tools/download_model.py --model IQ3_S --check     # what is already there (no network)
python tools/download_model.py --model IQ3_S --dry-run   # the plan, nothing is downloaded
```

| option | what it does |
|---|---|
| `--family` | which model family (setup's `--family`); default `qwen` |
| `--model` | which size (setup's `--model`); default: the family's `IQ3_XXS`, or its only size (the Coder: `IQ1_M`) |
| `--vision` | also the image encoder (~1 GB), for setup's `--vision yes`; refused where setup never uses it (UD-Q4_K_XL) |
| `--models-dir` | where the files go; default: the data folder setup.py remembers, else `<this checkout>/../Strata-data`, both ending in `/models` |
| `--source` | `auto` (default) / `modelscope` / `huggingface` |
| `--repo`, `--revision`, `--endpoint` | the ModelScope repository id (default: the Hugging Face repository's own id), its revision (`master`), and another ModelScope endpoint/mirror |
| `--shard N` | only these shards, repeatable |
| `--list` | every family and size with setup.py's own notes, then exit |
| `--check` | what is already on disk, whole and marked; exit 0 when nothing is missing, 1 otherwise (no network) |
| `--dry-run` | the plan - the paths, the ModelScope call or the Hugging Face URLs - downloading nothing |

The original's `-00002-of-00002` is the same file for all its GSQ-RCO sizes and the Coder (setup.py hard-links it), so
a second size usually needs `--shard 1` only.

Where the files land, and what is checked:

```
<data>/models/<tag>/<file>          tag = setup.py's folder name: IQ3_XXS, coder-IQ1_M, swift-IQ2_XS, unsloth-UD-IQ4_XS
<data>/models/mmproj-*.gguf         the image encoder, beside the folders (setup.py looks for it there)
```

- ModelScope keeps the repository's own folder layout (`IQ3_XXS/...`, `IQ1_M/...`; Swift 1.5's files are at the
  repository root).  The tool takes each file from there and moves it into setup.py's `<tag>/` folder.
- Each file is checked the way setup.py checks a downloaded model file: **as long as its own GGUF tensor directory
  says** (`check_shards`), and the Unsloth sizes against their pinned size and SHA-256.  Only then does it get its
  `.done` mark - so a later `START-HERE.bat --setup` uses these files and downloads nothing.
- ModelScope serves its **own revision** of the files, not the commit setup.py pins.  Everything that can be checked
  locally is checked (GGUF headers, the Unsloth SHA-256s); for the other sizes the bytes are as ModelScope serves
  them, and the tool says so.

## 2. The MTP draft layer: tools/mtp_fetch_ms.py

The GGUF packs ship no MTP head; the BF16 checkpoint has it, in 31 tensors over 28 of its 131 shards.  The tool reads
the safetensors headers with HTTP range requests and downloads **only those tensors (~5 GB of a 360 GB checkpoint)**.

```sh
python tools/mtp_fetch_ms.py probe     --out <data>/mtp          # endpoint, revision, does it honour Range?
python tools/mtp_fetch_ms.py inventory --out <data>/mtp          # headers only, a few KB per shard
python tools/mtp_fetch_ms.py fetch     --out <data>/mtp --jobs 4 # the tensors; resumable
python tools/mtp_fetch_ms.py verify    --out <data>/mtp          # offline; exit 3: missing or corrupt tensors
```

| option | what it does |
|---|---|
| `--jobs N` | N chunks downloaded together (each holds up to 64 MiB in RAM).  `1` (the default) is the plain tool's single-stream behaviour; 4-8 saturates a fast link |
| `--only SUBSTR` | only the tensors whose name contains this.  **It rewrites `mtp-manifest.json` as that subset**: run `fetch` again without it before `mtp_pack.py` |
| `--fallback-full-shard` | if the endpoint ignores Range, download the whole shard (GBs) and slice the ranges out of it locally |
| `--keep-shards` | keep those shards instead of deleting them |
| `--no-sha256` | do not check the tensors against the pinned checkpoint's SHA-256 (another revision) |

| environment | default |
|---|---|
| `MODELSCOPE_MTP_REPO` | `Qwen/Qwen3.8-Flash-Next` |
| `MODELSCOPE_MTP_REVISION` | `master` (a branch, tag or commit of that repository) |
| `MODELSCOPE_ENDPOINT` | `https://modelscope.cn` |
| `MTP_MS_SCHEME` | `api` (ModelScope's file API; `resolve` uses the `/models/<id>/resolve/<rev>/<path>` form) |

`fetch` writes one raw file per tensor plus `mtp-inventory.json`, `mtp-inventory.md` and `mtp-manifest.json` - exactly
what `tools/mtp_fetch.py` writes, so the next two steps are the same either way:

```sh
python tools/mtp_pack.py --src <data>/mtp --experts q2_0 --out <data>/mtp/mtp-q2_0.gguf
python tools/mtp_rt.py   --gguf <data>/mtp/mtp-q2_0.gguf --out <data>/mtp/rt
```

`rt/experts.bin` (plus `dense.bin`, `dense.txt`) is what the engine's `--mtp` flag wants; setup.py looks for it at
`<data>/mtp/rt/experts.bin` and skips step 6 entirely when it is there.

Reruns are cheap: a tensor that is already there with the right SHA-256 is kept ("already fetched, kept") and nothing
is asked for again; an interrupted transfer continues where it stopped (one file with `--jobs 1`, numbered
`.partNNNN` pieces with `--jobs N`, joined once they are all there).

## 3. A whole install, both halves from ModelScope

```sh
# 1. the model files (setup.py's step 5)
python tools/download_model.py --model IQ3_XXS --vision

# 2. the MTP draft layer (setup.py's step 6): ~5 GB
python tools/mtp_fetch_ms.py probe  --out <data>/mtp
python tools/mtp_fetch_ms.py fetch  --out <data>/mtp --jobs 4
python tools/mtp_fetch_ms.py verify --out <data>/mtp
python tools/mtp_pack.py --src <data>/mtp --experts q2_0 --out <data>/mtp/mtp-q2_0.gguf
python tools/mtp_rt.py   --gguf <data>/mtp/mtp-q2_0.gguf --out <data>/mtp/rt

# 3. the engine, the packs and the start script: setup.py finds all of the above and downloads nothing
START-HERE.bat --setup --family qwen --model IQ3_XXS      # Linux: ./setup.sh ...
```

`<data>` is the data folder setup.py uses (`Strata-data` next to the checkout unless a `--data-dir`/`--models-dir` was
given); `tools/download_model.py --check` and `--dry-run` print the paths it would use.

## 4. What is checked, and why

`tools/mtp_fetch_ms.py` keeps the discipline of `tools/mtp_fetch.py` (#327): **a range read must answer 206 with the
`Content-Range` that was asked for**, and every tensor is checked against the pinned checkpoint's SHA-256.  A mirror or
proxy that ignores the `Range` header answers 200 with the whole shard, and a proxy may cut that to the requested
length - neither the size nor a hash of what arrived would catch that, and the drafter then accepts nothing, silently.
So a sloppy endpoint is refused rather than guessed at; `--fallback-full-shard` is the way through when the endpoint
cannot do ranges.

The SHA-256 table is the pinned checkpoint's **bytes**, which is why it applies here too: ModelScope has no equivalent
of the pinned Hugging Face commit, but the tensors either are the checkpoint's or are not.  `verify` is offline, and
both tools share `tensors/verified.json` for the tensors they agree on.

`tools/download_model.py` is stricter about nothing and looser about nothing either: it uses the same checks setup.py
applies to a hand-copied file, and it never marks a file `.done` before those checks pass.

## 5. When something goes wrong

| message | what it means |
|---|---|
| `range request not honoured: HTTP 200 instead of 206` | the mirror or proxy ignores `Range`.  Another `--endpoint`, or `--fallback-full-shard` (whole shards: GBs each) |
| `sha256 ... is not the pinned checkpoint's` | that ModelScope revision does not carry the checkpoint's bytes: another `--revision`, or `--no-sha256` if you mean it |
| `ModelScope could not serve <repo>` | the repository or revision is not mirrored there: `--repo <id>`, `--revision <branch-or-commit>`, or `--source huggingface` |
| `the modelscope package is not installed` | `pip install modelscope`, or `--source huggingface` |
| `Qwen... Coder has no IQ3_XXS` | that family does not have that size; the message lists what it has (`--list` for the details) |
| `has no image support in setup.py` | setup.py never uses the encoder with that size (e.g. UD-Q4_K_XL) |
| `no finish mark: <file>` (from `--check`) | the file is there but setup.py would download it again; run the tool without `--check` to finish it |
| `is not whole: it is not as long as its own tensor directory says` | a truncated file: delete it and run the tool again |
| `mtp_rt.py` stops with a `KeyError` (or an assertion on the tensors' shapes) | `fetch --only` left a subset in `mtp-manifest.json`, so `mtp_pack.py` wrote a GGUF with only those tensors: run `fetch` again without `--only` |

## 6. The tests, and what was measured here

Both tools have offline tests (a fake ModelScope endpoint behind a mocked `urlopen`, a fake `snapshot_download`):

```sh
python -m unittest tools.test_download_model tools.test_mtp_fetch_ms tools.test_mtp_fetch
```

Measured on the PC this was written on (Windows, the data folder `E:\Strata-data`):

- `mtp_fetch_ms.py probe` against `Qwen/Qwen3.8-Flash-Next@master`: **31 mtp tensors, 5.214 GB, in 28 shards**, range
  reads answered **206** - the same inventory `tools/mtp_fetch.py` reads from Hugging Face.
- `mtp_fetch_ms.py verify` on an existing `mtp/tensors`: exit 0 (all 31 tensors the checkpoint's).
- `download_model.py --check --model IQ3_XXS`: two shards present and whole, 47.04 GB + 28.80 GB (75.84 GB, setup.py's
  estimate is 75.8 GB).
- The model-file repositories themselves (`ISTA-DASLab/...`, `unsloth/...`) were **not** tried from ModelScope here;
  `--dry-run` shows the exact calls, and `--source huggingface` is the fallback when one is not mirrored.