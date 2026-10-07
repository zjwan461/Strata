# 使用 ModelScope 下载

> English: [MODELSCOPE.md](MODELSCOPE.md)

setup.py 装东西时一律从 Hugging Face 取：模型文件（第 5 步）和 MTP 草稿层（第 6 步，来自原始 BF16 checkpoint）。在国内这常常只有几百 KB/s，而光是 MTP 那些张量就有约 5 GB。

下面两个工具改为从 [ModelScope](https://modelscope.cn)（modelscope.cn）读同样的文件，在国内通常能跑到几 MB/s 而且完全不需要代理：

| 工具 | 取什么 | 从哪取 |
|---|---|---|
| `tools/download_model.py` | setup.py 支持的任意 family / size 的模型文件（`--family`、`--model`、`--vision`） | ModelScope（默认），或 Hugging Face |
| `tools/mtp_fetch_ms.py` | BF16 checkpoint 的 31 个 `mtp.*` 张量（即草稿层） | ModelScope |

两者都是 setup.py 那两步下载的直接替代：写出的东西正好是 setup.py、`tools/mtp_pack.py`、`tools/mtp_rt.py` 期望的格式，并且会留下 `.done` 标记，所以之后再跑 setup 会直接跳过这些下载，而不是重下一遍。

- 原来的 Hugging Face 工具保持原样：`tools/mtp_fetch.py`（在钉住的 commit 上做 range 请求）和 setup.py 自带的 `download()`。
- `docs/ORCA.md` 记录了用原版工具手工构建草稿层的流程；`docs/UNSLOTH_Q4.md` 两个都提到了。

## 0. 需要装什么

- `tools/download_model.py` 的 `--source modelscope` 需要 **modelscope** 包：`pip install modelscope`。它的默认 `--source auto` 在这个包可导入时用 ModelScope，否则退回 Hugging Face（setup.py 自带的下载器），所以这个工具永远能跑。
- `tools/mtp_fetch_ms.py` 除 Python 外**什么都不需要**：它自己直接讲 ModelScope 的 HTTP API。

解释器很重要：在写这份文档的那台机器上，modelscope 装在 conda 环境 `strata` 里，而不在 `START-HERE.bat` 建的 `.venv` 里，所以用 `.venv` 跑时 `--source auto` 会打印 `source: Hugging Face`。下面的命令行都写作 `python` —— 用哪个解释器装了 modelscope 就用哪个（或者直接 `--source modelscope`，缺包时会给出提示）。

## 1. 模型文件：tools/download_model.py

可选项直接来自 setup.py 本身（`FAMILIES`、`MODELS`、`model_file`），没有任何写死的模型：`--family qwen`（原版）、`swift`、`coder`、`unsloth`；`--model Q2_0`、`IQ2_XS`、`IQ3_XXS`、`IQ3_S`、`IQ1_M`、`UD-Q4_K_XL`、`UD-IQ4_XS`。

```sh
python tools/download_model.py --list                    # 所有 family 和 size，附 setup.py 自己的说明
python tools/download_model.py                           # qwen IQ3_XXS（setup 的默认尺寸）
python tools/download_model.py --model IQ2_XS
python tools/download_model.py --family coder --model IQ1_M
python tools/download_model.py --family unsloth --model UD-IQ4_XS --vision
python tools/download_model.py --model IQ3_S --check     # 看看本地已经有什么（不联网）
python tools/download_model.py --model IQ3_S --dry-run   # 只打印计划，不下载
```

| 选项 | 作用 |
|---|---|
| `--family` | 选哪个模型家族（即 setup 的 `--family`）；默认 `qwen` |
| `--model` | 选哪个尺寸（即 setup 的 `--model`）；默认取该家族的 `IQ3_XXS`，或它唯一的尺寸（Coder 为 `IQ1_M`） |
| `--vision` | 同时下图像编码器（约 1 GB），对应 setup 的 `--vision yes`；setup 从不使用它的尺寸会被拒绝（如 UD-Q4_K_XL） |
| `--models-dir` | 文件放哪；默认是 setup.py 记住的数据目录，否则 `<本 checkout>/../Strata-data`，两者都再接 `/models` |
| `--source` | `auto`（默认）/ `modelscope` / `huggingface` |
| `--repo`、`--revision`、`--endpoint` | ModelScope 仓库 id（默认取该 Hugging Face 仓库的同名 id）、它的 revision（`master`）、以及另一个 ModelScope 端点或镜像 |
| `--shard N` | 只取这些分片，可重复 |
| `--list` | 列出所有 family 和 size 及 setup.py 的说明，然后退出 |
| `--check` | 只报告本地已有什么、是否完整、是否已打标记；齐了退出 0，否则 1（不联网） |
| `--dry-run` | 只打印计划（路径、ModelScope 调用或 Hugging Face URL），不下载任何东西 |

原版的 `-00002-of-00002` 对它的所有 GSQ-RCO 尺寸以及 Coder 都是同一个文件（setup.py 用硬链接复用），所以下第二个尺寸通常只需要 `--shard 1`。

文件落在哪、以及做了哪些校验：

```
<data>/models/<tag>/<file>          tag = setup.py 的目录名：IQ3_XXS、coder-IQ1_M、swift-IQ2_XS、unsloth-UD-IQ4_XS
<data>/models/mmproj-*.gguf         图像编码器，放在各目录旁边（setup.py 就在那里找它）
```

- ModelScope 保留仓库自己的目录结构（`IQ3_XXS/...`、`IQ1_M/...`；Swift 1.5 的文件在仓库根目录）。工具从那里取出文件，搬进 setup.py 的 `<tag>/` 目录。
- 每个文件都按 setup.py 校验下载下来的模型文件的方式检查：**长度要与它自己的 GGUF 张量目录一致**（`check_shards`），Unsloth 两个尺寸还要比对钉住的 size 和 SHA-256。只有检查通过才会写 `.done` 标记 —— 之后 `START-HERE.bat --setup` 就会直接用这些文件，不再下载。
- ModelScope 提供的是它**自己的 revision**，不是 setup.py 钉住的那个 commit。凡是在本地能校验的都校验了（GGUF 头、Unsloth 的 SHA-256）；其它尺寸的字节就是 ModelScope 给的字节，工具会明确说出来。

## 2. MTP 草稿层：tools/mtp_fetch_ms.py

GGUF 包里不含 MTP 头；BF16 checkpoint 里有，散落在 131 个分片中的 28 个上，共 31 个张量。工具用 HTTP range 请求读 safetensors 头部，然后**只下这些张量（约 5 GB，而 checkpoint 是 360 GB）**。

```sh
python tools/mtp_fetch_ms.py probe     --out <data>/mtp          # 端点、revision、是否支持 Range？
python tools/mtp_fetch_ms.py inventory --out <data>/mtp          # 只读头部，每分片几 KB
python tools/mtp_fetch_ms.py fetch     --out <data>/mtp --jobs 4 # 下张量；可断点续传
python tools/mtp_fetch_ms.py verify    --out <data>/mtp          # 离线；退出码 3：有缺失或损坏的张量
```

| 选项 | 作用 |
|---|---|
| `--jobs N` | 同时下载 N 个分块（每个最多占 64 MiB 内存）。`1`（默认）就是原版工具的单流行为；4-8 一般就能跑满快链路 |
| `--only SUBSTR` | 只下名字里含该子串的张量。**它会把 `mtp-manifest.json` 改写成这个子集**：交给 `mtp_pack.py` 之前必须再跑一次不带 `--only` 的 `fetch` |
| `--fallback-full-shard` | 若端点忽略 Range，就整片下载（每片数 GB）再在本地切出所需区间 |
| `--keep-shards` | 保留下载过的分片，而不是用完删掉 |
| `--no-sha256` | 不比对钉住 checkpoint 的 SHA-256（另一个 revision 时用） |

| 环境变量 | 默认值 |
|---|---|
| `MODELSCOPE_MTP_REPO` | `Qwen/Qwen3.8-Flash-Next` |
| `MODELSCOPE_MTP_REVISION` | `master`（该仓库的分支、tag 或 commit） |
| `MODELSCOPE_ENDPOINT` | `https://modelscope.cn` |
| `MTP_MS_SCHEME` | `api`（ModelScope 的文件 API；`resolve` 使用 `/models/<id>/resolve/<rev>/<path>` 形式） |

`fetch` 会写出每个张量一个原始文件，外加 `mtp-inventory.json`、`mtp-inventory.md`、`mtp-manifest.json` —— 与 `tools/mtp_fetch.py` 写出的完全一致，所以后面两步两条路都一样：

```sh
python tools/mtp_pack.py --src <data>/mtp --experts q2_0 --out <data>/mtp/mtp-q2_0.gguf
python tools/mtp_rt.py   --gguf <data>/mtp/mtp-q2_0.gguf --out <data>/mtp/rt
```

引擎的 `--mtp` 参数要的就是 `rt/experts.bin`（外加 `dense.bin`、`dense.txt`）；setup.py 会在 `<data>/mtp/rt/experts.bin` 找它，找到就整个跳过第 6 步。

重跑很便宜：已经存在且 SHA-256 正确的张量会被保留（打印 `already fetched, kept`），不会再发任何请求；中断的传输会从断点继续（`--jobs 1` 时是一个文件按大小续传，`--jobs N` 时是编号的 `.partNNNN` 分片，齐了再合并）。

## 3. 一次完整安装：两半都走 ModelScope

```sh
# 1. 模型文件（setup.py 的第 5 步）
python tools/download_model.py --model IQ3_XXS --vision

# 2. MTP 草稿层（setup.py 的第 6 步）：约 5 GB
python tools/mtp_fetch_ms.py probe  --out <data>/mtp
python tools/mtp_fetch_ms.py fetch  --out <data>/mtp --jobs 4
python tools/mtp_fetch_ms.py verify --out <data>/mtp
python tools/mtp_pack.py --src <data>/mtp --experts q2_0 --out <data>/mtp/mtp-q2_0.gguf
python tools/mtp_rt.py   --gguf <data>/mtp/mtp-q2_0.gguf --out <data>/mtp/rt

# 3. 引擎、pack 和启动脚本：setup.py 会认出上面这些文件，不再下载
START-HERE.bat --setup --family qwen --model IQ3_XXS      # Linux: ./setup.sh ...
```

`<data>` 就是 setup.py 使用的数据目录（除非给过 `--data-dir`/`--models-dir`，否则是 checkout 旁边的 `Strata-data`）；`tools/download_model.py --check` 和 `--dry-run` 会打印它将要用的路径。

## 4. 校验了什么，以及为什么

`tools/mtp_fetch_ms.py` 保留了 `tools/mtp_fetch.py` 的纪律（#327）：**range 读必须返回 206，且 `Content-Range` 正是所请求的区间**；每个张量都要比对钉住 checkpoint 的 SHA-256。忽略 `Range` 头的镜像或代理会返回 200 加整个分片，而代理还可能把它截到请求长度 —— 这种情形下「大小」和「收到内容的哈希」都抓不住，结果就是草稿层什么都接受不了，而且是静默的。所以不合规的端点会被直接拒绝，而不是靠猜；端点确实不会做 range 时才用 `--fallback-full-shard` 绕过去。

SHA-256 表是关于钉住 checkpoint 的**字节**的，这正是它在这里同样适用的原因：ModelScope 没有与「钉住的 Hugging Face commit」对应的东西，但这些张量要么是 checkpoint 的，要么不是。`verify` 完全离线，两个工具共用 `tensors/verified.json` 里彼此一致的那部分结论。

`tools/download_model.py` 既不更严也不更松：它用的就是 setup.py 对手工拷贝进来的文件所做的那套检查，并且在检查通过之前绝不写 `.done`。

## 5. 出问题时的对照

| 提示信息 | 含义 |
|---|---|
| `range request not honoured: HTTP 200 instead of 206` | 镜像或代理忽略了 `Range`。换 `--endpoint`，或用 `--fallback-full-shard`（整片下载，每片数 GB） |
| `sha256 ... is not the pinned checkpoint's` | 这个 ModelScope revision 不是 checkpoint 的字节：换 `--revision`，或确认后用 `--no-sha256` |
| `ModelScope could not serve <repo>` | 该仓库或 revision 没有镜像到那里：用 `--repo <id>`、`--revision <分支或 commit>`，或 `--source huggingface` |
| `the modelscope package is not installed` | `pip install modelscope`，或用 `--source huggingface` |
| `Qwen... Coder has no IQ3_XXS` | 该家族没有这个尺寸；消息里会列出它有什么（`--list` 看详情） |
| `has no image support in setup.py` | setup.py 从不对这个尺寸使用图像编码器（例如 UD-Q4_K_XL） |
| `no finish mark: <file>`（`--check` 报出） | 文件在，但 setup.py 会重新下载它；不带 `--check` 跑一次工具即可补上标记 |
| `is not whole: it is not as long as its own tensor directory says` | 文件被截断了：删掉再跑一次 |
| `mtp_rt.py` 报 `KeyError`（或张量形状断言失败） | `fetch --only` 在 `mtp-manifest.json` 里留下了子集，于是 `mtp_pack.py` 只打包了那些张量：不带 `--only` 再跑一次 `fetch` |

## 6. 测试，以及本机实测到的数字

两个工具都有离线测试（mock 掉 `urlopen` 的假 ModelScope 端点、假的 `snapshot_download`）：

```sh
python -m unittest tools.test_download_model tools.test_mtp_fetch_ms tools.test_mtp_fetch
```

在写这份文档的机器上（Windows，数据目录 `E:\Strata-data`）实测：

- `mtp_fetch_ms.py probe` 对 `Qwen/Qwen3.8-Flash-Next@master`：**31 个 mtp 张量、5.214 GB、28 个分片**，range 读返回 **206** —— 与 `tools/mtp_fetch.py` 从 Hugging Face 读到的清单一致。
- `mtp_fetch_ms.py verify` 对已有的 `mtp/tensors`：退出码 0（31 个张量都是 checkpoint 的）。
- `download_model.py --check --model IQ3_XXS`：两个分片都在且完整，47.04 GB + 28.80 GB（合计 75.84 GB，setup.py 的估算是 75.8 GB）。
- **模型文件所在的仓库**（`ISTA-DASLab/...`、`unsloth/...`）在这里**没有**从 ModelScope 试过；`--dry-run` 会显示确切的调用，某个仓库没有镜像时用 `--source huggingface` 兜底。