# Linux互換性監査レポート（全体コードベース）

## 1. 監査の結論（要約）

- **現状のままでは「完全にLinux対応」とは言えません。**
- 理由は、`FreeCAD`探索ロジックや起動スクリプトに **Windows固定のパス／区切り文字** が残っており、Linux環境で設定なしに実行すると起動失敗する可能性が高いためです。
- ただし、最適化ロジック自体（Optuna、評価フロー、サブプロセス管理の多く）はOS分岐を持っており、**ポイント修正でLinux実行可能性は高い**です。

---

## 2. 確認方法

### 2.1 静的確認（コード走査）

以下を中心に確認しました。

- OS依存コード（`os.name == "nt"`、`CREATE_NEW_PROCESS_GROUP` など）
- Windows固定パス（`C:\\...`、`Program Files`）
- FreeCAD / conda 環境探索ロジック
- エントリポイントとランチャー

### 2.2 実行確認（Linux上の単体テスト）

Linux環境で実行可能なテストを抜粋して実行し、基本的なロジックは動作することを確認しました（詳細は末尾のコマンドログ参照）。

---

## 3. 主な問題点と原因

## 3.1 `run_proto4.py` が Windows固定の conda パスを常にPATHへ追加

- 対象: `run_proto4.py`
- 該当箇所:
  - `conda_prefix = r"C:\Users\aokuni\AppData\Local\miniforge3\envs\fcad"`
  - `Library/bin` を前提にPATH追加

### 影響

- Linuxでも実行は継続する可能性がありますが、無効パスをPATHへ追加するため、環境解決が不安定になります。
- 実際にはLinuxの`CONDA_PREFIX`や`FREECAD_*`環境変数を使うべきです。

### 修正提案

1. `os.name == "nt"` のときだけ `Library/bin` を追加する。
2. Linuxでは `CONDA_PREFIX/lib` や `FREECAD_BIN` を優先する。
3. 無効パスを追加しないよう `Path.exists()` をチェックする。

### 想定副作用

- **副作用（軽微）**: PATH追加の順序が変わるため、同名ライブラリが複数ある環境で読み込まれる実体が変わる可能性。
- **対策**: 追加順序をログ出力し、`FREECAD_PYTHON`/`FREECAD_BIN`指定時はそれを最優先に固定。

---

## 3.2 `freecad_engine.py` の conda 探索がWindows前提（区切り文字とディレクトリ構造）

- 対象:
  - `src/v1/freecad_engine.py`
  - `src/proto4-codex/freecad_engine.py`
  - `src/proto4-claude/freecad_engine.py`
- 原因:
  - `CONDA_PREFIXES` を `split(";")` で分割（Linuxは通常 `:`）
  - `Library/bin`, `Library/lib` 前提（主にWindows condaレイアウト）
  - デフォルト候補が `C:\Users\...` 固定

### 影響

- Linuxでcondaを使っていても候補列挙に失敗し、`ImportError("FreeCAD not found ...")` へ到達しやすい。

### 修正提案

1. `CONDA_PREFIXES` は `os.pathsep` で分割する。
2. OSごとに探索パターンを分岐する。
   - Windows: `Library/bin`, `Library/lib`
   - Linux: `bin`, `lib`, `lib64`, `lib/pythonX.Y/site-packages`
3. Windows固定の候補 (`C:\Users\...`) は `os.name == "nt"` の場合のみ追加する。
4. `FREECAD_PYTHON` を受け取れる場合は最優先する（モジュールimportより前に解決可能）。

### 想定副作用

- **副作用（性能）**: 探索候補が増えるため初回importが数十ms〜数百ms遅くなる可能性。
- **副作用（挙動）**: 複数FreeCADが入っている環境では、従来と異なる候補が先に見つかる可能性。
- **対策**:
  - 明示的に `FREECAD_PYTHON` / `FREECAD_BIN` を優先
  - どの候補を採用したかをINFOログに必ず出す

---

## 3.3 `geometry_adapter.py` のデフォルト探索先がWindowsのみ

- 対象:
  - `src/v1/geometry_adapter.py`
  - `src/proto4-codex/geometry_adapter.py`
- 原因:
  - `FREECAD_BIN`未指定時のデフォルトが `C:\Program Files\FreeCAD 1.0\bin` のみ

### 影響

- Linuxで環境変数未設定の場合、FreeCAD Python解決に失敗しやすい。

### 修正提案

1. Linux向けデフォルト候補を追加:
   - `/usr/bin`
   - `/usr/lib/freecad/bin`
   - `/opt/freecad/bin`
   - `~/.local/bin`（必要に応じて）
2. `which freecadcmd` / `which python`（FreeCAD環境側）による探索を追加。
3. 失敗時エラーメッセージに「Linux推奨設定例」を含める。

### 想定副作用

- **副作用（誤検出）**: `python` が通常Pythonを指し、FreeCAD同梱Pythonでない場合に実行時エラーへ進む可能性。
- **対策**: ワーカー起動前に `import FreeCAD` のプリフライトを実施し、失敗時は候補を切替える。

---

## 3.4 実行導線にWindows専用ランチャーが含まれる

- 対象: `run_proto4.bat`

### 影響

- Linuxユーザーがランチャーをそのまま利用できない。

### 修正提案

- `run_proto4.sh` を追加し、READMEでLinux導線を明示。
- 既存Pythonエントリポイント（例: `scripts/run_v1.py`）へ統一誘導。

### 想定副作用

- **副作用（運用）**: 実行手順が増えてメンテ対象が増える。
- **対策**: バッチ/シェル双方を薄いラッパーにして本体ロジックはPythonへ寄せる。

---

## 3.5 ドキュメント上のOS要件がWindows中心

- 対象: `README.md`

### 影響

- 実装をLinux対応しても、利用者が公式非対応と解釈し導入を諦める可能性。

### 修正提案

- READMEのRequirementsを以下のように整理:
  - Linux/Windows両対応（ただしVEXIS/FreeCADの導入条件を明示）
  - 必須環境変数 (`FREECAD_PYTHON`, `FREECAD_BIN`, `CONDA_PREFIX`) の設定例を追記

### 想定副作用

- **副作用（サポート負荷）**: Linux利用者が増え、環境差分問い合わせが増加。
- **対策**: 「サポート対象ディストリ/バージョン」を明記し、非対象はベストエフォート扱いにする。

---

## 4. 優先度つき修正ロードマップ

1. **最優先（必須）**
   - `freecad_engine.py` 系3ファイルの `CONDA_PREFIXES` 分割と探索ロジックをOS非依存化
   - `geometry_adapter.py` 系2ファイルにLinux候補を追加
2. **高優先**
   - `run_proto4.py` のWindows固定PATH注入を条件化
3. **中優先**
   - `run_proto4.sh` 追加とREADME更新
4. **検証**
   - Linux CIジョブ（FreeCADあり/なし）を分けて追加

---

## 5. 参考: 実行したテスト／チェック

- `pytest -q tests/proto4_codex/test_geometry_adapter_subprocess.py tests/proto4_codex/test_runner_convergence.py tests/v1/test_constraints_domain.py`
- `pytest -q tests/proto4/test_proto4_codex_readiness.py tests/proto4_codex/test_cad_gate_io.py`

いずれもLinux上で成功しました（ただし、これらはFreeCAD/VEXIS実体を必要としない範囲の検証です）。

