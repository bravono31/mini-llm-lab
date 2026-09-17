# Mini LLM Lab

LLM の学習と推論の過程を、実際の数値・パラメータを見ながら 1 ステップずつ学べる Web ページです。
ブラウザ内で本当に動く手書きのミニ GPT（2 層・2 ヘッド・d=16・約 8,000 パラメータ）を使い、
トークン化 → 埋め込み → 注意機構 → MLP → 出力確率 → サンプリング、そして 損失 → 逆伝播 → AdamW 更新 までを
すべて可視化します。

## 起動

```bash
npm install
npm run dev        # http://localhost:5173
```

```bash
npm test           # vitest：勾配検証（有限差分）、トークナイザ、学習で損失が下がること
npm run typecheck  # tsc --noEmit
npm run build      # 型チェック + 本番ビルド（dist/）
npm run pretrain   # 両コーパスを学習し src/data/pretrained-*.json を再生成（約 20 秒 × 2）
```

## 構成

| ディレクトリ | 内容 |
|---|---|
| `src/engine/` | 数値計算エンジン。自動微分は使わず、層ごとに forward / backward を明示実装（llm.c 方式）。中間活性がすべて名前付きバッファとして残るので UI がそのまま表示できる |
| `src/engine/tokenizer.ts` | ミニ BPE。マージ履歴とエンコード手順を記録し、UI で再生できる |
| `src/data/` | 日本語（ひらがな）と英語のコーパス、事前学習済み重み（JSON） |
| `src/ui/chapters/` | 8 つの章。各章は `Step[]`（解説 + 数式）と、ステップごとの可視化を持つ |
| `src/ui/viz/` | ヒートマップ、棒グラフ、注意の弧、損失曲線、行列積など SVG コンポーネント |
| `scripts/pretrain.ts` | Node で事前学習して JSON を書き出す |
| `tests/` | vitest |

## モデル

GPT-2 系の decoder-only Transformer を極小化したもの。Pre-LN、学習位置埋め込み、GELU、出力層は埋め込み表と共有（weight tying）、
AdamW（2 次元テンソルにのみ weight decay）。語彙は BPE で日本語 96 / 英語 64。

## デザイン

「和紙 × 墨」のライトテーマを既定とし、OS がダークなら自動追従（手動切替あり）。
見出しは Noto Serif JP、数値・トークンは JetBrains Mono。ヒートマップは符号付きの値に藍↔朱の発散スケール、
確率・注意重みに藍の順次スケールを使います。
