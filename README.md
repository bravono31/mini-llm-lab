# Mini LLM Lab

LLM の学習と推論の過程を、実際の数値・パラメータを見ながら 1 ステップずつ学べる Web ページです。
ブラウザ内で本当に動く手書きのミニ GPT（2 層・2 ヘッド・d=16・約 8,000 パラメータ）を使い、
トークン化 → 埋め込み → 注意機構 → MLP → 出力確率 → サンプリング、そして 損失 → 逆伝播 → AdamW 更新 までを
すべて可視化します。

目次は 3 部構成です。

| 部 | 章 | 内容 |
|---|---|---|
| 全体像 | MAP | AI・機械学習・ニューラルネットワーク・深層学習・Transformer・LLM の入れ子と、自然言語処理・生成 AI との関係 |
| 機械学習編 | ML-1〜7 | LLM と同じ文集の「話題当て」を、人が作った単語辞書の Bag-of-Words とロジスティック回帰で学習・評価し、LLM と比べる |
| LLM 編 | LLM-1〜9 | ミニ GPT の推論・学習・RAG |

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
| `src/engine/classifier.ts` | ML 編の分類器。辞書による単語の区切り（動的計画法）、Bag-of-Words、softmax 回帰と勾配降下 |
| `src/data/ml-labels.ts` | ML 編の話題ラベル（人手）と単語辞書 |
| `src/data/` | 日本語（ひらがな）と英語のコーパス、事前学習済み重み（JSON） |
| `src/ui/chapters/` | 全体像・ML 編 7 章・LLM 編 9 章。各章は `Step[]`（解説 + 数式）と、ステップごとの可視化を持つ |
| `src/ui/viz/` | ヒートマップ、棒グラフ、注意の弧、損失曲線、行列積など SVG コンポーネント |
| `scripts/pretrain.ts` | Node で事前学習して JSON を書き出す |
| `tests/` | vitest |

## モデル

GPT-2 系の decoder-only Transformer を極小化したもの。Pre-LN、学習位置埋め込み、GELU、出力層は埋め込み表と共有（weight tying）、
AdamW（2 次元テンソルにのみ weight decay）。語彙は BPE で日本語 96 / 英語 64。

## デザイン

「和紙 × 墨」のライトテーマを既定とし、OS がダークなら自動追従（手動切替あり）。
見出しは Noto Serif JP、数値・トークンは JetBrains Mono。ヒートマップは符号付きの値に藍↔朱の発散スケール、
確率・注意重みに朱の順次スケール（濃いほど大きい）を使います。
