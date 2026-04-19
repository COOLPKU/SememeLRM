Enhancing Lexical Relation Mining with Structured Sememe Knowledge
[![Paper](https://img.shields.io/badge/Paper-ACL%202026-blue)]()

Data and code for the paper "Enhancing Lexical Relation Mining with Structured Sememe Knowledge" accepted by ACL 2026 main.

## 🌟 Key Contributions
- We propose an automated STC pipeline, aiming to tackle the challenges of adopting structured sememe knowledge in annotation-scarce scenarios;
- We propose the SememeLRM method to fully leverage structured sememe knowledge for enhancing LRC and LE, achieving a notable 1.6% improvement on average across benchmarks;
- We present a potentially generalizable framework to leverage complete sememe trees in downstream tasks, helping to unlock the value of such
intralexical knowledge in more NLP applications.

## 📊 Data

### 1. Lexical Relation Classification (LRC) Datasets
We evaluate SememeLRM on five widely used LRC benchmarks:
- **BLESS** 
- **K&H+N** 
- **EVALution** 
- **CogALexV** 
- **ROOT09** 

These datasets jointly cover 10 relation types, including Random, Synonymy, Hypernymy, Co-hyponymy, Antonymy, Meronymy, Part_of, Event, Attribute, and Made_of.

### 2. Lexical Entailment (LE) Dataset
- **HyperLex** 

### 3. Unified Data Format
All data across the above benchmarks is unified into a single JSONL file (e.g., `test_data.jsonl`), with one JSON sample per line. Each sample contains a word pair, the relation label, the data source, and the DEF sememe tree information for both words.

Each sample includes the following fields:

|Field|Description|
|:----|:----|
|**word1**|The first word of the pair (string)|
|**word2**|The second word of the pair (string)|
|**rel**|The lexical relation label (e.g., `hyper`, `syn`, `random`, `event`, `attri`)|
|**source**|The source dataset name (e.g., `BLESS`, `K&H+N`, `EVALution`, `CogALexV`, `ROOT09`)|
|**word1_senses**|A dict containing sememe information for all senses of `word1` (see below)|
|**word2_senses**|A dict containing sememe information for all senses of `word2` (see below)|

The `word1_senses` / `word2_senses` field contains the following sub-fields:

|Sub-field|Description|
|:----|:----|
|**sememes**|A list of sememe lists, where each inner list contains the sememes of one sense|
|**triples**|A list of triple lists, where each inner list contains `(child, relation, parent)` triples describing the sememe tree edges of one sense|
|**main_sememes**|A list of main sememes (root nodes), one per sense|

A (truncated) example sample is:
```json
{
  "word1": "turtle",
  "word2": "live",
  "rel": "event",
  "source": "BLESS",
  "word1_senses": {
    "sememes": [["fish|鱼", "noun"]],
    "triples": [[["noun", "part_of_speech", "fish|鱼"], ["fish|鱼", "self_relation", "fish|鱼"]]],
    "main_sememes": ["fish|鱼"]
  },
  "word2_senses": {
    "sememes": [["reside|住下", "verb"], ["alive|活着", "verb"]],
    "triples": [[["verb", "part_of_speech", "reside|住下"]], [["verb", "part_of_speech", "alive|活着"]]],
    "main_sememes": ["reside|住下", "alive|活着"]
  }
}
```

The corresponding train / dev / test files are:
- `train_data.jsonl`
- `dev_data.jsonl`
- `test_data.jsonl`

The R-GNN module additionally requires a global sememe vocabulary and a graph description file (`./data/graph_data.json`), which specifies the full set of candidate sememes (`sememe_type` field).

## 🛠️ Code

The code of SememeLRM is provided in a single self-contained script: `train_deberta_pos_rgcn.py`. It implements:
- Sememe vocabulary, relation vocabulary, and sememe-tree edge-type vocabulary construction;
- Sample-level preprocessing that aligns word tokens, sense-level sememe embeddings, and prompt tokens in a unified DeBERTa input sequence;
- A three-layer **R-GCN** module for encoding DEF sememe trees into sense-level embeddings;
- A `RelationClassifier` wrapper that fuses the R-GCN sememe embeddings with the DeBERTa-v3 encoder for joint lexical relation prediction;
- Full training, validation, test evaluation, per-class F1 / confusion matrix reporting, and a detailed error analysis pipeline.

### Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.0 (with CUDA support)
- transformers
- datasets
- torch-geometric
- scikit-learn
- numpy, tqdm

### Usage

To train and evaluate SememeLRM on one of the LRC benchmarks (e.g., `CogALexV`), run:

```bash
python train_deberta_pos_rgcn.py \
    --train_path ./data/train_data.jsonl \
    --dev_path   ./data/dev_data.jsonl \
    --test_path  ./data/test_data.jsonl \
    --source     CogALexV \
    --deberta_name ./deberta_large \
    --rgcn_hidden_dim 1024 \
    --batch_size 32 \
    --lr 2e-5 \
    --epochs 10 \
    --weight_decay 1e-5 \
    --warmup_ratio 0.1 \
    --output_dir ./trainer_output_deberta_rgcn \
    --seed 42
```

To switch to a different benchmark, simply change the `--source` argument to one of `BLESS`, `K&H+N`, `EVALution`, `CogALexV`, or `ROOT09`. The script will automatically filter the unified JSONL files by the `source` field.

Key arguments:
- `--source`: which benchmark to filter from the unified JSONL data.
- `--deberta_name`: path or name of the DeBERTa-large backbone.
- `--rgcn_hidden_dim`: hidden dimension of the R-GCN module (should typically match the PLM hidden size, e.g., 1024 for DeBERTa-large).
- `--batch_size`, `--lr`, `--epochs`, `--weight_decay`, `--warmup_ratio`: standard training hyperparameters.
- `--output_dir`: directory for model checkpoints, logs, and final evaluation reports (`error_analysis.json`, `predictions.txt`, etc.).

After training, the script automatically performs:
1. Test-set evaluation with weighted F1 and macro F1;
2. Per-class F1 reporting and a full confusion matrix;
3. A detailed per-sample error analysis, saved to `error_analysis.json` and `predictions.txt`.

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex

```

## 📄 More Resources

For more work and resources related to the Chinese Object-Oriented Lexicon (COOL), Peking University, please refer to [this repository](https://github.com/COOLPKU) (to be released in the near future).

