import json
import torch
import torch.nn as nn
from collections import defaultdict
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, Batch
from transformers import (
    DebertaV2Tokenizer, DebertaV2Model,
    TrainingArguments, Trainer,
    EarlyStoppingCallback, AutoConfig
)
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from torch.optim import AdamW
import argparse
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import os
import copy
import logging
from datasets import Dataset, DatasetDict
from datetime import datetime

# ====================== Logging Configuration ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ====================== Utility: Build Vocabularies (logic unchanged) ======================
def build_sememe_vocab(data_paths, graph_data_path='./data/graph_data.json'):
    """
    Build the sememe vocabulary, supplementing sememes that appear in the dataset but
    are not present in graph_data.
    """
    # Load sememes from graph_data
    with open(graph_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sememe_type = data.get('sememe_type', [])
    if not sememe_type:
        raise ValueError("Field 'sememe_type' not found in graph_data.json")
    
    # Initial sememe dictionary
    sememe2id = {sem: idx for idx, sem in enumerate(sememe_type)}
    sememe_id2name = {v: k for k, v in sememe2id.items()}
    initial_sememe_count = len(sememe2id)
    logger.info(f"Initial sememe vocabulary size: {initial_sememe_count}, index range: [0, {initial_sememe_count-1}]")
    
    # Collect all sememes appearing in the dataset
    dataset_semes = set()
    for data_path in data_paths:
        with open(data_path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                sample = json.loads(line.strip())
                
                # Collect sememes of word1
                word1_senses = sample["word1_senses"]
                word1_sememes = []
                for tmp_list in word1_senses["sememes"]:
                    word1_sememes+=tmp_list
                dataset_semes.update(word1_sememes)
                # Collect sememes of word2
                word2_senses = sample["word2_senses"]
                word2_sememes = []
                for tmp_list in word2_senses["sememes"]:
                    word2_sememes+=tmp_list
                dataset_semes.update(word2_sememes)
    
    # Supplement missing sememes
    new_semes = []
    for sem in dataset_semes:
        if sem not in sememe2id:
            sememe2id[sem] = len(sememe2id)
            sememe_id2name[len(sememe_id2name)] = sem
            new_semes.append(sem)
    
    # Print supplement information
    logger.info(f"Number of sememes supplemented from the dataset: {len(new_semes)}")
    if new_semes:
        logger.info(f"Supplemented sememe list: {new_semes}")
    logger.info(f"Final sememe vocabulary size: {len(sememe2id)}, index range: [0, {len(sememe2id)-1}]")
    
    return sememe2id, sememe_id2name

def build_rel_vocab(data_paths, source):
    """Build the sample-level relation-type vocabulary (filtered by the `source` argument)."""
    rel_vocab = defaultdict(int)
    rel_id = 0
    for data_path in data_paths:
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                
                if sample.get('source') != source:  # Use the passed-in source argument instead of a hard-coded one
                    continue
                if 'random' in sample['rel']:
                    continue
                rel = sample.get("rel", "")
                if rel and rel not in rel_vocab:
                    rel_vocab[rel] = rel_id
                    rel_id += 1
    logger.info(f"Number of sample relation types: {len(rel_vocab)}, types: {list(rel_vocab.keys())}")
    return rel_vocab

def build_edge_type_vocab(data_paths):
    """Build the vocabulary of sememe edge types (the second element of each triple)."""
    edge_type_vocab = defaultdict(int)
    edge_type_id = 0
    for data_path in data_paths:
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                # Process the triples of word1
                for triples in sample["word1_senses"]["triples"]:
                    for triple in triples:
                        assert len(triple) == 3 
                        edge_type = triple[1]
                        if edge_type not in edge_type_vocab:
                            edge_type_vocab[edge_type] = edge_type_id
                            edge_type_id += 1
                for triples in sample["word2_senses"]["triples"]:
                    for triple in triples:
                        assert len(triple) == 3 
                        edge_type = triple[1]
                        if edge_type not in edge_type_vocab:
                            edge_type_vocab[edge_type] = edge_type_id
                            edge_type_id += 1
    logger.info(f"Number of sememe edge types: {len(edge_type_vocab)}, types: {list(edge_type_vocab.keys())}")
    return edge_type_vocab

def build_sememe_graph(valid_semes, triples, sememe2id, edge_type_vocab):
    """Build a sememe graph (including edge types)."""
    if not valid_semes:
        # Empty graph: includes the edge_type field
        return Data(
            x=torch.tensor([[0]], dtype=torch.long),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_type=torch.empty((0,), dtype=torch.long)
        )
    
    x = torch.tensor([sememe2id[sem] for sem in valid_semes], dtype=torch.long).unsqueeze(1)
    
    edge_index = []
    edge_type = []
    for src_sem, rel, dst_sem in triples:  # Adapt to the new triple format
        if src_sem not in valid_semes or dst_sem not in valid_semes:
            continue
        if rel not in edge_type_vocab:
            print('missing rel_type', rel)
            continue
        src_idx = valid_semes.index(src_sem)
        dst_idx = valid_semes.index(dst_sem)
        edge_index.append([src_idx, dst_idx])
        # Convert edge type to id
        edge_type.append(edge_type_vocab[rel])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long) if edge_type else torch.empty((0,), dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_type=edge_type)

# ====================== Data Preprocessing (logic unchanged) ======================
def preprocess_sample(sample, tokenizer, sememe2id, rel_vocab, edge_type_vocab, source):
    """Preprocess a single sample (with the added `source` argument)."""
    if sample.get('source') != source:  # Use the passed-in `source` argument
        return None
    if 'random' in sample['rel']:
        return None
    
    word1 = sample["word1"]
    word2 = sample["word2"]
    word1_senses = sample["word1_senses"]
    word2_senses = sample["word2_senses"]
    
    #word1_main_semes = word1_senses["main_sememes"]
    word1_sememes = []
    for tmp_semes in word1_senses["sememes"]:
        word1_sememes += tmp_semes
    word1_main_semes = word1_sememes
    word1_triples = []
    for tmp_triples in word1_senses["triples"]:
        word1_triples += tmp_triples

    #word2_main_semes = word2_senses["main_sememes"]
    word2_sememes = []
    for tmp_semes in word2_senses["sememes"]:
        word2_sememes += tmp_semes
    word2_main_semes = word2_sememes
    word2_triples = []
    for tmp_triples in word2_senses["triples"]:
        word2_triples += tmp_triples

    # Build the input token sequence
    input_tokens = [tokenizer.cls_token]
    input_tokens.extend(tokenizer.tokenize(f"Today, I finally discovered the relation between"))
    
    # Parameter arguments (position markers)
    position_markers = {
        'word1_start': len(input_tokens),
    }
    input_tokens.extend(tokenizer.tokenize(f"(word1: "))
    input_tokens.extend(tokenizer.tokenize("#"))
    input_tokens.extend(tokenizer.tokenize(f"{word1}") if word1 else [])
    input_tokens.extend(tokenizer.tokenize("#"))
    input_tokens.extend(tokenizer.tokenize(f"(word1 sense information: "))
    input_tokens.extend(word1_main_semes)
    input_tokens.extend(tokenizer.tokenize(f")"))
    position_markers['word1_end'] = len(input_tokens) - 1
    
    position_markers['and_start'] = len(input_tokens)
    input_tokens.extend(tokenizer.tokenize("and"))
    position_markers['and_end'] = len(input_tokens) - 1

    position_markers['word2_start'] = len(input_tokens)
    input_tokens.extend(tokenizer.tokenize(f"(word2: "))
    input_tokens.extend(tokenizer.tokenize("#"))
    input_tokens.extend(tokenizer.tokenize(f"{word2}") if word2 else [])
    input_tokens.extend(tokenizer.tokenize("#"))
    input_tokens.extend(tokenizer.tokenize(f"(word2 sense information: "))
    input_tokens.extend(word2_main_semes)
    input_tokens.extend(tokenizer.tokenize(f")"))
    position_markers['word2_end'] = len(input_tokens) - 1
    
    input_tokens.append(tokenizer.sep_token)
    position_markers['sep'] = len(input_tokens) - 1

    # Encode token IDs
    input_ids = []
    sememe_positions = []
    sememe_names = []
    for idx_token, token in enumerate(input_tokens):
        if token in sememe2id:
            input_ids.append(tokenizer.unk_token_id)
            sememe_positions.append(idx_token)
            sememe_names.append(token)
        else:
            token_id = tokenizer.convert_tokens_to_ids(token) if token in tokenizer.get_vocab() else tokenizer.unk_token_id
            input_ids.append(token_id)

    # Filter valid sememes
    valid_word1_semes = []
    for sem in word1_sememes:
        if sem not in valid_word1_semes and sem in sememe2id:
            valid_word1_semes.append(sem)
    
    valid_word2_semes = []
    for sem in word2_sememes:
        if sem not in valid_word2_semes and sem in sememe2id:
            valid_word2_semes.append(sem)

    # Build sememe graphs (with edge types)
    def serialize_graph(graph):
        return {
            'x': graph.x.numpy().tolist(),
            'edge_index': graph.edge_index.numpy().tolist(),
            'edge_type': graph.edge_type.numpy().tolist()
        }
    
    word1_g = build_sememe_graph(valid_word1_semes, word1_triples, sememe2id, edge_type_vocab)
    word2_g = build_sememe_graph(valid_word2_semes, word2_triples, sememe2id, edge_type_vocab)
    
    # Indices of main sememes
    word1_main_indices = [valid_word1_semes.index(s) for s in word1_main_semes if s in valid_word1_semes]
    word2_main_indices = [valid_word2_semes.index(s) for s in word2_main_semes if s in valid_word2_semes]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "word1_g": serialize_graph(word1_g),
        "word2_g": serialize_graph(word2_g),
        "word1_main_indices": word1_main_indices,
        "word2_main_indices": word2_main_indices,
        "sememe_positions": sememe_positions,
        "sememe_names": sememe_names,
        "rel_label": rel_vocab[sample["rel"]],
        "word1_start": position_markers['word1_start'],
        "word1_end": position_markers['word1_end'],
        "word2_start": position_markers['word2_start'],
        "word2_end": position_markers['word2_end'],
        "and_start": position_markers['and_start'],
        "and_end": position_markers['and_end'],
        "sep_pos": position_markers['sep'],
        "rel_name": sample["rel"]
    }

def load_and_preprocess_data(train_path, dev_path, test_path, tokenizer, sememe2id, rel_vocab, edge_type_vocab, source):
    """Load and preprocess the data (with the added `source` argument)."""
    def load_jsonl(path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    # Load raw data
    train_data = load_jsonl(train_path)
    dev_data = load_jsonl(dev_path)
    test_data = load_jsonl(test_path)
    
    # Filter and preprocess
    def filter_and_map(data, source, cog=False):  # added `source` argument
        processed = []
        for sample in tqdm(data):
            if sample['source'] != source:
                continue
            if 'random' in sample['rel'] and cog==True:continue
            res = preprocess_sample(sample, tokenizer, sememe2id, rel_vocab, edge_type_vocab, source)
            if res is not None:
                processed.append(res)
        return processed
    
    logger.info("Preprocessing training set...")
    train_processed = filter_and_map(train_data, source, True)
    logger.info("Preprocessing validation set...")
    dev_processed = filter_and_map(dev_data, source, True)
    logger.info("Preprocessing test set...")
    test_processed = filter_and_map(test_data, source, True)
    
    # Convert to HuggingFace Dataset
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_processed),
        "validation": Dataset.from_list(dev_processed),
        "test": Dataset.from_list(test_processed)
    })
    
    # Dynamic padding
    def pad_function(examples):
        max_len = max(len(ids) for ids in examples["input_ids"])
        
        input_ids = []
        attention_mask = []
        for ids, mask in zip(examples["input_ids"], examples["attention_mask"]):
            pad_len = max_len - len(ids)
            input_ids.append(ids + [0] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word1_g": examples["word1_g"],
            "word2_g": examples["word2_g"],
            "word1_main_indices": examples["word1_main_indices"],
            "word2_main_indices": examples["word2_main_indices"],
            "sememe_positions": examples["sememe_positions"],
            "sememe_names": examples["sememe_names"],
            "rel_label": examples["rel_label"],
            "word1_start": examples["word1_start"],
            "word1_end": examples["word1_end"],
            "word2_start": examples["word2_start"],
            "word2_end": examples["word2_end"],
        }
    
    # Set format and batch padding
    dataset_dict = dataset_dict.map(
        pad_function,
        batched=True,
        batch_size=None,
        desc="Padding sequences"
    )
    
    # Convert tensor-typed columns
    tensor_columns = ["input_ids", "attention_mask", "rel_label"]
    for split in dataset_dict:
        dataset_dict[split].set_format(
            type="torch",
            columns=tensor_columns,
            output_all_columns=True
        )
    
    logger.info(f"Data loading finished: train={len(dataset_dict['train'])}, val={len(dataset_dict['validation'])}, test={len(dataset_dict['test'])}")
    return dataset_dict

# ====================== Custom Data Collator (logic unchanged) ======================
def custom_data_collator(features):
    """Custom data collator (compatible with the edge-type field)."""
    batch = {}
    
    # Handle tensor-typed fields
    tensor_keys = ["input_ids", "attention_mask", "rel_label"]
    for key in tensor_keys:
        if key in features[0]:
            batch[key] = torch.stack([f[key] for f in features])
    
    # Handle list/dict-typed fields (including edge types)
    non_tensor_keys = [
        "word1_g", "word2_g", "word1_main_indices", "word2_main_indices",
        "sememe_positions", "sememe_names", "word1_start", "word1_end",
        "word2_start", "word2_end"
    ]
    for key in non_tensor_keys:
        if key in features[0]:
            batch[key] = [f[key] for f in features]
    
    return batch

# ====================== Model Components (core change: RGAT -> RGCN) ======================
class SememeRGCN(nn.Module):
    """Relational Graph Convolutional Network (R-GCN) over sememes - replaces the original RGAT."""
    def __init__(self, num_semes, num_relations, embed_dim, rgcn_hidden_dim, sememe_id2name, tokenizer):
        super().__init__()
        self.sememe_id2name = sememe_id2name
        self.tokenizer = tokenizer
        self.num_relations = num_relations  # Number of edge types
        sememe_embed_dim = embed_dim
        
        # Sememe embedding layer
        self.sememe_emb = nn.Embedding(num_semes, sememe_embed_dim)
        self.num_semes = num_semes
        self._emb_init_flag = False

        # Three-layer R-GCN (Relational GCN) - replaces the original RGAT
        self.rgcn1 = pyg_nn.RGCNConv(
            in_channels=sememe_embed_dim,
            out_channels=rgcn_hidden_dim,
            num_relations=num_relations,
            
        )
        self.rgcn2 = pyg_nn.RGCNConv(
            in_channels=rgcn_hidden_dim,
            out_channels=rgcn_hidden_dim,
            num_relations=num_relations,
            
        )
        self.rgcn3 = pyg_nn.RGCNConv(
            in_channels=rgcn_hidden_dim,
            out_channels=embed_dim,
            num_relations=num_relations,
            
        )

        # Activation / normalization / dropout (adjust LayerNorm dim; no multi-head anymore)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.ln1 = nn.LayerNorm(rgcn_hidden_dim)  # No multi-head concat, use hidden_dim directly
        self.ln2 = nn.LayerNorm(rgcn_hidden_dim)

    def _init_sememe_embeddings(self, deberta_word_emb):
        """Initialize the sememe embeddings (no changes)."""
        device = self.sememe_emb.weight.device
        for sem_id in range(len(self.sememe_id2name)):
            sem_name = self.sememe_id2name[sem_id]
            if '|' in sem_name:
                english_part = sem_name.split('|')[0].strip()
            else:
                english_part = sem_name.strip()
            
            tokens = self.tokenizer.tokenize(english_part)
            if not tokens:
                tokens = [self.tokenizer.unk_token]
            
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            with torch.no_grad():
                if len(token_ids) > 0:
                    token_ids_tensor = torch.tensor(token_ids, device=deberta_word_emb.weight.device)
                    embeddings = deberta_word_emb(token_ids_tensor)
                    avg_embedding = embeddings.mean(dim=0)
                else:
                    avg_embedding = torch.zeros(self.sememe_emb.embedding_dim, device=device)
                
                avg_embedding = avg_embedding.to(device).clone()
                self.sememe_emb.weight[sem_id] = avg_embedding
        
        self._emb_init_flag = True
        logger.info("Sememe embedding initialization finished (aligned with DeBERTa-v3)")

    def forward(self, batch, deberta_word_emb=None):
        """RGCN forward pass (single-head, simplified logic)."""
        # First-time initialization
        if not self._emb_init_flag and deberta_word_emb is not None:
            self._init_sememe_embeddings(deberta_word_emb)
        
        # Handle empty graph
        if batch.x.numel() == 0:
            return torch.empty((0, self.sememe_emb.embedding_dim), device=batch.x.device)
        
        # Sanity check
        sem_id = batch.x.squeeze()
        if (sem_id < 0).any() or (sem_id >= self.num_semes).any():
            invalid_ids = sem_id[(sem_id < 0) | (sem_id >= self.num_semes)]
            raise ValueError(f"Invalid sememe IDs: {invalid_ids.tolist()}, vocabulary size: {self.num_semes}")

        # RGCN forward pass (no multi-head, dims flow through directly)
        x = self.sememe_emb(sem_id)  # [num_nodes, embed_dim]
        
        # First RGCN layer
        x = self.rgcn1(x, batch.edge_index, batch.edge_type)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Second RGCN layer
        x = self.rgcn2(x, batch.edge_index, batch.edge_type)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Third RGCN layer
        x = self.rgcn3(x, batch.edge_index, batch.edge_type)
        
        return x

class SememeAwareEmbedding(nn.Module):
    """Sememe-aware embedding (no changes)."""
    def __init__(self, deberta_dim, sememe_id2name):
        super().__init__()
        self.deberta_dim = deberta_dim
        self.sememe_id2name = sememe_id2name

    def forward(self, input_ids, deberta_word_emb, sememe_rgcn,  # renamed to sememe_rgcn
                word1_g_batch, word2_g_batch,
                word1_global_indices, word2_global_indices,
                word1_main_counts, word2_main_counts,
                sememe_positions, sememe_names):
        batch_size, seq_len = input_ids.shape
        deberta_emb = deberta_word_emb(input_ids)
        
        # Call the RGCN (replacing the original RGAT)
        word1_gat_emb = sememe_rgcn(word1_g_batch, deberta_word_emb=deberta_word_emb)
        word2_gat_emb = sememe_rgcn(word2_g_batch, deberta_word_emb=deberta_word_emb)
        
        def build_sem_map(g_batch, gat_emb):
            sem_map = {}
            if g_batch.x.numel() == 0 or len(gat_emb) == 0:
                return sem_map
            
            sem_ids = g_batch.x.squeeze().tolist()
            if not isinstance(sem_ids, list):
                sem_ids = [sem_ids]
            
            for sem_id, emb in zip(sem_ids, gat_emb):
                sem_name = self.sememe_id2name.get(sem_id, None)
                if sem_name:
                    sem_map[sem_name] = emb
            return sem_map
        
        word1_sem_map = build_sem_map(word1_g_batch, word1_gat_emb)
        word2_sem_map = build_sem_map(word2_g_batch, word2_gat_emb)
        sem_map = {**word1_sem_map, **word2_sem_map}
        
        combined_emb = deberta_emb.clone()
        for i in range(batch_size):
            positions = sememe_positions[i]
            names = sememe_names[i]
            for pos, name in zip(positions, names):
                if pos >= seq_len or name not in sem_map:
                    continue
                combined_emb[i, pos] = sem_map[name]

        return combined_emb

class RelationClassifier(nn.Module):
    """Relation classifier (MoE removed, adapted to R-GCN)."""
    def __init__(self, deberta_name, sememe2id, sememe_id2name, num_classes, rgcn_hidden_dim, 
                 edge_type_vocab, tokenizer, expert_hidden_dim=None):
        super().__init__()
        # Load the DeBERTa-v3 model
        self.deberta = DebertaV2Model.from_pretrained(deberta_name)
        self.config = self.deberta.config
        self.config.problem_type = "single_label_classification"
        
        self.deberta_dim = self.deberta.config.hidden_size
        self.max_position_embeddings = self.deberta.config.max_position_embeddings
        
        self.special_token_emb = nn.Embedding(5, self.deberta_dim)
        nn.init.xavier_uniform_(self.special_token_emb.weight)
        
        # Initialize the R-GCN (replaces the original RGAT; num_heads removed)
        self.sememe_rgcn = SememeRGCN(  # class and argument names updated
            num_semes=len(sememe2id),
            num_relations=len(edge_type_vocab),
            embed_dim=self.deberta_dim,
            rgcn_hidden_dim=rgcn_hidden_dim,
            sememe_id2name=sememe_id2name,
            tokenizer=tokenizer
        )

        # Initialize the sememe-aware embedding (adapted to R-GCN)
        self.sememe_aware_emb = SememeAwareEmbedding(
            deberta_dim=self.deberta_dim,
            sememe_id2name=sememe_id2name
        )
        
        # Single classifier (no changes)
        self.expert_hidden_dim = self.deberta_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(self.deberta_dim * 4, self.expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.expert_hidden_dim, num_classes)
        )
        
        self.cls_pos = 0
        self.special_token_start = 1
        self.special_token_end = 5
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def _deserialize_graph(self, graph_dict, device):
        """Deserialize graph data (no changes)."""
        x = torch.tensor(graph_dict['x'], dtype=torch.long, device=device)
        edge_index = torch.tensor(graph_dict['edge_index'], dtype=torch.long, device=device)
        edge_type = torch.tensor(graph_dict['edge_type'], dtype=torch.long, device=device)
        return Data(x=x, edge_index=edge_index, edge_type=edge_type)

    def _get_graph_batch(self, graph_list, device):
        """Batch-process a list of graphs (no changes)."""
        data_list = [self._deserialize_graph(g, device) for g in graph_list]
        return Batch.from_data_list(data_list)

    def _get_global_indices(self, graph_list, main_indices_list):
        """Compute global node indices (no changes)."""
        offsets = [0]
        for g in graph_list[:-1]:
            g_data = self._deserialize_graph(g, 'cpu')
            offsets.append(offsets[-1] + g_data.x.shape[0])
        
        global_indices = []
        main_counts = []
        for i, indices in enumerate(main_indices_list):
            main_counts.append(len(indices))
            global_indices.extend([idx + offsets[i] for idx in indices])
        
        return global_indices, main_counts

    def forward(self, input_ids, attention_mask, rel_label=None, **kwargs):
        """Forward pass (adapted to R-GCN, other logic unchanged)."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if seq_len > self.max_position_embeddings:
            raise ValueError(f"Sequence length {seq_len} exceeds the model's max limit {self.max_position_embeddings}")

        # Parse extra arguments
        word1_g_list = kwargs.get('word1_g', [])
        word2_g_list = kwargs.get('word2_g', [])
        word1_main_indices_list = kwargs.get('word1_main_indices', [])
        word2_main_indices_list = kwargs.get('word2_main_indices', [])
        sememe_positions = kwargs.get('sememe_positions', [])
        sememe_names = kwargs.get('sememe_names', [])
        
        # Position arguments
        word1_starts = kwargs.get('word1_start', [0]*batch_size)
        word1_ends = kwargs.get('word1_end', [0]*batch_size)
        word2_starts = kwargs.get('word2_start', [0]*batch_size)
        word2_ends = kwargs.get('word2_end', [0]*batch_size)

        # Build the graph batches
        word1_g_batch = self._get_graph_batch(word1_g_list, device)
        word2_g_batch = self._get_graph_batch(word2_g_list, device)
        
        # Global indices
        word1_global_indices, word1_main_counts = self._get_global_indices(word1_g_list, word1_main_indices_list)
        word2_global_indices, word2_main_counts = self._get_global_indices(word2_g_list, word2_main_indices_list)

        # Produce the fused embeddings (calls the R-GCN)
        combined_emb = self.sememe_aware_emb(
            input_ids=input_ids,
            deberta_word_emb=self.deberta.embeddings.word_embeddings,
            sememe_rgcn=self.sememe_rgcn,  # Pass the R-GCN instance
            word1_g_batch=word1_g_batch,
            word2_g_batch=word2_g_batch,
            word1_global_indices=word1_global_indices,
            word2_global_indices=word2_global_indices,
            word1_main_counts=word1_main_counts,
            word2_main_counts=word2_main_counts,
            sememe_positions=sememe_positions,
            sememe_names=sememe_names
        )

        # DeBERTa-v3 encoding
        outputs = self.deberta(
            attention_mask=attention_mask,
            inputs_embeds=combined_emb
        )
        last_hidden_state = outputs.last_hidden_state

        # Compute mean representations of word1 / word2
        word1_avg_list = []
        word2_avg_list = []
        for i in range(batch_size):
            word1_start = min(max(word1_starts[i], 0), seq_len)
            word1_end = min(max(word1_ends[i], word1_start), seq_len)
            word2_start = min(max(word2_starts[i], word1_end + 1), seq_len)
            word2_end = min(max(word2_ends[i], word2_start), seq_len)
            
            word1_mask = attention_mask[i, word1_start:word1_end+1]
            word1_emb = last_hidden_state[i, word1_start:word1_end+1]
            if word1_mask.sum() > 0:
                word1_avg = (word1_emb * word1_mask.unsqueeze(-1)).sum(dim=0) / word1_mask.sum()
            else:
                word1_avg = torch.zeros(self.deberta_dim, device=device)
            
            word2_mask = attention_mask[i, word2_start:word2_end+1]
            word2_emb = last_hidden_state[i, word2_start:word2_end+1]
            if word2_mask.sum() > 0:
                word2_avg = (word2_emb * word2_mask.unsqueeze(-1)).sum(dim=0) / word2_mask.sum()
            else:
                word2_avg = torch.zeros(self.deberta_dim, device=device)
            
            word1_avg_list.append(word1_avg)
            word2_avg_list.append(word2_avg)
        
        word1_avg = torch.stack(word1_avg_list)
        word2_avg = torch.stack(word2_avg_list)
        diff_avg = word1_avg - word2_avg
        cls_emb = last_hidden_state[:, 0, :]
        
        concat_features = torch.cat([cls_emb, word1_avg, word2_avg, diff_avg], dim=-1)
        
        # Forward pass of the single classifier
        logits = self.classifier(concat_features)
        
        # Return loss during training and logits during evaluation
        if self.training and rel_label is not None:
            ce_loss = self.loss_fn(logits, rel_label)
            return {"loss": ce_loss, "logits": logits}
        else:
            return {"logits": logits}

# ====================== Custom Trainer (no changes) ======================
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(** inputs)
        
        if "loss" in outputs:
            loss = outputs["loss"]
        else:
            logits = outputs["logits"]
            labels = inputs["rel_label"]
            loss = model.loss_fn(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# ====================== Evaluation and Error Analysis (no changes) ======================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    
    weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
    return {
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1
    }

def get_detailed_test_metrics(model, test_dataset, rel_vocab, device):
    """Compute per-class F1 on the test set and the full confusion matrix."""
    model.eval()
    id2rel = {v: k for k, v in rel_vocab.items()}
    rel_names = [id2rel[i] for i in range(len(id2rel))]
    
    all_preds = []
    all_labels = []
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=custom_data_collator
    )
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Computing detailed test metrics"):
            for key in ["input_ids", "attention_mask", "rel_label"]:
                if key in batch:
                    batch[key] = batch[key].to(device)
            
            outputs = model(**batch)
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch["rel_label"].cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Compute per-class F1 and classification report
    logger.info("\n" + "="*80)
    logger.info("Detailed classification report (per-class F1)")
    logger.info("="*80)
    cls_report = classification_report(all_labels, all_preds, target_names=rel_names, zero_division=0)
    logger.info(f"\n{cls_report}")
    
    # Compute and print the confusion matrix
    logger.info("\n" + "="*80)
    logger.info("Full confusion matrix")
    logger.info("="*80)
    conf_mat = confusion_matrix(all_labels, all_preds)
    # Pretty-print the confusion matrix (with class names)
    logger.info(f"Class order: {rel_names}")
    logger.info(f"\nConfusion matrix:\n{conf_mat}")
    
    # Return detailed metrics
    cls_report_dict = classification_report(all_labels, all_preds, target_names=rel_names, output_dict=True, zero_division=0)
    return {
        "classification_report": cls_report_dict,
        "confusion_matrix": conf_mat.tolist(),
        "rel_names": rel_names
    }




def analyze_errors(model, test_dataset, tokenizer, rel_vocab, device, save_path="error_analysis.json", pred_save_path="predictions.txt"):
    """
    Analyze the model's errors on the test set and save the per-sample predictions.
    
    Args:
        model: the trained model
        test_dataset: the test dataset
        tokenizer: the tokenizer
        rel_vocab: relation label vocabulary (rel_name -> id)
        device: compute device (cpu/cuda)
        save_path: path for saving the error-analysis statistics
        pred_save_path: path for saving per-sample predictions
    """
    model.eval()
    id2rel = {v: k for k, v in rel_vocab.items()}
    
    error_stats = {
        "total_samples": 0,
        "total_errors": 0,
        "error_rate": 0.0,
        "errors_by_true_label": defaultdict(int),
        "errors_by_pred_label": defaultdict(int),
        "confusion_matrix": defaultdict(lambda: defaultdict(int)),
        "per_class_metrics": {}
    }
    total_by_class = defaultdict(int)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=custom_data_collator
    )
    
    all_preds = []
    all_labels = []
    all_rel_names = []
    # Added: record sample index, predicted label, and true label for each sample
    sample_predictions = []
    
    with torch.no_grad():
        # Added: track the current sample index
        sample_idx = 0
        for batch in tqdm(test_loader, desc="Analyzing error samples"):
            for key in ["input_ids", "attention_mask", "rel_label"]:
                if key in batch:
                    batch[key] = batch[key].to(device)
            
            outputs = model(** batch)
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch["rel_label"].cpu().numpy()
            rel_names = [test_dataset[i]["rel_name"] for i in range(len(preds))]
            
            # Added: record each sample's prediction (with its index) in the current batch
            for i in range(len(preds)):
                pred_rel = id2rel[preds[i]]   # predicted relation name
                true_rel = id2rel[labels[i]]  # ground-truth relation name
                sample_predictions.append(f"{sample_idx + i} {pred_rel} {true_rel}")
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_rel_names.extend(rel_names)
            # Update the sample index
            sample_idx += len(preds)
    
    # Original logic: aggregate error statistics
    for pred, true, rel_name in zip(all_preds, all_labels, all_rel_names):
        true_rel = id2rel[true]
        pred_rel = id2rel[pred]
        total_by_class[true_rel] += 1
        error_stats["total_samples"] += 1
        error_stats["confusion_matrix"][true_rel][pred_rel] += 1
        
        if true != pred:
            error_stats["total_errors"] += 1
            error_stats["errors_by_true_label"][true_rel] += 1
            error_stats["errors_by_pred_label"][pred_rel] += 1
    
    error_stats["error_rate"] = error_stats["total_errors"] / error_stats["total_samples"] if error_stats["total_samples"] > 0 else 0.0
    for rel in total_by_class:
        total = total_by_class[rel]
        errors = error_stats["errors_by_true_label"].get(rel, 0)
        accuracy = (total - errors) / total if total > 0 else 0.0
        error_stats["per_class_metrics"][rel] = {
            "total": total,
            "correct": total - errors,
            "errors": errors,
            "accuracy": accuracy,
            "error_rate": 1 - accuracy
        }
    
    # Print logs
    logger.info("\n" + "="*80)
    logger.info("Error analysis report")
    logger.info("="*80)
    logger.info(f"Total samples: {error_stats['total_samples']}")
    logger.info(f"Error samples: {error_stats['total_errors']}")
    logger.info(f"Overall error rate: {error_stats['error_rate']:.2%}")
    logger.info(f"Overall accuracy: {1 - error_stats['error_rate']:.2%}")
    
    logger.info("\nError distribution by class (grouped by true label):")
    for rel, count in error_stats["errors_by_true_label"].items():
        total = total_by_class[rel]
        error_rate = count / total if total > 0 else 0.0
        logger.info(f"  {rel}: {count}/{total} ({error_rate:.2%})")
    
    logger.info("\nPer-class accuracy:")
    for rel, metrics in sorted(error_stats["per_class_metrics"].items(), key=lambda x: x[1]["accuracy"]):
        logger.info(f"  {rel}: {metrics['accuracy']:.2%} (correct: {metrics['correct']}/{metrics['total']})")
    
    # Save the aggregated error-analysis statistics
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(error_stats, f, ensure_ascii=False, indent=2)
    logger.info(f"Error-analysis results saved to: {save_path}")
    
    # Added: save per-sample predictions
    with open(pred_save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sample_predictions))
    logger.info(f"Per-sample predictions saved to: {pred_save_path}")
    
    return error_stats

# ====================== Main Function (adapted to RGCN, num_heads removed) ======================
def main():
    parser = argparse.ArgumentParser(description='Word relation classification: DeBERTa-v3-large + Sememe-RGCN + single classifier')
    # Data arguments
    parser.add_argument('--train_path', type=str, default='./data/train_data.jsonl', help='Path to the training data')
    parser.add_argument('--dev_path', type=str, default='./data/dev_data.jsonl', help='Path to the validation data')
    parser.add_argument('--test_path', type=str, default='./data/test_data.jsonl', help='Path to the test data')
    parser.add_argument('--source', type=str, default='CogALexV', help='Source filter condition (customizable)')
    # Model arguments (num_heads removed, replaced with rgcn_hidden_dim)
    parser.add_argument('--deberta_name', type=str, default="./deberta", help='Path/name of the DeBERTa-v3 model')
    parser.add_argument('--rgcn_hidden_dim', type=int, default=1024, help='Hidden dim of the R-GCN module')
    parser.add_argument('--expert_hidden_dim', type=int, default=None, help='Hidden dim of the single classifier')
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=8, help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--patience', type=int, default=100, help='Early-stopping patience')
    parser.add_argument('--output_dir', type=str, default='./trainer_output_deberta_rgcn', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    logger.info(f"Hyperparameter configuration: {args}")

    # Device configuration
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize tokenizer
    logger.info("Loading DeBERTa-v3 tokenizer...")
    tokenizer = DebertaV2Tokenizer.from_pretrained(
        args.deberta_name,
        use_fast=False,
        local_files_only=True
    )
    logger.info(f"DeBERTa-v3 vocabulary size: {tokenizer.vocab_size}, max sequence length: {tokenizer.model_max_length}")

    # Build vocabularies
    data_paths = [args.train_path, args.dev_path, args.test_path]
    sememe2id, sememe_id2name = build_sememe_vocab(data_paths)
    rel_vocab = build_rel_vocab(data_paths, args.source)
    edge_type_vocab = build_edge_type_vocab(data_paths)
    num_classes = len(rel_vocab)
    logger.info(f"Number of relation classes: {num_classes}")
    logger.info(f"Number of sememe edge types: {len(edge_type_vocab)}")

    # Initialize model (num_heads removed, rgcn_hidden_dim passed in)
    logger.info("Initializing DeBERTa-v3 + R-GCN model (single classifier)...")
    model = RelationClassifier(
        deberta_name=args.deberta_name,
        sememe2id=sememe2id,
        sememe_id2name=sememe_id2name,
        num_classes=num_classes,
        rgcn_hidden_dim=args.rgcn_hidden_dim,  # Pass the R-GCN hidden dim
        edge_type_vocab=edge_type_vocab,
        tokenizer=tokenizer,
        expert_hidden_dim=args.expert_hidden_dim
    ).to(device)
    
    
    # Parameter-count statistics (adapted to R-GCN)
    total_model_params = sum(p.numel() for p in model.parameters())
    trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_model_params_m = total_model_params / 1e6
    trainable_model_params_m = trainable_model_params / 1e6
    total_model_params_b = total_model_params / 1e9
    trainable_model_params_b = trainable_model_params / 1e9

    # Count parameters of the R-GCN component (replaces the original RGAT)
    rgcn_total_params = sum(p.numel() for p in model.sememe_rgcn.parameters())
    rgcn_trainable_params = sum(p.numel() for p in model.sememe_rgcn.parameters() if p.requires_grad)
    rgcn_total_params_m = rgcn_total_params / 1e6
    rgcn_trainable_params_m = rgcn_trainable_params / 1e6

    fnn_total_params = sum(p.numel() for p in model.classifier.parameters())
    fnn_trainable_params = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    fnn_total_params_m = fnn_total_params / 1e6
    fnn_trainable_params_m = fnn_trainable_params / 1e6

    sb_total_params = sum(p.numel() for p in model.deberta.parameters())
    sb_trainable_params = sum(p.numel() for p in model.deberta.parameters() if p.requires_grad)
    sb_total_params_m = sb_total_params / 1e6
    sb_trainable_params_m = sb_trainable_params / 1e6

    # Print parameter-count information
    logger.info("="*100)
    logger.info("Model parameter-count report")
    logger.info("="*100)
    logger.info(f"Total parameters of the entire model: {total_model_params:,} (≈ {total_model_params_m:.2f} M / {total_model_params_b:.4f} B)")
    logger.info(f"Trainable parameters of the entire model: {trainable_model_params:,} (≈ {trainable_model_params_m:.2f} M / {trainable_model_params_b:.4f} B)")
    logger.info(f"------------------------------")
    logger.info(f"Total parameters of the R-GCN component: {rgcn_total_params:,} (≈ {rgcn_total_params_m:.4f} M)")
    logger.info(f"Trainable parameters of the R-GCN component: {rgcn_trainable_params:,} (≈ {rgcn_trainable_params_m:.4f} M)")
    logger.info(f"Ratio of R-GCN parameters to the entire model: {(rgcn_total_params / total_model_params) * 100:.4f}%")
    logger.info("="*100)
    logger.info(f"Total parameters of the classifier (FNN): {fnn_total_params:,} (≈ {fnn_total_params_m:.4f} M)")
    logger.info(f"Trainable parameters of the classifier (FNN): {fnn_trainable_params:,} (≈ {fnn_trainable_params_m:.4f} M)")
    logger.info(f"Ratio of classifier parameters to the entire model: {(fnn_total_params / total_model_params) * 100:.4f}%")
    logger.info("="*100)
    logger.info(f"Total parameters of the DeBERTa backbone: {sb_total_params:,} (≈ {sb_total_params_m:.2f} M)")
    logger.info(f"Trainable parameters of the DeBERTa backbone: {sb_trainable_params:,} (≈ {sb_trainable_params_m:.2f} M)")
    logger.info(f"Ratio of DeBERTa parameters to the entire model: {(sb_total_params / total_model_params) * 100:.4f}%")
    logger.info("Start loading and preprocessing train/val/test data...")
    dataset_dict = load_and_preprocess_data(
        train_path=args.train_path,
        dev_path=args.dev_path,
        test_path=args.test_path,
        tokenizer=tokenizer,
        sememe2id=sememe2id,
        rel_vocab=rel_vocab,
        edge_type_vocab=edge_type_vocab,
        source=args.source
    )
    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=30,
        save_strategy="steps",
        save_steps=30,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=30,
        logging_first_step=True,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
        disable_tqdm=False,
        remove_unused_columns=False,
        label_smoothing_factor=0.1,
    )

    # Initialize Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        compute_metrics=compute_metrics,
        data_collator=custom_data_collator,
    )

    # Start training
    logger.info("=== Start training (DeBERTa-v3 + R-GCN + single classifier) ===")
    trainer.train()

    # Test-set evaluation
    logger.info("=== Test-set evaluation ===")
    test_results = trainer.evaluate(dataset_dict["test"])
    logger.info(f"Test-set basic metrics: {test_results}")
    
    # Get detailed test metrics
    detailed_test_metrics = get_detailed_test_metrics(model, dataset_dict["test"], rel_vocab, device)
    
    # Error analysis
    analyze_errors(model, dataset_dict["test"], tokenizer, rel_vocab, device)

    # Save the final model
    trainer.save_model(f"{args.output_dir}/final_model")
    logger.info(f"Final model saved to: {args.output_dir}/final_model")
    logger.info("Training finished!")

    # Print final statistics
    logger.info(f"\n===== Final statistics =====")
    logger.info(f"Number of sememe edge types: {len(edge_type_vocab)}")
    logger.info(f"Final sememe vocabulary size: {len(sememe2id)}")
    logger.info(f"Data source: {args.source}")

if __name__ == "__main__":
    main()