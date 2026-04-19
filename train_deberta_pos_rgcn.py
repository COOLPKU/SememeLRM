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

# ====================== 日志配置 ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ====================== 工具函数：构建词汇表（无修改，保持原有逻辑） ======================
def build_sememe_vocab(data_paths, graph_data_path='./data/graph_data.json'):
    """
    构建义原词汇表，补充数据集中未在graph_data中出现的义原
    """
    # 加载graph_data中的义原
    with open(graph_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sememe_type = data.get('sememe_type', [])
    if not sememe_type:
        raise ValueError("graph_data.json中未找到sememe_type字段")
    
    # 初始义原字典
    sememe2id = {sem: idx for idx, sem in enumerate(sememe_type)}
    sememe_id2name = {v: k for k, v in sememe2id.items()}
    initial_sememe_count = len(sememe2id)
    logger.info(f"初始Sememe词汇表大小: {initial_sememe_count}, 索引范围: [0, {initial_sememe_count-1}]")
    
    # 从数据集中收集所有义原
    dataset_semes = set()
    for data_path in data_paths:
        with open(data_path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                sample = json.loads(line.strip())
                
                # 收集word1义原
                word1_senses = sample["word1_senses"]
                word1_sememes = []
                for tmp_list in word1_senses["sememes"]:
                    word1_sememes+=tmp_list
                dataset_semes.update(word1_sememes)
                # 收集word2义原
                word2_senses = sample["word2_senses"]
                word2_sememes = []
                for tmp_list in word2_senses["sememes"]:
                    word2_sememes+=tmp_list
                dataset_semes.update(word2_sememes)
    
    # 补充未收录的义原
    new_semes = []
    for sem in dataset_semes:
        if sem not in sememe2id:
            sememe2id[sem] = len(sememe2id)
            sememe_id2name[len(sememe_id2name)] = sem
            new_semes.append(sem)
    
    # 打印补充信息
    logger.info(f"从数据集中补充的义原数量: {len(new_semes)}")
    if new_semes:
        logger.info(f"补充的义原列表: {new_semes}")
    logger.info(f"最终Sememe词汇表大小: {len(sememe2id)}, 索引范围: [0, {len(sememe2id)-1}]")
    
    return sememe2id, sememe_id2name

def build_rel_vocab(data_paths, source):
    """构建样本关系类型字典（新增source参数）"""
    rel_vocab = defaultdict(int)
    rel_id = 0
    for data_path in data_paths:
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                
                if sample.get('source') != source:  # 使用传入的source参数，不再硬编码ROOT09
                    continue
                if 'random' in sample['rel']:
                    continue
                rel = sample.get("rel", "")
                if rel and rel not in rel_vocab:
                    rel_vocab[rel] = rel_id
                    rel_id += 1
    logger.info(f"样本关系类型数量: {len(rel_vocab)}, 类型: {list(rel_vocab.keys())}")
    return rel_vocab

def build_edge_type_vocab(data_paths):
    """构建义原边类型（三元组第二个元素）字典"""
    edge_type_vocab = defaultdict(int)
    edge_type_id = 0
    for data_path in data_paths:
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                # 处理word1的三元组
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
    logger.info(f"义原边类型数量: {len(edge_type_vocab)}, 类型: {list(edge_type_vocab.keys())}")
    return edge_type_vocab

def build_sememe_graph(valid_semes, triples, sememe2id, edge_type_vocab):
    """构建义原图（包含边类型，无修改）"""
    if not valid_semes:
        # 空图：包含edge_type字段
        return Data(
            x=torch.tensor([[0]], dtype=torch.long),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_type=torch.empty((0,), dtype=torch.long)
        )
    
    x = torch.tensor([sememe2id[sem] for sem in valid_semes], dtype=torch.long).unsqueeze(1)
    
    edge_index = []
    edge_type = []
    for src_sem, rel, dst_sem in triples:  # 适配新三元组格式
        if src_sem not in valid_semes or dst_sem not in valid_semes:
            continue
        if rel not in edge_type_vocab:
            print('missing rel_type',rel)
            continue
        src_idx = valid_semes.index(src_sem)
        dst_idx = valid_semes.index(dst_sem)
        edge_index.append([src_idx, dst_idx])
        # 边类型转id
        edge_type.append(edge_type_vocab[rel])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long) if edge_type else torch.empty((0,), dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_type=edge_type)

# ====================== 数据预处理（无修改，保持原有逻辑） ======================
def preprocess_sample(sample, tokenizer, sememe2id, rel_vocab, edge_type_vocab, source):
    """单样本预处理（新增source参数）"""
    if sample.get('source') != source:  # 使用传入的source参数
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

    # 构建输入token序列
    input_tokens = [tokenizer.cls_token]
    input_tokens.extend(tokenizer.tokenize(f"Today, I finally discovered the relation between"))
    
    # 位置标记
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

    # 编码token ID
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

    # 过滤有效sememe
    valid_word1_semes = []
    for sem in word1_sememes:
        if sem not in valid_word1_semes and sem in sememe2id:
            valid_word1_semes.append(sem)
    
    valid_word2_semes = []
    for sem in word2_sememes:
        if sem not in valid_word2_semes and sem in sememe2id:
            valid_word2_semes.append(sem)

    # 构建sememe图（包含边类型）
    def serialize_graph(graph):
        return {
            'x': graph.x.numpy().tolist(),
            'edge_index': graph.edge_index.numpy().tolist(),
            'edge_type': graph.edge_type.numpy().tolist()
        }
    
    word1_g = build_sememe_graph(valid_word1_semes, word1_triples, sememe2id, edge_type_vocab)
    word2_g = build_sememe_graph(valid_word2_semes, word2_triples, sememe2id, edge_type_vocab)
    
    # main sememe索引
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
    """加载并预处理数据（新增source参数）"""
    def load_jsonl(path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    # 加载原始数据
    train_data = load_jsonl(train_path)
    dev_data = load_jsonl(dev_path)
    test_data = load_jsonl(test_path)
    
    # 过滤并预处理
    def filter_and_map(data, source,cog=False):  # 新增source参数
        processed = []
        for sample in tqdm(data):
            if sample['source'] != source:
                continue
            if 'random' in sample['rel'] and cog==True:continue
            res = preprocess_sample(sample, tokenizer, sememe2id, rel_vocab, edge_type_vocab, source)
            if res is not None:
                processed.append(res)
        return processed
    
    logger.info("预处理训练集...")
    train_processed = filter_and_map(train_data, source,True)
    logger.info("预处理验证集...")
    dev_processed = filter_and_map(dev_data, source,True)
    logger.info("预处理测试集...")
    test_processed = filter_and_map(test_data, source,True)
    
    # 转换为Dataset
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_processed),
        "validation": Dataset.from_list(dev_processed),
        "test": Dataset.from_list(test_processed)
    })
    
    # 动态padding
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
    
    # 设置格式并批量padding
    dataset_dict = dataset_dict.map(
        pad_function,
        batched=True,
        batch_size=None,
        desc="Padding sequences"
    )
    
    # 转换Tensor类型字段
    tensor_columns = ["input_ids", "attention_mask", "rel_label"]
    for split in dataset_dict:
        dataset_dict[split].set_format(
            type="torch",
            columns=tensor_columns,
            output_all_columns=True
        )
    
    logger.info(f"数据加载完成：训练集{len(dataset_dict['train'])}，验证集{len(dataset_dict['validation'])}，测试集{len(dataset_dict['test'])}")
    return dataset_dict

# ====================== 自定义数据Collator（无修改，保持原有逻辑） ======================
def custom_data_collator(features):
    """自定义数据collator（兼容边类型字段）"""
    batch = {}
    
    # 处理Tensor类型字段
    tensor_keys = ["input_ids", "attention_mask", "rel_label"]
    for key in tensor_keys:
        if key in features[0]:
            batch[key] = torch.stack([f[key] for f in features])
    
    # 处理列表/字典类型字段（包含边类型）
    non_tensor_keys = [
        "word1_g", "word2_g", "word1_main_indices", "word2_main_indices",
        "sememe_positions", "sememe_names", "word1_start", "word1_end",
        "word2_start", "word2_end"
    ]
    for key in non_tensor_keys:
        if key in features[0]:
            batch[key] = [f[key] for f in features]
    
    return batch

# ====================== 模型组件（核心修改：RGAT -> RGCN） ======================
class SememeRGCN(nn.Module):
    """义原关系图卷积网络（RGCN）- 替换原RGAT"""
    def __init__(self, num_semes, num_relations, embed_dim, rgcn_hidden_dim, sememe_id2name, tokenizer):
        super().__init__()
        self.sememe_id2name = sememe_id2name
        self.tokenizer = tokenizer
        self.num_relations = num_relations  # 边类型数量
        sememe_embed_dim = embed_dim
        
        # 义原嵌入层
        self.sememe_emb = nn.Embedding(num_semes, sememe_embed_dim)
        self.num_semes = num_semes
        self._emb_init_flag = False

        # 三层RGCN（Relational GCN）- 替换原RGAT
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

        # 激活/归一化/Dropout（调整层归一化维度，移除多头相关计算）
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.ln1 = nn.LayerNorm(rgcn_hidden_dim)  # 无多头拼接，直接用hidden_dim
        self.ln2 = nn.LayerNorm(rgcn_hidden_dim)

    def _init_sememe_embeddings(self, deberta_word_emb):
        """初始化义原嵌入（无修改）"""
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
        logger.info("Sememe embedding初始化完成（适配DeBERTa-v3）")

    def forward(self, batch, deberta_word_emb=None):
        """RGCN前向传播（无多头，逻辑简化）"""
        # 首次初始化
        if not self._emb_init_flag and deberta_word_emb is not None:
            self._init_sememe_embeddings(deberta_word_emb)
        
        # 空图处理
        if batch.x.numel() == 0:
            return torch.empty((0, self.sememe_emb.embedding_dim), device=batch.x.device)
        
        # 合法性检查
        sem_id = batch.x.squeeze()
        if (sem_id < 0).any() or (sem_id >= self.num_semes).any():
            invalid_ids = sem_id[(sem_id < 0) | (sem_id >= self.num_semes)]
            raise ValueError(f"无效Sememe ID: {invalid_ids.tolist()}, 词表大小: {self.num_semes}")

        # RGCN前向传播（无多头，维度直接传递）
        x = self.sememe_emb(sem_id)  # [num_nodes, embed_dim]
        
        # 第一层RGCN
        x = self.rgcn1(x, batch.edge_index, batch.edge_type)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # 第二层RGCN
        x = self.rgcn2(x, batch.edge_index, batch.edge_type)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # 第三层RGCN
        x = self.rgcn3(x, batch.edge_index, batch.edge_type)
        
        return x

class SememeAwareEmbedding(nn.Module):
    """义原感知嵌入（无修改）"""
    def __init__(self, deberta_dim, sememe_id2name):
        super().__init__()
        self.deberta_dim = deberta_dim
        self.sememe_id2name = sememe_id2name

    def forward(self, input_ids, deberta_word_emb, sememe_rgcn,  # 参数名改为sememe_rgcn
                word1_g_batch, word2_g_batch,
                word1_global_indices, word2_global_indices,
                word1_main_counts, word2_main_counts,
                sememe_positions, sememe_names):
        batch_size, seq_len = input_ids.shape
        deberta_emb = deberta_word_emb(input_ids)
        
        # 调用RGCN（替换原RGAT）
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
    """关系分类器（移除MOE，适配RGCN）"""
    def __init__(self, deberta_name, sememe2id, sememe_id2name, num_classes, rgcn_hidden_dim, 
                 edge_type_vocab, tokenizer, expert_hidden_dim=None):
        super().__init__()
        # 加载DeBERTa-v3模型
        self.deberta = DebertaV2Model.from_pretrained(deberta_name)
        self.config = self.deberta.config
        self.config.problem_type = "single_label_classification"
        
        self.deberta_dim = self.deberta.config.hidden_size
        self.max_position_embeddings = self.deberta.config.max_position_embeddings
        
        self.special_token_emb = nn.Embedding(5, self.deberta_dim)
        nn.init.xavier_uniform_(self.special_token_emb.weight)
        
        # 初始化RGCN（替换原RGAT，移除num_heads参数）
        self.sememe_rgcn = SememeRGCN(  # 类名和参数名修改
            num_semes=len(sememe2id),
            num_relations=len(edge_type_vocab),
            embed_dim=self.deberta_dim,
            rgcn_hidden_dim=rgcn_hidden_dim,
            sememe_id2name=sememe_id2name,
            tokenizer=tokenizer
        )

        # 初始化义原感知嵌入（适配RGCN）
        self.sememe_aware_emb = SememeAwareEmbedding(
            deberta_dim=self.deberta_dim,
            sememe_id2name=sememe_id2name
        )
        
        # 单个分类器（无修改）
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
        
        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss()

    def _deserialize_graph(self, graph_dict, device):
        """反序列化图数据（无修改）"""
        x = torch.tensor(graph_dict['x'], dtype=torch.long, device=device)
        edge_index = torch.tensor(graph_dict['edge_index'], dtype=torch.long, device=device)
        edge_type = torch.tensor(graph_dict['edge_type'], dtype=torch.long, device=device)
        return Data(x=x, edge_index=edge_index, edge_type=edge_type)

    def _get_graph_batch(self, graph_list, device):
        """批量处理图数据（无修改）"""
        data_list = [self._deserialize_graph(g, device) for g in graph_list]
        return Batch.from_data_list(data_list)

    def _get_global_indices(self, graph_list, main_indices_list):
        """计算全局索引（无修改）"""
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
        """前向传播（适配RGCN，无修改其他逻辑）"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if seq_len > self.max_position_embeddings:
            raise ValueError(f"序列长度{seq_len}超过模型最大限制{self.max_position_embeddings}")

        # 解析额外参数
        word1_g_list = kwargs.get('word1_g', [])
        word2_g_list = kwargs.get('word2_g', [])
        word1_main_indices_list = kwargs.get('word1_main_indices', [])
        word2_main_indices_list = kwargs.get('word2_main_indices', [])
        sememe_positions = kwargs.get('sememe_positions', [])
        sememe_names = kwargs.get('sememe_names', [])
        
        # 位置参数
        word1_starts = kwargs.get('word1_start', [0]*batch_size)
        word1_ends = kwargs.get('word1_end', [0]*batch_size)
        word2_starts = kwargs.get('word2_start', [0]*batch_size)
        word2_ends = kwargs.get('word2_end', [0]*batch_size)

        # 处理图数据
        word1_g_batch = self._get_graph_batch(word1_g_list, device)
        word2_g_batch = self._get_graph_batch(word2_g_list, device)
        
        # 全局索引
        word1_global_indices, word1_main_counts = self._get_global_indices(word1_g_list, word1_main_indices_list)
        word2_global_indices, word2_main_counts = self._get_global_indices(word2_g_list, word2_main_indices_list)

        # 生成整合嵌入（调用RGCN）
        combined_emb = self.sememe_aware_emb(
            input_ids=input_ids,
            deberta_word_emb=self.deberta.embeddings.word_embeddings,
            sememe_rgcn=self.sememe_rgcn,  # 传递RGCN实例
            word1_g_batch=word1_g_batch,
            word2_g_batch=word2_g_batch,
            word1_global_indices=word1_global_indices,
            word2_global_indices=word2_global_indices,
            word1_main_counts=word1_main_counts,
            word2_main_counts=word2_main_counts,
            sememe_positions=sememe_positions,
            sememe_names=sememe_names
        )

        # DeBERTa-v3编码
        outputs = self.deberta(
            attention_mask=attention_mask,
            inputs_embeds=combined_emb
        )
        last_hidden_state = outputs.last_hidden_state

        # 计算word1/word2均值
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
        
        # 单个分类器前向传播
        logits = self.classifier(concat_features)
        
        # 训练时返回loss，评估时返回logits
        if self.training and rel_label is not None:
            ce_loss = self.loss_fn(logits, rel_label)
            return {"loss": ce_loss, "logits": logits}
        else:
            return {"logits": logits}

# ====================== 自定义Trainer（无修改） ======================
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

# ====================== 评估和错误分析（无修改） ======================
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
    """获取测试集每个类别F1值和完整混淆矩阵"""
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
        for batch in tqdm(test_loader, desc="计算详细测试指标"):
            for key in ["input_ids", "attention_mask", "rel_label"]:
                if key in batch:
                    batch[key] = batch[key].to(device)
            
            outputs = model(**batch)
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch["rel_label"].cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # 计算每个类别F1和分类报告
    logger.info("\n" + "="*80)
    logger.info("详细分类报告（每个类别F1值）")
    logger.info("="*80)
    cls_report = classification_report(all_labels, all_preds, target_names=rel_names, zero_division=0)
    logger.info(f"\n{cls_report}")
    
    # 计算并打印混淆矩阵
    logger.info("\n" + "="*80)
    logger.info("完整混淆矩阵")
    logger.info("="*80)
    conf_mat = confusion_matrix(all_labels, all_preds)
    # 格式化打印混淆矩阵（带类别名称）
    logger.info(f"类别顺序: {rel_names}")
    logger.info(f"\n混淆矩阵:\n{conf_mat}")
    
    # 返回详细指标
    cls_report_dict = classification_report(all_labels, all_preds, target_names=rel_names, output_dict=True, zero_division=0)
    return {
        "classification_report": cls_report_dict,
        "confusion_matrix": conf_mat.tolist(),
        "rel_names": rel_names
    }




def analyze_errors(model, test_dataset, tokenizer, rel_vocab, device, save_path="error_analysis.json", pred_save_path="predictions.txt"):
    """
    分析模型在测试集上的错误，并保存每个样本的预测结果
    
    Args:
        model: 训练好的模型
        test_dataset: 测试数据集
        tokenizer: 分词器
        rel_vocab: 关系标签词典 (rel_name -> id)
        device: 计算设备 (cpu/cuda)
        save_path: 错误分析统计结果保存路径
        pred_save_path: 逐样本预测结果保存路径
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
    # 新增：保存每个样本的索引、预测标签、真实标签
    sample_predictions = []
    
    with torch.no_grad():
        # 新增：记录当前处理到的样本索引
        sample_idx = 0
        for batch in tqdm(test_loader, desc="分析错误样本"):
            for key in ["input_ids", "attention_mask", "rel_label"]:
                if key in batch:
                    batch[key] = batch[key].to(device)
            
            outputs = model(** batch)
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch["rel_label"].cpu().numpy()
            rel_names = [test_dataset[i]["rel_name"] for i in range(len(preds))]
            
            # 新增：记录当前batch中每个样本的预测结果（包含索引）
            for i in range(len(preds)):
                pred_rel = id2rel[preds[i]]  # 预测的关系名称
                true_rel = id2rel[labels[i]]  # 真实的关系名称
                sample_predictions.append(f"{sample_idx + i} {pred_rel} {true_rel}")
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_rel_names.extend(rel_names)
            # 更新样本索引
            sample_idx += len(preds)
    
    # 原有逻辑：统计错误信息
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
    
    # 打印日志
    logger.info("\n" + "="*80)
    logger.info("错误分析报告")
    logger.info("="*80)
    logger.info(f"总样本数: {error_stats['total_samples']}")
    logger.info(f"错误样本数: {error_stats['total_errors']}")
    logger.info(f"总体错误率: {error_stats['error_rate']:.2%}")
    logger.info(f"总体准确率: {1 - error_stats['error_rate']:.2%}")
    
    logger.info("\n各类别错误分布（按真实标签）:")
    for rel, count in error_stats["errors_by_true_label"].items():
        total = total_by_class[rel]
        error_rate = count / total if total > 0 else 0.0
        logger.info(f"  {rel}: {count}/{total} ({error_rate:.2%})")
    
    logger.info("\n各类别准确率:")
    for rel, metrics in sorted(error_stats["per_class_metrics"].items(), key=lambda x: x[1]["accuracy"]):
        logger.info(f"  {rel}: {metrics['accuracy']:.2%} (正确: {metrics['correct']}/{metrics['total']})")
    
    # 保存错误分析统计结果
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(error_stats, f, ensure_ascii=False, indent=2)
    logger.info(f"错误分析结果已保存至: {save_path}")
    
    # 新增：保存逐样本预测结果
    with open(pred_save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sample_predictions))
    logger.info(f"逐样本预测结果已保存至: {pred_save_path}")
    
    return error_stats

# ====================== 主函数（适配RGCN，移除num_heads参数） ======================
def main():
    parser = argparse.ArgumentParser(description='词关系分类：DeBERTa-v3-large + Sememe-RGCN + 单个分类器')
    # 数据参数
    parser.add_argument('--train_path', type=str, default='./data/train_data1231.jsonl', help='训练数据路径')
    parser.add_argument('--dev_path', type=str, default='./data/dev_data1231.jsonl', help='验证数据路径')
    parser.add_argument('--test_path', type=str, default='./data/test_data1231.jsonl', help='测试数据路径')
    parser.add_argument('--source', type=str, default='CogALexV', help='数据来源过滤条件（可自定义修改）')
    # 模型参数（移除num_heads，修改为rgcn_hidden_dim）
    parser.add_argument('--deberta_name', type=str, default="./debertaV3", help='DeBERTa-v3模型路径/名称')
    parser.add_argument('--rgcn_hidden_dim', type=int, default=1024, help='RGCN隐藏层维度')
    parser.add_argument('--expert_hidden_dim', type=int, default=None, help='单个分类器隐藏层维度')
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-5, help='初始学习率')
    parser.add_argument('--epochs', type=int, default=8, help='训练轮次')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup比例')
    parser.add_argument('--patience', type=int, default=100, help='早停patience')
    parser.add_argument('--output_dir', type=str, default='./trainer_output_deberta_rgcn', help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    logger.info(f"超参数配置: {args}")

    # 设备配置
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 初始化Tokenizer
    logger.info("加载DeBERTa-v3 Tokenizer...")
    tokenizer = DebertaV2Tokenizer.from_pretrained(
        args.deberta_name,
        use_fast=False,
        local_files_only=True
    )
    logger.info(f"DeBERTa-v3词表大小: {tokenizer.vocab_size}, 最大序列长度: {tokenizer.model_max_length}")

    # 构建词汇表
    data_paths = [args.train_path, args.dev_path, args.test_path]
    sememe2id, sememe_id2name = build_sememe_vocab(data_paths)
    rel_vocab = build_rel_vocab(data_paths, args.source)
    edge_type_vocab = build_edge_type_vocab(data_paths)
    num_classes = len(rel_vocab)
    logger.info(f"关系类别数: {num_classes}")
    logger.info(f"义原边类型数: {len(edge_type_vocab)}")

    # 初始化模型（移除num_heads参数，传入rgcn_hidden_dim）
    logger.info("初始化DeBERTa-v3 + RGCN模型（单个分类器）...")
    model = RelationClassifier(
        deberta_name=args.deberta_name,
        sememe2id=sememe2id,
        sememe_id2name=sememe_id2name,
        num_classes=num_classes,
        rgcn_hidden_dim=args.rgcn_hidden_dim,  # 传入RGCN隐藏层维度
        edge_type_vocab=edge_type_vocab,
        tokenizer=tokenizer,
        expert_hidden_dim=args.expert_hidden_dim
    ).to(device)
    
    
    # 参数量统计（适配RGCN）
    total_model_params = sum(p.numel() for p in model.parameters())
    trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_model_params_m = total_model_params / 1e6
    trainable_model_params_m = trainable_model_params / 1e6
    total_model_params_b = total_model_params / 1e9
    trainable_model_params_b = trainable_model_params / 1e9

    # 统计RGCN组件参数量（替换原RGAT）
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

    # 打印参数量信息
    logger.info("="*100)
    logger.info("模型参数量统计报告")
    logger.info("="*100)
    logger.info(f"整个模型总参数量: {total_model_params:,} (≈ {total_model_params_m:.2f} M / {total_model_params_b:.4f} B)")
    logger.info(f"整个模型可训练参数量: {trainable_model_params:,} (≈ {trainable_model_params_m:.2f} M / {trainable_model_params_b:.4f} B)")
    logger.info(f"------------------------------")
    logger.info(f"RGCN组件总参数量: {rgcn_total_params:,} (≈ {rgcn_total_params_m:.4f} M)")
    logger.info(f"RGCN组件可训练参数量: {rgcn_trainable_params:,} (≈ {rgcn_trainable_params_m:.4f} M)")
    logger.info(f"RGCN参数量占整个模型的比例: {(rgcn_total_params / total_model_params) * 100:.4f}%")
    logger.info("="*100)
    logger.info(f"fnn组件总参数量: {fnn_total_params:,} (≈ {fnn_total_params_m:.4f} M)")
    logger.info(f"fnn组件可训练参数量: {fnn_trainable_params:,} (≈ {fnn_trainable_params_m:.4f} M)")
    logger.info(f"fnn参数量占整个模型的比例: {(fnn_total_params / total_model_params) * 100:.4f}%")
    logger.info("="*100)
    logger.info(f"sb组件总参数量: {sb_total_params:,} (≈ {sb_total_params_m:.4f} M)")
    logger.info(f"sb组件可训练参数量: {sb_trainable_params:,} (≈ {sb_trainable_params_m:.4f} M)")
    logger.info(f"sb参数量占整个模型的比例: {(sb_total_params / total_model_params) * 100:.4f}%")
    logger.info("开始加载并预处理训练/验证/测试数据...")
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
    # 定义TrainingArguments
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

    # 初始化Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        compute_metrics=compute_metrics,
        data_collator=custom_data_collator,
    )

    # 开始训练
    logger.info("=== 开始训练（DeBERTa-v3 + RGCN + 单个分类器） ===")
    trainer.train()

    # 测试集评估
    logger.info("=== 测试集评估 ===")
    test_results = trainer.evaluate(dataset_dict["test"])
    logger.info(f"测试集基础指标: {test_results}")
    
    # 获取详细测试指标
    detailed_test_metrics = get_detailed_test_metrics(model, dataset_dict["test"], rel_vocab, device)
    
    # 错误分析
    analyze_errors(model, dataset_dict["test"], tokenizer, rel_vocab, device)

    # 保存最终模型
    trainer.save_model(f"{args.output_dir}/final_model")
    logger.info(f"最终模型已保存至: {args.output_dir}/final_model")
    logger.info("训练完成！")

    # 打印最终统计
    logger.info(f"\n===== 最终统计 =====")
    logger.info(f"义原边类型数量: {len(edge_type_vocab)}")
    logger.info(f"最终义原词汇表大小: {len(sememe2id)}")
    logger.info(f"数据来源: {args.source}")

if __name__ == "__main__":
    main()