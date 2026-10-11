"""Pipeline nodes for the QA multi-hop system."""
import time
import logging
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict

import torch
from langgraph.graph import StateGraph
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from src.orchestrator.state import PipelineState, PipelineConfig, Triplet
from src.gnn.model import TripleScorerGNN
from src.llm.load_llm import load_base_model, generate_answer
from src.llm.prompt import build_prompt_from_triplets

logger = logging.getLogger(__name__)


class NodeBase:
    """Base class for pipeline nodes with logging."""
    
    def __init__(self, name: str, config: PipelineConfig):
        self.name = name
        self.config = config
    
    def _ensure_state(self, state) -> PipelineState:
        """Convert dict to PipelineState if needed."""
        if isinstance(state, dict):
            return PipelineState(**state)
        return state
    
    def log(self, state, key: str, value: Any):
        """Log a value to the appropriate log dict."""
        state = self._ensure_state(state)
        log_attr = f"{self.name}_log"
        if hasattr(state, log_attr):
            getattr(state, log_attr)[key] = value


class EntityDetectionNode(NodeBase):
    """Detect entities in the question using triplet matching."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__("entity_detection", config)
        self.embedder = SentenceTransformer(config.embedder_name, device=config.device)
        self.qid_to_triplets = self._load_triplets()
    
    def _load_triplets(self) -> Dict[str, List[Dict]]:
        """Load and group triplets by qid."""
        from collections import defaultdict
        by_qid = defaultdict(list)
        with open("data/processed/kg_triplets.jsonl") as f:
            for line in f:
                t = json.loads(line)
                by_qid[t["qid"]].append(t)
        return by_qid
    
    def __call__(self, state) -> PipelineState:
        state = self._ensure_state(state)
        start = time.time()
        
        # Simple entity detection: find entities from kg_triplets that appear in question
        # In production, this could use NER or linking
        question_lower = state.question.lower()
        
        # Load all unique entities from triplets
        all_entities = set()
        for triplets in self.qid_to_triplets.values():
            for t in triplets:
                all_entities.add(t["h"])
                all_entities.add(t["t"])
        
        # Find entities mentioned in question (simple substring match)
        detected = []
        for ent in all_entities:
            ent_lower = ent.lower()
            if ent_lower in question_lower and len(ent_lower) > 2:
                detected.append(ent)
        
        # If no entities found via substring, use embedding similarity
        if not detected:
            detected = self._embedding_entity_match(state.question, list(all_entities))
        
        state.question_entities = detected[:20]  # Limit
        self.log(state, "entities_detected", len(detected))
        self.log(state, "entities", detected)
        self.log(state, "latency_ms", (time.time() - start) * 1000)
        
        return state
    
    def _embedding_entity_match(self, question: str, entities: List[str], top_k: int = 10) -> List[str]:
        """Match entities using embedding similarity."""
        with torch.no_grad():
            q_emb = self.embedder.encode(
                [f"Represent this sentence for searching relevant passages: {question}"],
                normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False
            )
            ent_embs = self.embedder.encode(
                entities, normalize_embeddings=True, convert_to_tensor=True, 
                show_progress_bar=False, batch_size=64
            )
            scores = (ent_embs @ q_emb.T).squeeze()
            if scores.ndim == 0:
                scores = scores.unsqueeze(0)
            top_indices = torch.topk(scores, k=min(top_k, len(entities))).indices.tolist()
            return [entities[i] for i in top_indices]


class Neo4jRetrievalNode(NodeBase):
    """Retrieve neighborhood from Neo4j."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__("neo4j_retrieval", config)
        self.driver = None
        self._connect()
    
    def _connect(self):
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri, auth=self.config.neo4j_auth
            )
            self.driver.verify_connectivity()
            logger.info("Neo4j connected")
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {e}")
            self.driver = None
    
    def __call__(self, state) -> PipelineState:
        state = self._ensure_state(state)
        start = time.time()
        
        if not self.driver or not state.question_entities:
            state.retrieved_triplets = []
            state.retrieval_method = "neo4j_failed"
            self.log(state, "error", "No Neo4j connection or no entities")
            return state
        
        try:
            retrieved = self._retrieve_neighborhood(state.question_entities)
            state.retrieved_triplets = retrieved
            state.retrieval_method = "neo4j"
            self.log(state, "triplets_retrieved", len(retrieved))
        except Exception as e:
            logger.error(f"Neo4j retrieval failed: {e}")
            state.retrieved_triplets = []
            state.retrieval_method = "neo4j_error"
            self.log(state, "error", str(e))
        
        self.log(state, "latency_ms", (time.time() - start) * 1000)
        return state
    
    def _retrieve_neighborhood(self, entities: List[str]) -> List[Triplet]:
        """Query Neo4j for 1-2 hop neighborhood."""
        query = """
        UNWIND $entities AS ent_name
        MATCH (e:Entity {name: ent_name})
        OPTIONAL MATCH (e)-[r1:REL]-(e2:Entity)
        OPTIONAL MATCH (e2)-[r2:REL]-(e3:Entity)
        RETURN e.name as h1, r1.type as r1, e2.name as t1, r2.type as r2, e3.name as t2,
               r1.source_title as st1, r1.sentence as s1, r1.label as l1,
               r2.source_title as st2, r2.sentence as s2, r2.label as l2
        """
        with self.driver.session() as session:
            result = session.run(query, entities=entities)
            
            triplets = []
            seen = set()
            for record in result:
                # 1-hop
                if record["r1"] and record["t1"]:
                    key = (record["h1"], record["r1"], record["t1"])
                    if key not in seen:
                        seen.add(key)
                        triplets.append(Triplet(
                            h=record["h1"], r=record["r1"], t=record["t1"],
                            source_title=record["st1"] or "",
                            sentence=record["s1"] or "",
                            label=record["l1"] or 0
                        ))
                # 2-hop
                if record["r2"] and record["t2"]:
                    key = (record["t1"], record["r2"], record["t2"])
                    if key not in seen:
                        seen.add(key)
                        triplets.append(Triplet(
                            h=record["t1"], r=record["r2"], t=record["t2"],
                            source_title=record["st2"] or "",
                            sentence=record["s2"] or "",
                            label=record["l2"] or 0
                        ))
            return triplets


class PyGFallbackRetrievalNode(NodeBase):
    """Fallback retrieval using pre-computed PyG data."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__("pyg_fallback", config)
        self.qid_to_triplets = self._load_triplets()
    
    def _load_triplets(self) -> Dict[str, List[Dict]]:
        from collections import defaultdict
        by_qid = defaultdict(list)
        with open("data/processed/kg_triplets.jsonl") as f:
            for line in f:
                t = json.loads(line)
                by_qid[t["qid"]].append(t)
        return by_qid
    
    def __call__(self, state) -> PipelineState:
        state = self._ensure_state(state)
        start = time.time()
        
        # This node is a fallback - in practice we'd need the qid
        # For inference without qid, we return empty and rely on GNN to score all triplets
        # For now, we'll just log and return empty
        state.retrieved_triplets = []
        state.retrieval_method = "pyg_fallback_skipped"
        self.log(state, "note", "Fallback not implemented for inference without qid")
        self.log(state, "latency_ms", (time.time() - start) * 1000)
        
        return state


class GNNScoringNode(NodeBase):
    """Score triplets using the trained GNN."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__("gnn_scoring", config)
        self.device = config.device
        self.model = None
        self.embedder = None
        self._load_model()
    
    def _load_model(self):
        """Load GNN model and embedder."""
        checkpoint = torch.load(self.config.gnn_checkpoint, map_location=self.device, weights_only=False)
        gnn_config = checkpoint["config"]
        
        # Recreate model (A8: GATv2 + BM25)
        from src.gnn.predict_topk import TripleScorerGNN_BM25
        self.model = TripleScorerGNN_BM25(in_dim=384, **gnn_config).to(self.device)
        self.model.edge_proj = torch.nn.Linear(770, gnn_config['hidden_dim']).to(self.device)
        self.model.edge_scorer = torch.nn.Sequential(
            torch.nn.Linear(386, gnn_config['hidden_dim']),
            torch.nn.ReLU(),
            torch.nn.Dropout(gnn_config['dropout']),
            torch.nn.Linear(gnn_config['hidden_dim'], 1),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        self.embedder = SentenceTransformer(self.config.embedder_name, device=self.device)
        
        logger.info(f"GNN loaded: {sum(p.numel() for p in self.model.parameters()):,} params")
    
    def __call__(self, state) -> PipelineState:
        state = self._ensure_state(state)
        start = time.time()
        
        if not state.retrieved_triplets:
            state.scored_triplets = []
            self.log(state, "error", "No triplets to score")
            return state
        
        try:
            # Build PyG Data for this question
            data = self._build_pyg_data(state.question, state.retrieved_triplets, state.qid or "inference")
            
            # Compute BM25 scores
            bm25_scores = self._compute_bm25(state.question, state.retrieved_triplets)
            data.bm25_score = torch.tensor(bm25_scores, dtype=torch.float)
            
            # Score with GNN
            with torch.no_grad():
                data = data.to(self.device)
                logits = self.model(data)
                scores = torch.sigmoid(logits).cpu().tolist()
            
            # Attach scores to triplets
            scored = []
            for i, t in enumerate(state.retrieved_triplets):
                t_dict = t.model_dump()
                t_dict["score"] = scores[i] if i < len(scores) else 0.0
                scored.append(Triplet(**t_dict))
            
            # Sort by score descending
            scored.sort(key=lambda x: x.score, reverse=True)
            state.scored_triplets = scored
            
            self.log(state, "triplets_scored", len(scored))
            self.log(state, "max_score", max(scores) if scores else 0)
            self.log(state, "min_score", min(scores) if scores else 0)
            
        except Exception as e:
            logger.error(f"GNN scoring failed: {e}")
            state.scored_triplets = state.retrieved_triplets
            self.log(state, "error", str(e))
        
        self.log(state, "latency_ms", (time.time() - start) * 1000)
        return state
    
    def _build_pyg_data(self, question: str, triplets: List[Triplet], qid: str = "inference"):
        """Build PyG Data object for a single question."""
        from torch_geometric.data import Data
        
        # Unique entities
        entities = set()
        for t in triplets:
            entities.add(t.h)
            entities.add(t.t)
        entity_list = sorted(entities)
        entity_to_idx = {e: i for i, e in enumerate(entity_list)}
        
        # Node features
        with torch.no_grad():
            x = self.embedder.encode(
                entity_list, normalize_embeddings=True, show_progress_bar=False, 
                convert_to_tensor=True, batch_size=64
            ).cpu()
        
        # Edges
        edge_index = []
        relation_texts = []
        triplet_texts = []
        for t in triplets:
            h_idx = entity_to_idx[t.h]
            t_idx = entity_to_idx[t.t]
            edge_index.append([h_idx, t_idx])
            relation_texts.append(t.r)
            triplet_texts.append(f"{t.h} {t.r} {t.t}")
        
        # Edge attributes (relation embeddings)
        with torch.no_grad():
            edge_attr = self.embedder.encode(
                relation_texts, normalize_embeddings=True, show_progress_bar=False,
                convert_to_tensor=True, batch_size=64
            ).cpu()
        
        # Question embedding
        with torch.no_grad():
            q_emb = self.embedder.encode(
                [f"Represent this sentence for searching relevant passages: {question}"],
                normalize_embeddings=True, show_progress_bar=False, convert_to_tensor=True
            ).cpu()
            q_emb_squeezed = q_emb.squeeze(0)
            
            triplet_embs = self.embedder.encode(
                [f"Represent this passage for retrieval: {text}" for text in triplet_texts],
                normalize_embeddings=True, show_progress_bar=False, convert_to_tensor=True, batch_size=64
            ).cpu()
            cos_qt = (triplet_embs @ q_emb_squeezed).squeeze()
            if cos_qt.ndim == 0:
                cos_qt = cos_qt.unsqueeze(0)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        y = torch.zeros(len(triplets), dtype=torch.float)  # dummy labels
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            cos_qt=cos_qt,
            q_emb=q_emb,
            qid=qid,
            num_nodes=len(entity_list),
        )
        data.entity_list = entity_list
        data.triplets = [t.model_dump() for t in triplets]
        
        return data
    
    def _compute_bm25(self, question: str, triplets: List[Triplet]) -> List[float]:
        """Compute BM25 scores for triplets."""
        def triplet_tokens(t):
            return f"{t.h} {t.r} {t.t}".lower().split()
        
        corpus = [triplet_tokens(t) for t in triplets]
        if not corpus:
            return []
        bm25 = BM25Okapi(corpus)
        query_tokens = question.lower().split()
        scores = bm25.get_scores(query_tokens)
        return scores.tolist() if hasattr(scores, 'tolist') else list(scores)


class TopKSelectionNode(NodeBase):
    """Select top-k triplets for generation."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__("topk_selection", config)
    
    def __call__(self, state) -> PipelineState:
        state = self._ensure_state(state)
        start = time.time()
        
        k = self.config.gnn_k
        score_thresh = self.config.gnn_score_threshold
        
        # Filter by score threshold
        filtered = [t for t in state.scored_triplets if t.score >= score_thresh]
        
        # Take top-k
        topk = filtered[:k]
        
        state.topk_triplets = topk
        self.log(state, "input_count", len(state.scored_triplets))
        self.log(state, "after_threshold", len(filtered))
        self.log(state, "selected", len(topk))
        self.log(state, "scores", [t.score for t in topk])
        self.log(state, "latency_ms", (time.time() - start) * 1000)
        
        return state


class AbstentionNode(NodeBase):
    """Check if we should abstain from answering."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__("abstention", config)
    
    def __call__(self, state) -> PipelineState:
        state = self._ensure_state(state)
        start = time.time()
        
        reasons = []
        
        # Check minimum triplets
        if len(state.topk_triplets) < self.config.min_triplets_for_generation:
            reasons.append(f"Only {len(state.topk_triplets)} triplets (min {self.config.min_triplets_for_generation})")
        
        # Check max score
        if state.topk_triplets:
            max_score = max(t.score for t in state.topk_triplets)
            if max_score < self.config.max_score_threshold:
                reasons.append(f"Max score {max_score:.3f} below threshold {self.config.max_score_threshold}")
        else:
            reasons.append("No triplets available")
        
        if reasons:
            state.abstained = True
            state.abstention_reason = "; ".join(reasons)
            state.answer = "Information insuffisante"
            state.sources = []
            logger.info(f"Abstention: {state.abstention_reason}")
        else:
            state.abstained = False
        
        self.log(state, "abstained", state.abstained)
        self.log(state, "reasons", reasons)
        self.log(state, "latency_ms", (time.time() - start) * 1000)
        
        return state


class GenerationNode(NodeBase):
    """Generate answer using the fine-tuned LLM."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__("generation", config)
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the fused model."""
        self.model, self.tokenizer = load_base_model(
            self.config.device, use_4bit=self.config.llm_use_4bit
        )
        # Load the fused model weights (they're already merged in merged_final)
        logger.info(f"LLM loaded from {self.config.llm_path}")
    
    def __call__(self, state) -> PipelineState:
        state = self._ensure_state(state)
        start = time.time()
        
        if state.abstained:
            self.log(state, "skipped", True)
            return state
        
        try:
            # Build prompt
            triplet_dicts = [t.model_dump() for t in state.topk_triplets]
            prompt = build_prompt_from_triplets(state.question, triplet_dicts)
            state.prompt = prompt
            
            # Generate
            gen_start = time.time()
            response = generate_answer(
                self.model, self.tokenizer, prompt, 
                max_new_tokens=self.config.llm_max_new_tokens
            )
            gen_latency = (time.time() - gen_start) * 1000
            
            # Parse sources
            answer, sources = self._parse_response(response)
            
            state.answer = answer
            state.sources = sources
            state.generation_log["raw_response"] = response
            self.log(state, "generation_latency_ms", gen_latency)
            self.log(state, "total_latency_ms", (time.time() - start) * 1000)
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            state.answer = "Erreur lors de la génération"
            state.sources = []
            state.errors.append(f"Generation: {str(e)}")
        
        return state
    
    def _parse_response(self, response: str) -> tuple[str, List[int]]:
        """Parse answer and sources from response."""
        if "Sources:" in response:
            parts = response.split("Sources:")
            answer = parts[0].strip()
            sources_str = parts[1].strip()
            sources = [int(x) for x in re.findall(r"\[(\d+)\]", sources_str)]
        else:
            answer = response.strip()
            sources = []
        return answer, sources


import json