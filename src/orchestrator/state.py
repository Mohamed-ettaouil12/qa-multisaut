"""State definitions for the QA pipeline."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass, field


class Triplet(BaseModel):
    """A knowledge graph triplet."""
    h: str
    r: str
    t: str
    source_title: str = ""
    sentence: str = ""
    label: int = 0
    score: float = 0.0


class PipelineState(BaseModel):
    """State passed between pipeline nodes."""
    question: str
    qid: Optional[str] = None
    
    # Entity detection
    question_entities: List[str] = Field(default_factory=list)
    
    # Retrieval
    retrieved_triplets: List[Triplet] = Field(default_factory=list)
    retrieval_method: str = "none"
    retrieval_log: Dict[str, Any] = Field(default_factory=dict)
    
    # GNN scoring
    scored_triplets: List[Triplet] = Field(default_factory=list)
    gnn_log: Dict[str, Any] = Field(default_factory=dict)
    
    # Top-k selection
    topk_triplets: List[Triplet] = Field(default_factory=list)
    selection_log: Dict[str, Any] = Field(default_factory=dict)
    
    # Generation
    prompt: str = ""
    answer: str = ""
    sources: List[int] = Field(default_factory=list)
    generation_log: Dict[str, Any] = Field(default_factory=dict)
    
    # Abstention
    abstained: bool = False
    abstention_reason: str = ""
    
    # Metadata
    latency_ms: float = 0.0
    config: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


class PipelineConfig(BaseModel):
    """Configuration for the pipeline."""
    # GNN settings
    gnn_checkpoint: str = "checkpoints/gnn_best.pt"
    gnn_k: int = 5
    gnn_score_threshold: float = 0.1
    
    # Retrieval settings
    use_neo4j: bool = True
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_auth: tuple = ("neo4j", "REDACTED")
    max_hops: int = 2
    
    # LLM settings
    llm_path: str = "checkpoints/merged_final"
    llm_max_new_tokens: int = 64
    llm_use_4bit: bool = True
    
    # Abstention
    min_triplets_for_generation: int = 1
    max_score_threshold: float = 0.0
    
    # Embedding model
    embedder_name: str = "BAAI/bge-small-en-v1.5"
    
    # Device
    device: str = "cuda"


DEFAULT_CONFIG = PipelineConfig()