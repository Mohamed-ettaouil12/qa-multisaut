"""LangGraph pipeline for QA multi-hop."""
from langgraph.graph import StateGraph, END
from src.orchestrator.state import PipelineState, PipelineConfig, DEFAULT_CONFIG
from src.orchestrator.nodes import (
    EntityDetectionNode,
    Neo4jRetrievalNode,
    PyGFallbackRetrievalNode,
    GNNScoringNode,
    TopKSelectionNode,
    AbstentionNode,
    GenerationNode,
)


def create_pipeline(config: PipelineConfig = DEFAULT_CONFIG) -> StateGraph:
    """Create the QA pipeline graph."""
    
    # Initialize nodes
    entity_node = EntityDetectionNode(config)
    neo4j_node = Neo4jRetrievalNode(config)
    pyg_node = PyGFallbackRetrievalNode(config)
    gnn_node = GNNScoringNode(config)
    topk_node = TopKSelectionNode(config)
    abstain_node = AbstentionNode(config)
    gen_node = GenerationNode(config)
    
    # Build graph
    workflow = StateGraph(PipelineState)
    
    # Add nodes
    workflow.add_node("detect_entities", entity_node)
    workflow.add_node("retrieve_neo4j", neo4j_node)
    workflow.add_node("retrieve_pyg", pyg_node)
    workflow.add_node("score_gnn", gnn_node)
    workflow.add_node("select_topk", topk_node)
    workflow.add_node("check_abstain", abstain_node)
    workflow.add_node("generate", gen_node)
    
    # Define edges
    workflow.set_entry_point("detect_entities")
    
    # Entity detection -> Neo4j retrieval
    workflow.add_edge("detect_entities", "retrieve_neo4j")
    
    # Neo4j retrieval -> PyG fallback (if Neo4j failed)
    def should_use_pyg(state: PipelineState) -> str:
        if state.retrieval_method in ("neo4j_failed", "neo4j_error", "neo4j"):
            if state.retrieved_triplets:
                return "score_gnn"
            return "retrieve_pyg"
        return "score_gnn"
    
    workflow.add_conditional_edges(
        "retrieve_neo4j",
        should_use_pyg,
        {
            "score_gnn": "score_gnn",
            "retrieve_pyg": "retrieve_pyg",
        }
    )
    
    # PyG fallback -> GNN scoring
    workflow.add_edge("retrieve_pyg", "score_gnn")
    
    # GNN scoring -> TopK selection
    workflow.add_edge("score_gnn", "select_topk")
    
    # TopK -> Abstention check
    workflow.add_edge("select_topk", "check_abstain")
    
    # Abstention -> Generation or END
    def should_generate(state: PipelineState) -> str:
        return "generate" if not state.abstained else END
    
    workflow.add_conditional_edges(
        "check_abstain",
        should_generate,
        {
            "generate": "generate",
            END: END,
        }
    )
    
    # Generation -> END
    workflow.add_edge("generate", END)
    
    return workflow.compile()


# Convenience function
def run_pipeline(question: str, config: PipelineConfig = None) -> PipelineState:
    """Run the pipeline on a single question."""
    if config is None:
        config = DEFAULT_CONFIG
    
    pipeline = create_pipeline(config)
    
    initial_state = PipelineState(
        question=question,
        config=config.model_dump()
    )
    
    result = pipeline.invoke(initial_state)
    # LangGraph returns dict, convert back to PipelineState
    if isinstance(result, dict):
        result = PipelineState(**result)
    return result


if __name__ == "__main__":
    # Quick test
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # This would require GPU and models loaded
    print("Pipeline graph created successfully")