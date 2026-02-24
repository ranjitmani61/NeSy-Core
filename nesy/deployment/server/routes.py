"""
nesy/deployment/server/routes.py
==================================
REST API routes for NeSy-Core inference server.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# ─── Request/Response Models ────────────────────────────────────

class ReasonRequest(BaseModel):
    facts: List[Dict[str, Any]]         # [{"name": "HasSymptom", "args": ["p1", "fever"]}]
    context_type: str = "general"
    neural_confidence: float = 0.90
    raw_input: Optional[str] = None

class LearnRequest(BaseModel):
    rule_id: str
    antecedents: List[List[str]]        # [["HasSymptom", "?p", "fever"], ...]
    consequents: List[List[str]]
    weight: float = 1.0
    make_anchor: bool = False
    description: str = ""

class ReasonResponse(BaseModel):
    answer:          str
    status:          str
    confidence:      Dict[str, float]
    flags:           List[str]
    reasoning_steps: int
    critical_nulls:  int
    trustworthy:     bool

# ─── Global model instance ──────────────────────────────────────
# In production: inject via dependency injection
_model = None

def get_model():
    global _model
    if _model is None:
        from nesy.api.nesy_model import NeSyModel
        _model = NeSyModel()
    return _model

# ─── Routes ─────────────────────────────────────────────────────

@router.post("/reason", response_model=ReasonResponse)
async def reason(request: ReasonRequest):
    from nesy.core.types import Predicate
    model = get_model()
    facts = {
        Predicate(name=f["name"], args=tuple(f.get("args", [])))
        for f in request.facts
    }
    output = model.reason(
        facts=facts,
        context_type=request.context_type,
        neural_confidence=request.neural_confidence,
        raw_input=request.raw_input,
    )
    return ReasonResponse(
        answer=output.answer,
        status=output.status.value,
        confidence={
            "factual":            output.confidence.factual,
            "reasoning":          output.confidence.reasoning,
            "knowledge_boundary": output.confidence.knowledge_boundary,
        },
        flags=output.flags,
        reasoning_steps=len(output.reasoning_trace.steps),
        critical_nulls=len(output.null_set.critical_items),
        trustworthy=output.is_trustworthy(),
    )

@router.post("/learn")
async def learn(request: LearnRequest):
    from nesy.core.types import Predicate, SymbolicRule
    model = get_model()
    rule = SymbolicRule(
        id=request.rule_id,
        antecedents=[Predicate(a[0], tuple(a[1:])) for a in request.antecedents],
        consequents=[Predicate(c[0], tuple(c[1:])) for c in request.consequents],
        weight=request.weight,
        description=request.description,
    )
    model.learn(rule, make_anchor=request.make_anchor)
    return {"status": "learned", "rule_id": request.rule_id, "total_rules": model.rule_count}

@router.get("/rules")
async def list_rules():
    model = get_model()
    return {
        "total": model.rule_count,
        "anchored": model.anchored_rules,
        "graph": model.concept_graph_stats,
    }
