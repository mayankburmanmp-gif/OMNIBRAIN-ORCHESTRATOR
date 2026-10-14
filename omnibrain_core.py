"""
OmniBrain core logic.

This file contains the main agents used for:
- routing questions
- document retrieval
- simple numeric analysis
- PDF image checking
- answer generation and validation
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class AgentState:
    query: str
    documents: List[Dict[str, Any]] = field(default_factory=list)
    selected_document_ids: List[str] = field(default_factory=list)
    vector_results: List[Dict[str, Any]] = field(default_factory=list)
    sql_results: List[Dict[str, Any]] = field(default_factory=list)
    vision_results: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    guardrails: Dict[str, Any] = field(default_factory=dict)


class EmbeddingService:
    """Create embeddings, with a simple word-count fallback."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception:
            print("Sentence-transformers not available. Using basic text matching.")

    @property
    def semantic_available(self) -> bool:
        return self.model is not None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)

        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        if self.model is not None:
            vectors = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(vectors, dtype=np.float32)

        # Basic fallback so the project can still run without the ML package.
        vocabulary = {}
        token_lists = []

        for text in texts:
            tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
            token_lists.append(tokens)

            for token in tokens:
                if token not in vocabulary:
                    vocabulary[token] = len(vocabulary)

        matrix = np.zeros(
            (len(texts), len(vocabulary)),
            dtype=np.float32,
        )

        for row_index, tokens in enumerate(token_lists):
            for token in tokens:
                matrix[row_index, vocabulary[token]] += 1

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-8)


class SupervisorAgent:
    """Choose which agents should handle a user question."""

    def route(self, query: str) -> Dict[str, Any]:
        query = query.lower()
        routes = []

        vision_words = [
            "chart", "graph", "table", "figure", "image",
            "diagram", "visual", "plot", "bar", "line graph", "pie chart"
        ]

        sql_words = [
            "trend", "historical", "over time", "quarter", "year",
            "month", "price", "revenue", "sales", "growth", "compare"
        ]

        retrieval_words = [
            "what", "why", "how", "explain", "summarize",
            "describe", "according", "report", "policy",
            "financial", "find"
        ]

        if any(word in query for word in vision_words):
            routes.append("vision")

        if any(word in query for word in sql_words):
            routes.extend(["retrieval", "sql"])

        if not routes or any(word in query for word in retrieval_words):
            if "retrieval" not in routes:
                routes.insert(0, "retrieval")

        # Remove duplicates without changing the order.
        routes = list(dict.fromkeys(routes))

        return {
            "routes": routes,
            "reason": "Selected agents using simple query keywords.",
        }


class VectorSearchAgent:
    """Find the most relevant document chunks."""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    def search(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 6,
    ) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        texts = [chunk.get("text", "") for chunk in chunks]
        matrix = self.embedding_service.encode(texts)
        query_vector = self.embedding_service.encode([query])[0]

        if matrix.size == 0 or query_vector.size == 0:
            return []

        scores = matrix @ query_vector
        indexes = np.argsort(-scores)[:top_k]

        results = []

        for index in indexes:
            item = dict(chunks[int(index)])
            score = float(scores[int(index)])

            if self.embedding_service.semantic_available:
                similarity = max(0.0, min(1.0, score))
            else:
                similarity = max(0.0, min(1.0, (score + 1) / 2))

            item["similarity"] = similarity
            results.append(item)

        return results


class SQLQueryAgent:
    """Perform a few safe calculations on structured data."""

    def run(
        self,
        query: str,
        rows: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        rows = rows or []
        query = query.lower()

        if not rows:
            return [{
                "status": "no_structured_data",
                "message": "No structured numeric data is available.",
            }]

        numeric_fields = set()

        for row in rows:
            for key, value in row.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_fields.add(key)

        result = {
            "status": "ok",
            "numeric_fields": sorted(numeric_fields),
        }

        if "average" in query or "mean" in query:
            averages = {}

            for key in numeric_fields:
                values = [
                    float(row[key])
                    for row in rows
                    if isinstance(row.get(key), (int, float))
                ]

                if values:
                    averages[key] = sum(values) / len(values)

            result["averages"] = averages

        if "total" in query or "sum" in query:
            totals = {}

            for key in numeric_fields:
                values = [
                    float(row[key])
                    for row in rows
                    if isinstance(row.get(key), (int, float))
                ]

                if values:
                    totals[key] = sum(values)

            result["totals"] = totals

        if "maximum" in query or "highest" in query or "max" in query:
            maximums = {}

            for key in numeric_fields:
                values = [
                    float(row[key])
                    for row in rows
                    if isinstance(row.get(key), (int, float))
                ]

                if values:
                    maximums[key] = max(values)

            result["maximums"] = maximums

        return [result]


class VisionLanguageAgent:
    """Check whether PDF images are available for visual analysis."""

    def analyze(
        self,
        query: str,
        images: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not images:
            return [{
                "status": "no_images",
                "message": "No PDF images are available.",
            }]

        results = []

        for image in images[:8]:
            results.append({
                "status": "image_available",
                "document_id": image.get("document_id"),
                "page": image.get("page"),
                "image_path": image.get("image_path"),
                "message": (
                    "PDF image found. A vision model is needed for "
                    "numerical chart or table interpretation."
                ),
            })

        return results


class SelfCorrectionLoop:
    """Retry retrieval once when the first search is weak."""

    def __init__(self, retriever: VectorSearchAgent, max_retries: int = 1):
        self.retriever = retriever
        self.max_retries = max_retries

    def retrieve_with_retry(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 6,
        min_score: float = 0.28,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        trace = []
        current_query = query
        results = []

        for attempt in range(self.max_retries + 1):
            results = self.retriever.search(current_query, chunks, top_k)
            best_score = results[0]["similarity"] if results else 0.0

            trace.append({
                "step": f"retrieval_attempt_{attempt + 1}",
                "action": "searched document chunks",
                "query": current_query,
                "best_similarity": round(best_score, 3),
            })

            if results and best_score >= min_score:
                return results, trace

            if attempt < self.max_retries:
                current_query = query + " relevant evidence from the document"
                trace.append({
                    "step": "query_rewrite",
                    "action": "rewrote query after weak retrieval",
                })

        return results, trace


class GuardrailsEvaluator:
    """Basic check for evidence and empty answers."""

    def evaluate(
        self,
        query: str,
        answer: str,
        citations: List[Dict[str, Any]],
        min_citations: int = 1,
    ) -> Dict[str, Any]:
        has_evidence = len(citations) >= min_citations
        empty_answer = not answer.strip()
        grounded = has_evidence and not empty_answer

        citation_coverage = min(
            1.0,
            len(citations) / max(min_citations, 1),
        )

        return {
            "is_grounded": grounded,
            "outside_scope": empty_answer,
            "citation_count": len(citations),
            "citation_coverage": round(citation_coverage, 3),
            "hallucination_risk": round(1.0 - citation_coverage, 3),
            "message": (
                "Answer has document evidence."
                if grounded
                else "More evidence is needed before treating this answer as grounded."
            ),
        }


class SynthesisAgent:
    """Build the final answer from retrieved evidence."""

    def synthesize(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        sql_results: List[Dict[str, Any]],
        vision_results: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        citations = []
        evidence_lines = []

        for item in vector_results[:5]:
            text = item.get("text", "").strip()

            if not text:
                continue

            citation = {
                "document_id": item.get("document_id"),
                "document_name": item.get("document_name", "document.pdf"),
                "page": item.get("page"),
                "chunk_id": item.get("chunk_id"),
                "similarity": round(float(item.get("similarity", 0)), 3),
                "content": text[:600],
            }

            citations.append(citation)
            evidence_lines.append(
                f"- {text[:700]} "
                f"[{citation['document_name']}, page {citation['page']}]"
            )

        if not evidence_lines:
            return (
                "I could not find sufficiently relevant evidence in the uploaded documents.",
                [],
                0.0,
            )

        answer = "Based on the retrieved document evidence:\n\n"
        answer += "\n".join(evidence_lines)

        if sql_results and sql_results[0].get("status") == "ok":
            answer += "\n\nStructured analysis:\n"
            answer += str(sql_results[0])

        image_found = any(
            item.get("status") == "image_available"
            for item in vision_results
        )

        if image_found:
            answer += (
                "\n\nVisual evidence: relevant PDF images were found. "
                "No chart values were interpreted without a vision model."
            )

        similarities = [item["similarity"] for item in citations]
        average_similarity = float(np.mean(similarities)) if similarities else 0.0

        confidence = min(
            1.0,
            0.70 * average_similarity
            + 0.30 * min(1.0, len(citations) / 3),
        )

        return answer, citations, confidence


class OmniBrainOrchestrator:
    """Run the OmniBrain agents in a simple sequence."""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.supervisor = SupervisorAgent()
        self.retriever = VectorSearchAgent(self.embedding_service)
        self.sql_agent = SQLQueryAgent()
        self.vision_agent = VisionLanguageAgent()
        self.self_rag = SelfCorrectionLoop(self.retriever)
        self.synthesizer = SynthesisAgent()
        self.guardrails = GuardrailsEvaluator()

    def process_query(
        self,
        query: str,
        documents: Optional[List[Dict[str, Any]]] = None,
        selected_document_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        documents = documents or []
        selected_document_ids = selected_document_ids or []

        if selected_document_ids:
            documents = [
                doc for doc in documents
                if doc.get("document_id") in selected_document_ids
            ]

        chunks = []
        images = []

        for document in documents:
            for chunk in document.get("chunks", []):
                item = dict(chunk)
                item.setdefault("document_id", document.get("document_id"))
                item.setdefault("document_name", document.get("filename"))
                chunks.append(item)

            for image in document.get("images", []):
                item = dict(image)
                item.setdefault("document_id", document.get("document_id"))
                images.append(item)

        state = AgentState(
            query=query,
            documents=documents,
            selected_document_ids=selected_document_ids,
        )

        route = self.supervisor.route(query)

        state.execution_trace.append({
            "step": "supervisor",
            "action": "selected agents",
            "routes": route["routes"],
        })

        if "retrieval" in route["routes"]:
            state.vector_results, trace = self.self_rag.retrieve_with_retry(
                query,
                chunks,
            )
            state.execution_trace.extend(trace)

        if "sql" in route["routes"]:
            # The current PDF flow does not create structured rows.
            state.sql_results = self.sql_agent.run(query)
            state.execution_trace.append({
                "step": "sql_agent",
                "action": "checked for structured data",
            })

        if "vision" in route["routes"]:
            state.vision_results = self.vision_agent.analyze(query, images)
            state.execution_trace.append({
                "step": "vision_agent",
                "action": "checked PDF image evidence",
                "images_checked": len(images),
            })

        (
            state.final_answer,
            state.citations,
            state.confidence,
        ) = self.synthesizer.synthesize(
            query,
            state.vector_results,
            state.sql_results,
            state.vision_results,
        )

        state.guardrails = self.guardrails.evaluate(
            query,
            state.final_answer,
            state.citations,
        )

        state.execution_trace.append({
            "step": "validation",
            "action": "checked grounding and citations",
            "grounded": state.guardrails["is_grounded"],
        })

        processing_time = time.perf_counter() - start_time

        return {
            "answer": state.final_answer,
            "citations": state.citations,
            "confidence": round(state.confidence, 3),
            "execution_trace": state.execution_trace,
            "guardrails": state.guardrails,
            "processing_time": round(processing_time, 3),
            "routes": route["routes"],
            "semantic_embeddings": self.embedding_service.semantic_available,
        }
