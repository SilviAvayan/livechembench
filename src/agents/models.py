from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Paper Quality Evaluator models
# ---------------------------------------------------------------------------

class DocumentType(str, Enum):
    research_paper = "research_paper"
    review = "review"
    supplementary = "supplementary"
    dataset = "dataset"
    protocol = "protocol"
    other = "other"


class OcrQuality(str, Enum):
    good = "good"
    partial = "partial"
    poor = "poor"


class PaperQualityEvaluation(BaseModel):
    paper_id: str = Field(description="Identifier of the evaluated paper.")
    document_type: DocumentType = Field(
        description="Classification of the document type."
    )
    is_real_paper: bool = Field(
        description="True if this is a genuine research paper or review, not supplementary material or raw data."
    )
    ocr_quality: OcrQuality = Field(
        description="Quality of the OCR/text extraction: good, partial, or poor."
    )
    has_abstract: bool = Field(
        description="True if a meaningful abstract was extracted."
    )
    has_figures: bool = Field(
        description="True if figure paths were extracted from the document."
    )
    has_tables: bool = Field(
        description="True if tables were extracted from the document."
    )
    worth_pursuing: bool = Field(
        description="True if this paper is suitable for inclusion in the chemistry benchmark."
    )
    justification: str = Field(
        description="Natural language explanation of the worth_pursuing decision. Be specific and concise."
    )
    evaluated_at: str = Field(
        description="ISO 8601 timestamp of when the evaluation was performed."
    )


# ---------------------------------------------------------------------------
# Question Proposer models
# ---------------------------------------------------------------------------

class QuestionType(str, Enum):
    T1 = "T1"  # PubChem property query
    T2 = "T2"  # RDKit structural reasoning
    T3 = "T3"  # Segment-anchored contrastive / comparative


class AnswerType(str, Enum):
    float_ = "float"
    int_ = "int"
    string = "string"
    choice = "choice"


class CandidateQuestion(BaseModel):
    question_text: str = Field(description="The full self-contained question.")
    answer: str = Field(description="The expected correct answer.")
    answer_type: AnswerType = Field(description="The type of the answer.")
    answer_units: Optional[str] = Field(
        default=None, description="Units for numerical answers, e.g. 'Da', 'g/mol'."
    )
    tolerance: Optional[float] = Field(
        default=None, description="Acceptable numerical tolerance for float answers."
    )
    question_type: QuestionType = Field(description="T1, T2, or T3.")
    chemical_entities: List[str] = Field(
        description="Chemical names, SMILES strings, or PubChem CIDs mentioned in the question."
    )
    verification_recipe: str = Field(
        description="How to programmatically verify the answer (RDKit, PubChem API, or direct comparison)."
    )
    source_segment: str = Field(
        description="Which part of the paper this question came from: abstract, key_points, conclusion, or tables."
    )


class PaperQuestions(BaseModel):
    paper_id: str = Field(description="Identifier of the source paper.")
    questions: List[CandidateQuestion] = Field(
        description="List of candidate questions generated from this paper."
    )
    proposed_at: str = Field(description="ISO 8601 timestamp of generation.")


# ---------------------------------------------------------------------------
# Critic models
# ---------------------------------------------------------------------------

class CriticVerdict(str, Enum):
    pass_ = "PASS"
    fail = "FAIL"
    needs_repair = "NEEDS_REPAIR"


class CriticName(str, Enum):
    ill_defined = "ill_defined"
    missing_conditions = "missing_conditions"


class CriticResult(BaseModel):
    verdict: CriticVerdict = Field(
        description="PASS if the question is acceptable, FAIL if it should be dropped, NEEDS_REPAIR if it can be fixed."
    )
    reason: str = Field(
        description="Concise explanation of the verdict (1-3 sentences)."
    )
    missing_conditions: Optional[List[str]] = Field(
        default=None,
        description="For missing_conditions critic: structured list of what must be added to the question."
    )
    suggested_fix: Optional[str] = Field(
        default=None,
        description="If verdict is NEEDS_REPAIR, a specific suggestion for how to fix the question."
    )


class QuestionCritiqueRecord(BaseModel):
    paper_id: str = Field(description="Source paper identifier.")
    question_index: int = Field(description="Index of the question within the paper's question list.")
    question_text: str = Field(description="The original question text.")
    critic: CriticName = Field(description="Which critic produced this result.")
    result: CriticResult = Field(description="The critic's verdict and reasoning.")
    evaluated_at: str = Field(description="ISO 8601 timestamp.")


class PaperCritiqueReport(BaseModel):
    paper_id: str = Field(description="Source paper identifier.")
    critiques: List[QuestionCritiqueRecord] = Field(
        description="All critique records for all questions in this paper."
    )
    critiqued_at: str = Field(description="ISO 8601 timestamp.")


# ---------------------------------------------------------------------------
# Question Repairer models
# ---------------------------------------------------------------------------

class RepairOutcome(str, Enum):
    kept_original = "kept_original"   # all critics passed, no repair needed
    repaired = "repaired"             # was broken, successfully repaired
    dropped = "dropped"               # repair attempted but still failing


class RepairedQuestion(BaseModel):
    original: CandidateQuestion = Field(description="The original proposed question.")
    repaired: Optional[CandidateQuestion] = Field(
        default=None,
        description="The rewritten question (None if kept original or dropped)."
    )
    outcome: RepairOutcome = Field(description="What happened to this question.")
    repair_notes: Optional[str] = Field(
        default=None,
        description="Summary of what was changed or why it was dropped."
    )


class RepairedPaperQuestions(BaseModel):
    paper_id: str = Field(description="Source paper identifier.")
    questions: List[RepairedQuestion] = Field(
        description="All questions with their repair outcomes."
    )
    repaired_at: str = Field(description="ISO 8601 timestamp.")

    def surviving(self) -> List[CandidateQuestion]:
        """Return only questions suitable for the benchmark (kept or repaired)."""
        result = []
        for rq in self.questions:
            if rq.outcome == RepairOutcome.kept_original:
                result.append(rq.original)
            elif rq.outcome == RepairOutcome.repaired and rq.repaired is not None:
                result.append(rq.repaired)
        return result


# ---------------------------------------------------------------------------
# Novelty Selector models
# ---------------------------------------------------------------------------

class NoveltyVerdict(str, Enum):
    pass_ = "PASS"
    fail = "FAIL"


class NoveltyResult(BaseModel):
    verdict: NoveltyVerdict = Field(
        description="PASS if question requires paper-specific knowledge, FAIL if answerable from general chemistry knowledge."
    )
    reason: str = Field(description="Concise explanation of the verdict.")


class SelectedQuestion(BaseModel):
    question: CandidateQuestion = Field(description="The surviving question.")
    novelty_verdict: NoveltyVerdict = Field(description="Result of novelty check.")
    novelty_reason: str = Field(description="Why it passed or failed novelty.")


class SelectedPaperQuestions(BaseModel):
    paper_id: str = Field(description="Source paper identifier.")
    questions: List[SelectedQuestion] = Field(
        description="All surviving questions with their novelty verdicts."
    )
    selected_at: str = Field(description="ISO 8601 timestamp.")

    def benchmark_ready(self) -> List[CandidateQuestion]:
        """Return only questions that passed the novelty check."""
        return [sq.question for sq in self.questions if sq.novelty_verdict == NoveltyVerdict.pass_]


# ---------------------------------------------------------------------------
# Dataset Builder models
# ---------------------------------------------------------------------------

class VerifierRecipe(BaseModel):
    """Structured recipe describing how to programmatically verify the answer."""
    endpoint: Optional[str] = Field(default=None, description="PubChem PUG REST endpoint template, e.g. '/compound/cid/{cid}/property/{prop}/JSON'.")
    field: Optional[str] = Field(default=None, description="JSON field to extract from PubChem response, e.g. 'ExactMass'.")
    function: Optional[str] = Field(default=None, description="RDKit function call, e.g. 'Chem.rdMolDescriptors.CalcNumRotatableBonds'.")
    input: Optional[str] = Field(default=None, description="Input type for RDKit, e.g. 'smiles'.")
    description: Optional[str] = Field(default=None, description="Free-text description of verification procedure.")


class Verifier(BaseModel):
    type: str = Field(description="Verification method: 'pubchem_pugrest', 'rdkit', or 'hybrid'.")
    recipe: VerifierRecipe


class Filters(BaseModel):
    """Quality filter results from the critic pipeline."""
    ill_defined: bool = Field(default=False, description="True if question failed the ill-defined critic (Critic 1).")
    missing_conditions: List[str] = Field(default_factory=list, description="Conditions identified as missing by Critic 2.")
    guessable: bool = Field(default=False, description="True if question failed the novelty/guessability check (Critic 3).")


class Provenance(BaseModel):
    """Source metadata for traceability."""
    month: str = Field(description="Year-month of question generation, e.g. '2026-05'.")
    paper_source: str = Field(description="Source repository: 'pubmed', 'chemrxiv', 'pmc', etc.")
    conversion_tool: str = Field(default="paddle_vl", description="PDF-to-text tool used.")
    pubchem_query_log_hash: Optional[str] = Field(default=None, description="SHA-256 hex digest of the sorted PubChem CID list queried for this question.")


class BenchmarkQuestion(BaseModel):
    id: str = Field(description="Unique question ID, e.g. '2026-05_001'.")
    paper_id: str = Field(description="Source paper in 'source:TYPE:ID' format, e.g. 'source:chemrxiv:30467288'.")
    segment_id: str = Field(description="Paper segment from which the question was derived, e.g. 'abstract' or 'tables'.")
    cid: Optional[int] = Field(default=None, description="Primary PubChem Compound ID for the question's main chemical entity.")
    question: str = Field(description="The full self-contained question text.")
    answer: str
    answer_type: AnswerType
    tolerance: Optional[float] = None
    verifier: Verifier
    filters: Filters
    provenance: Provenance
    # Internal fields retained for answer_verifier.py and model_evaluator.py
    question_type: QuestionType = Field(description="Internal question category: T1, T2, or T3.")
    chemical_entities: List[str] = Field(description="Chemical names, SMILES strings, or PubChem CIDs mentioned in the question.")
    answer_units: Optional[str] = None


class BenchmarkStats(BaseModel):
    total: int
    by_type: dict = Field(description="Count per question type (T1/T2/T3).")
    by_paper: dict = Field(description="Count per paper_id.")
    by_answer_type: dict = Field(description="Count per answer type.")


class LiveChemBench(BaseModel):
    name: str = "LiveChemBench"
    version: str
    created_at: str
    stats: BenchmarkStats
    questions: List[BenchmarkQuestion]


# ---------------------------------------------------------------------------
# Answer Verifier models
# ---------------------------------------------------------------------------

class VerificationStatus(str, Enum):
    correct = "correct"
    wrong = "wrong"
    error = "error"      # could not run verification (missing dep, bad SMILES, etc.)
    skipped = "skipped"  # no verifier implemented for this question


class VerificationResult(BaseModel):
    question_id: str
    question_type: QuestionType
    expected_answer: str
    computed_answer: Optional[str] = None
    status: VerificationStatus
    error: Optional[str] = None


class VerificationReport(BaseModel):
    benchmark_version: str
    verified_at: str
    results: List[VerificationResult]
    summary: dict = Field(description="Counts per status (correct/wrong/error/skipped).")


# ---------------------------------------------------------------------------
# Model Evaluator models
# ---------------------------------------------------------------------------

class EvalResult(BaseModel):
    question_id: str
    paper_id: str
    question_type: QuestionType
    answer_type: AnswerType
    question_text: str
    expected_answer: str
    model_raw_response: str
    model_answer: str
    correct: bool


class EvalScores(BaseModel):
    overall: float
    by_type: dict = Field(description="Accuracy per question type (T1/T2/T3).")
    by_paper: dict = Field(description="Accuracy per paper_id.")
    by_answer_type: dict = Field(description="Accuracy per answer type.")
    n_correct: int
    n_total: int


class EvalReport(BaseModel):
    model: str
    benchmark_version: str
    evaluated_at: str
    scores: EvalScores
    results: List[EvalResult]
