"""RAG context retrieval for LLM student simulation.

Retrieves similar submissions to provide additional context:
Retrieves the target student's own prior work on similar problems.

Embeddings are lazy-loaded from HuggingFace. Falls back to pass-rate
filtering when embeddings are unavailable.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAX_CODE_LINES = 40


def _truncate_code(code: str, max_lines: int = MAX_CODE_LINES) -> str:
    lines = code.split("\n")
    if len(lines) <= max_lines:
        return code
    return "\n".join(lines[:max_lines]) + "\n// ... (truncated)"


def _pass_rate(pass_str: str) -> float:
    if not pass_str or not isinstance(pass_str, str):
        return 0.0
    clean = pass_str.replace(".", "").strip()
    if not clean:
        return 0.0
    return clean.count("1") / len(clean)


@dataclass
class SelfTrajectory:
    """Test student's own attempt history on a similar question."""
    question_name: str
    similarity_score: float
    attempts: List[Dict]  # [{response, response_type, pass_pattern}, ...]


class RAGRetriever:
    def __init__(self, main_df: pd.DataFrame, question_infos: pd.DataFrame, recency_weight: float = 0.5):
        logger.info("RAGRetriever: building indices...")
        subs = main_df[main_df["response_type"].isin(["Submit", "Prechecked"])].copy()
        subs = subs.dropna(subset=["response"])
        subs["pass"] = subs["pass"].astype(str).fillna("")
        subs["pass_rate"] = subs["pass"].apply(_pass_rate)
        subs["timestamp_dt"] = pd.to_datetime(
            subs["timestamp"], format="%d/%m/%y, %H:%M:%S", errors="coerce"
        )

        self._question_subs: Dict[str, pd.DataFrame] = {}
        for qid, group in subs.groupby("question_unittest_id"):
            self._question_subs[str(int(qid) if isinstance(qid, float) else qid)] = group

        self._student_subs: Dict[str, pd.DataFrame] = {}
        for sid, group in subs.groupby("student_id"):
            self._student_subs[str(sid)] = group

        student_pass_rates = subs.groupby("student_id")["pass_rate"].mean()
        self._student_pass_rates: Dict[str, float] = {
            str(k): float(v) for k, v in student_pass_rates.items()
        }

        self._recency_weight = recency_weight

        # Build TF-IDF similarity matrix over question text
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        q_dedup = question_infos.drop_duplicates(subset=["question_id"])
        q_texts = {}
        for _, row in q_dedup.iterrows():
            qid = str(int(row["question_id"]) if isinstance(row["question_id"], float) else row["question_id"])
            text = str(row.get("question_text", "")) + " " + str(row.get("question_name", ""))
            if text.strip():
                q_texts[qid] = text

        self._q_ids = list(q_texts.keys())
        self._q_id_to_idx = {qid: i for i, qid in enumerate(self._q_ids)}

        self._q_id_to_name = {}
        self._q_id_to_week = {}
        for _, row in q_dedup.iterrows():
            qid = str(int(row["question_id"]) if isinstance(row["question_id"], float) else row["question_id"])
            self._q_id_to_name[qid] = str(row.get("question_name", qid))
            if pd.notna(row.get("week")):
                self._q_id_to_week[qid] = int(row["week"])

        if self._q_ids:
            vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            tfidf_matrix = vectorizer.fit_transform([q_texts[qid] for qid in self._q_ids])
            self._sim_matrix = cosine_similarity(tfidf_matrix)
        else:
            self._sim_matrix = None

        logger.info(
            "RAGRetriever: indexed %d questions, %d students, %d question embeddings.",
            len(self._question_subs), len(self._student_subs), len(self._q_ids),
        )

    def _retrieve_self_examples(
        self, student_id: str, question_id: str, max_self: int,
        cutoff=None, target_week=None,
    ) -> List[dict]:
        """Return RAG-selected prior submissions as structured example dicts.

        Only includes questions from weeks strictly before target_week.
        """
        s_df = self._student_subs.get(student_id)
        if s_df is None or len(s_df) == 0:
            return []

        other_qs = s_df[s_df["question_unittest_id"].astype(str) != question_id]
        if cutoff is not None:
            other_qs = other_qs[other_qs["timestamp_dt"] < cutoff]

        # Filter to questions from prior weeks only
        if target_week is not None:
            prior_qids = set(
                qid for qid, week in self._q_id_to_week.items()
                if week < target_week
            )
            other_qs = other_qs[other_qs["question_unittest_id"].apply(
                lambda q: str(int(q) if isinstance(q, float) else q) in prior_qids
            )]

        if other_qs.empty:
            return []

        # Group all attempts per question
        q_groups = {}
        for _, row in other_qs.iterrows():
            qid = row["question_unittest_id"]
            q_groups.setdefault(qid, []).append(row)

        # Score each question by similarity + recency (using last attempt timestamp)
        target_idx = self._q_id_to_idx.get(question_id)
        scored = []
        all_ts = [r["timestamp_dt"] for rows in q_groups.values() for r in rows if pd.notna(r.get("timestamp_dt"))]
        max_ts = max(all_ts) if all_ts else None
        decay_halflife = 28 * 24 * 3600

        for qid, rows in q_groups.items():
            qid_str = str(int(qid) if isinstance(qid, float) else qid)
            cand_idx = self._q_id_to_idx.get(qid_str)

            if target_idx is not None and cand_idx is not None and self._sim_matrix is not None:
                sim = float(self._sim_matrix[target_idx, cand_idx])
            else:
                sim = 0.0

            last_ts = max((r.get("timestamp_dt") for r in rows if pd.notna(r.get("timestamp_dt"))), default=None)
            if last_ts and max_ts:
                age = (max_ts - last_ts).total_seconds()
                recency = np.exp(-0.693 * age / decay_halflife)
            else:
                recency = 0.0

            combined = (1 - self._recency_weight) * sim + self._recency_weight * recency
            scored.append((qid, combined, rows))

        scored.sort(key=lambda x: x[1], reverse=True)

        examples = []
        for qid, score, rows in scored[:max_self]:
            qid_str = str(int(qid) if isinstance(qid, float) else qid)
            q_name = self._q_id_to_name.get(qid_str, qid_str)
            for r in rows:
                examples.append({
                    "question_name": q_name,
                    "question_text": str(r.get("question_text", "")),
                    "question_template": str(r.get("question_template", "")),
                    "response": str(r.get("response", "")),
                    "response_type": str(r.get("response_type", "Submit")),
                    "pass_pattern": str(r.get("pass", "")),
                })

        return examples

    @classmethod
    def from_student_split(
        cls,
        full_main_df: pd.DataFrame,
        question_infos: pd.DataFrame,
        train_student_ids: Set[str],
        test_student_ids: Set[str],
        train_week_cutoff: int = 3,
        qid_to_week: Optional[Dict] = None,
    ) -> "RAGRetriever":
        """Construct RAG respecting the student split boundary.

        The filtered main_df contains:
        - All train students' full data (all weeks)
        - Test students' weeks 1-cutoff data only
        """
        full_main_df = full_main_df.copy()
        full_main_df["student_id"] = full_main_df["student_id"].astype(str)

        train_mask = full_main_df["student_id"].isin(train_student_ids)

        if qid_to_week:
            test_mask = full_main_df["student_id"].isin(test_student_ids)
            test_week = full_main_df["question_unittest_id"].map(
                lambda q: qid_to_week.get(int(q), 99) if pd.notna(q) else 99
            )
            test_mask = test_mask & (test_week <= train_week_cutoff)
        else:
            test_mask = full_main_df["student_id"].isin(test_student_ids)

        filtered_df = full_main_df[train_mask | test_mask]
        logger.info("RAG from_student_split: %d train + %d test rows = %d total",
                     train_mask.sum(), test_mask.sum(), len(filtered_df))

        return cls(filtered_df, question_infos)

    def retrieve_self_trajectories(
        self, student_id: str, question_id: str,
        max_self: int = 5, cutoff=None, target_week: Optional[int] = None,
    ) -> List[SelfTrajectory]:
        """Retrieve test student's own prior question trajectories for summarization.

        Ranked by TF-IDF similarity + recency to the target question.
        """
        student_id = str(student_id)
        question_id = str(int(float(question_id)) if "." in str(question_id) else question_id)

        s_df = self._student_subs.get(student_id)
        if s_df is None or len(s_df) == 0:
            return []

        other_qs = s_df[s_df["question_unittest_id"].astype(str) != question_id]
        if cutoff is not None:
            other_qs = other_qs[other_qs["timestamp_dt"] < cutoff]

        if target_week is not None:
            prior_qids = set(
                qid for qid, week in self._q_id_to_week.items()
                if week < target_week
            )
            other_qs = other_qs[other_qs["question_unittest_id"].apply(
                lambda q: str(int(q) if isinstance(q, float) else q) in prior_qids
            )]

        if other_qs.empty:
            return []

        q_groups = {}
        for _, row in other_qs.iterrows():
            qid = row["question_unittest_id"]
            q_groups.setdefault(qid, []).append(row)

        target_idx = self._q_id_to_idx.get(question_id)
        scored = []
        all_ts = [r["timestamp_dt"] for rows in q_groups.values() for r in rows if pd.notna(r.get("timestamp_dt"))]
        max_ts = max(all_ts) if all_ts else None
        decay_halflife = 28 * 24 * 3600

        for qid, rows in q_groups.items():
            qid_str = str(int(qid) if isinstance(qid, float) else qid)
            cand_idx = self._q_id_to_idx.get(qid_str)

            if target_idx is not None and cand_idx is not None and self._sim_matrix is not None:
                sim = float(self._sim_matrix[target_idx, cand_idx])
            else:
                sim = 0.0

            last_ts = max((r.get("timestamp_dt") for r in rows if pd.notna(r.get("timestamp_dt"))), default=None)
            if last_ts and max_ts:
                age = (max_ts - last_ts).total_seconds()
                recency = np.exp(-0.693 * age / decay_halflife)
            else:
                recency = 0.0

            combined = (1 - self._recency_weight) * sim + self._recency_weight * recency
            scored.append((qid, combined, rows))

        scored.sort(key=lambda x: x[1], reverse=True)

        trajectories = []
        for qid, score, rows in scored[:max_self]:
            qid_str = str(int(qid) if isinstance(qid, float) else qid)
            q_name = self._q_id_to_name.get(qid_str, qid_str)

            attempts = []
            for r in sorted(rows, key=lambda x: x.get("timestamp_dt", pd.NaT)):
                attempts.append({
                    "response": _truncate_code(str(r["response"])),
                    "response_type": str(r.get("response_type", "Submit")),
                    "pass_pattern": str(r.get("pass", "")),
                })

            trajectories.append(SelfTrajectory(
                question_name=q_name,
                similarity_score=score,
                attempts=attempts,
            ))

        return trajectories
