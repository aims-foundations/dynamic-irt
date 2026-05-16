"""RAG context retrieval for LLM student simulation.

Retrieves similar submissions to provide additional context:
1. Peer submissions: Other students' work on the same question at similar ability
2. Self-similar: The target student's own prior work on similar problems

Embeddings are lazy-loaded from HuggingFace. Falls back to pass-rate
filtering when embeddings are unavailable.
"""

import logging
from typing import Dict, List, Optional, Tuple

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

    def _try_load_embeddings(self):
        if self._embeddings_loaded:
            return
        self._embeddings_loaded = True
        try:
            from datasets import load_dataset
            logger.info("RAGRetriever: loading embeddings from HuggingFace...")
            ds = load_dataset("CodeInsightTeam/code_insights_csv", split="embeddings")
            self._embeddings = np.array(ds["embedding"], dtype=np.float32)
            logger.info("RAGRetriever: loaded %d embeddings.", len(self._embeddings))
        except Exception as e:
            logger.warning("RAGRetriever: could not load embeddings (%s). Using pass-rate fallback.", e)
            self._embeddings = None

    def retrieve_examples(
        self,
        student_id: str,
        question_id: str,
        target_timestamp: Optional[str] = None,
        max_self: int = 3,
    ) -> List[dict]:
        """Retrieve prior submissions as structured example dicts.

        Only retrieves from weeks strictly before the target question's week,
        matching the temporal split used by other models (RSSM, etc.).

        Returns list of dicts with keys: question_name, question_text,
        question_template, response, response_type, pass_pattern.
        Same format as what the summarizer expects.
        """
        student_id = str(student_id)
        question_id = str(int(float(question_id)) if "." in str(question_id) else question_id)

        cutoff = None
        if target_timestamp:
            cutoff = pd.to_datetime(target_timestamp, format="%d/%m/%y, %H:%M:%S", errors="coerce")

        target_week = self._q_id_to_week.get(question_id)

        return self._retrieve_self_examples(student_id, question_id, max_self, cutoff, target_week)

    def retrieve_context(
        self,
        student_id: str,
        question_id: str,
        student_pass_rate: float,
        target_timestamp: Optional[str] = None,
        max_peers: int = 3,
        max_self: int = 3,
    ) -> Optional[str]:
        student_id = str(student_id)
        question_id = str(int(float(question_id)) if "." in str(question_id) else question_id)

        cutoff = None
        if target_timestamp:
            cutoff = pd.to_datetime(target_timestamp, format="%d/%m/%y, %H:%M:%S", errors="coerce")

        self_block = self._retrieve_self_similar(student_id, question_id, max_self, cutoff)
        if not self_block:
            return None
        return "=== Your Prior Work on Similar Problems ===\n\n" + self_block

    def _retrieve_peers(
        self, student_id: str, question_id: str,
        student_pass_rate: float, max_peers: int,
        cutoff=None,
    ) -> Optional[str]:
        q_df = self._question_subs.get(question_id)
        if q_df is None or len(q_df) == 0:
            return None

        others = q_df[q_df["student_id"].astype(str) != student_id].copy()
        if cutoff is not None:
            others = others[others["timestamp_dt"] < cutoff]
        if others.empty:
            return None

        others = others.copy()
        others["student_pr"] = others["student_id"].astype(str).map(self._student_pass_rates)
        others = others.dropna(subset=["student_pr"])
        margin = 0.15
        similar = others[
            (others["student_pr"] >= student_pass_rate - margin)
            & (others["student_pr"] <= student_pass_rate + margin)
        ]

        if similar.empty:
            similar = others.copy()
            similar["pr_dist"] = (similar["student_pr"] - student_pass_rate).abs()
            similar = similar.nsmallest(max_peers * 2, "pr_dist")

        selected = similar.drop_duplicates(subset=["student_id"]).head(max_peers)
        if selected.empty:
            return None

        lines = ["Submissions by students at a similar level on this problem:"]
        for _, row in selected.iterrows():
            pr = row.get("student_pr", 0)
            code = _truncate_code(str(row["response"]))
            pp = str(row.get("pass", ""))
            lines.append(f"\n--- Peer (overall pass rate: {pr:.0%}, result: {pp}) ---")
            lines.append(code)

        return "\n".join(lines)

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

    def _retrieve_self_similar(
        self, student_id: str, question_id: str, max_self: int,
        cutoff=None,
    ) -> Optional[str]:
        s_df = self._student_subs.get(student_id)
        if s_df is None or len(s_df) == 0:
            return None

        other_qs = s_df[s_df["question_unittest_id"].astype(str) != question_id]
        if cutoff is not None:
            other_qs = other_qs[other_qs["timestamp_dt"] < cutoff]
        if other_qs.empty:
            return None

        last_per_q = other_qs.sort_values("timestamp").groupby("question_unittest_id").last()
        if last_per_q.empty:
            return None

        # Score each prior question: similarity to target + recency bias
        target_idx = self._q_id_to_idx.get(question_id)
        scores = []
        max_ts = last_per_q["timestamp_dt"].max()
        decay_halflife = 28 * 24 * 3600  # 4 weeks in seconds

        for qid, row in last_per_q.iterrows():
            qid_str = str(int(qid) if isinstance(qid, float) else qid)
            cand_idx = self._q_id_to_idx.get(qid_str)

            # Similarity score (0-1)
            if target_idx is not None and cand_idx is not None and self._sim_matrix is not None:
                sim = float(self._sim_matrix[target_idx, cand_idx])
            else:
                sim = 0.0

            # Recency score: exponential decay with 1-week half-life
            ts = row.get("timestamp_dt")
            if pd.notna(ts) and pd.notna(max_ts):
                age_seconds = (max_ts - ts).total_seconds()
                recency = np.exp(-0.693 * age_seconds / decay_halflife)  # 0.693 = ln(2)
            else:
                recency = 0.0

            combined = (1 - self._recency_weight) * sim + self._recency_weight * recency
            scores.append((qid, combined, row))

        scores.sort(key=lambda x: x[1], reverse=True)
        selected = scores[:max_self]

        lines = ["Your own solutions to similar problems:"]
        for qid, score, row in selected:
            code = _truncate_code(str(row["response"]))
            pp = str(row.get("pass", ""))
            qid_str = str(int(qid) if isinstance(qid, float) else qid)
            q_name = self._q_id_to_name.get(qid_str, qid_str)
            lines.append(f"\n--- Your solution to \"{q_name}\" (result: {pp}) ---")
            lines.append(code)

        return "\n".join(lines)
