"""Rich student persona construction for LLM simulation.

Computes behavioral metrics, per-topic knowledge state, coding style
indicators, and archetype classification from submission history.
Generates Character Card-style persona text for use as system prompts.

Inspired by Character Card V3 (SillyTavern), Character AI structured
profiles, and educational simulation research (MathVC, Agent4Edu,
Personality-Aware Student Simulation EMNLP 2024).
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from Levenshtein import distance as levenshtein_distance

logger = logging.getLogger(__name__)

COURSE_NAMES = {
    "dsa_hk231": "Data Structures and Algorithms",
    "dsa_hk221": "Data Structures and Algorithms",
    "pf_hk232": "Programming Fundamentals",
    "pf_hk222": "Programming Fundamentals",
}

SECTION_TYPES = {
    "L": "Regular",
    "CC": "Credit-Constrained",
    "CN": "Compact",
    "DT": "Deferred/Repeat",
}


def _pass_rate(pass_str: str) -> float:
    if not pass_str or not isinstance(pass_str, str):
        return 0.0
    clean = pass_str.replace(".", "").strip()
    if not clean:
        return 0.0
    return clean.count("1") / len(clean)


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    if seconds < 3600:
        return f"{seconds / 60:.1f} minutes"
    return f"{seconds / 3600:.1f} hours"


@dataclass
class StudentProfile:
    student_id: str

    avg_time_between_submissions: float
    avg_edit_distance: float
    total_submissions: int
    precheck_ratio: float
    avg_attempts_per_question: float
    rewrite_tendency: float

    overall_pass_rate: float
    recent_pass_rate: float
    improvement_trend: str
    unique_questions_attempted: int

    topic_pass_rates: Dict[str, float] = field(default_factory=dict)
    strongest_topic: str = ""
    weakest_topic: str = ""

    course_name: str = ""
    section_type: str = ""
    weeks_into_course: float = 0.0

    archetype: str = ""
    archetype_description: str = ""

    avg_code_length: float = 0.0
    avg_line_count: float = 0.0
    code_complexity: str = "moderate"


def _normalize_topic(raw_topic: str) -> str:
    """Collapse granular topics into broader categories.

    E.g. 'Exam_L03', 'Exam_CC14' → 'Exam',
         'Pointer_Basic_Inlab' → 'Pointer Basic',
         'ClassString_Postlab' → 'ClassString'
    """
    if raw_topic.startswith("Exam"):
        return "Exam"
    for suffix in ("_Inlab", "_Prelab", "_Postlab", "_InLab", "_PreLab", "_PostLab"):
        if raw_topic.endswith(suffix):
            raw_topic = raw_topic[: -len(suffix)]
            break
    return raw_topic.replace("_", " ").strip()


class PersonaBuilder:
    def __init__(
        self,
        main_df: pd.DataFrame,
        question_infos: pd.DataFrame,
        course_infos: Optional[pd.DataFrame] = None,
        section_infos: Optional[pd.DataFrame] = None,
    ):
        logger.info("PersonaBuilder: computing profiles for all students...")
        self._profiles: Dict[str, StudentProfile] = {}
        self._build_all(main_df, question_infos, course_infos, section_infos)
        logger.info("PersonaBuilder: %d student profiles built.", len(self._profiles))

    def __len__(self):
        return len(self._profiles)

    def _build_all(
        self,
        main_df: pd.DataFrame,
        question_infos: pd.DataFrame,
        course_infos: Optional[pd.DataFrame] = None,
        section_infos: Optional[pd.DataFrame] = None,
    ):
        subs = main_df[main_df["response_type"].isin(["Submit", "Prechecked"])].copy()
        subs = subs.dropna(subset=["response"])
        subs["pass"] = subs["pass"].astype(str).fillna("")
        subs["timestamp_dt"] = pd.to_datetime(
            subs["timestamp"], format="%d/%m/%y, %H:%M:%S", errors="coerce"
        )
        subs = subs.dropna(subset=["timestamp_dt"])
        subs = subs.sort_values(["student_id", "timestamp_dt"])

        q_topics = {}
        if "topic" in question_infos.columns:
            for _, row in question_infos.iterrows():
                qid = row.get("question_id", row.get("question_unittest_id"))
                if pd.notna(qid) and pd.notna(row.get("topic")):
                    q_topics[str(int(qid)) if isinstance(qid, float) else str(qid)] = str(row["topic"])

        course_map = {}
        if course_infos is not None and "course_id" in course_infos.columns:
            for _, row in course_infos.iterrows():
                cname = str(row["course_name"]) if "course_name" in course_infos.columns else str(row["course_id"])
                course_map[str(row["course_id"])] = COURSE_NAMES.get(cname, cname)

        section_id_to_type = {}
        if section_infos is not None and "section_name" in section_infos.columns:
            for _, row in section_infos.iterrows():
                sname = str(row["section_name"])
                prefix = "".join(c for c in sname if c.isalpha()).upper()
                section_id_to_type[str(row["section_id"])] = SECTION_TYPES.get(prefix, "Regular")

        all_time_diffs: Dict[str, List[float]] = {}
        all_edit_dists: Dict[str, List[float]] = {}
        all_pass_rates: Dict[str, List[float]] = {}
        all_sub_counts: Dict[str, int] = {}
        all_precheck_counts: Dict[str, int] = {}
        all_question_sets: Dict[str, set] = {}
        all_code_lengths: Dict[str, List[int]] = {}
        all_line_counts: Dict[str, List[int]] = {}
        topic_scores: Dict[str, Dict[str, List[float]]] = {}
        student_courses: Dict[str, str] = {}
        student_sections: Dict[str, str] = {}
        student_weeks: Dict[str, float] = {}

        for (student_id, qid), g in subs.groupby(["student_id", "question_unittest_id"]):
            sid = str(student_id)
            responses = g["response"].tolist()
            timestamps = g["timestamp_dt"].tolist()
            pass_strs = g["pass"].tolist()
            response_types = g["response_type"].tolist()

            all_sub_counts[sid] = all_sub_counts.get(sid, 0) + len(responses)
            all_question_sets.setdefault(sid, set()).add(str(qid))

            for rt in response_types:
                if rt == "Prechecked":
                    all_precheck_counts[sid] = all_precheck_counts.get(sid, 0) + 1

            for ps in pass_strs:
                all_pass_rates.setdefault(sid, []).append(_pass_rate(ps))

            raw_topic = q_topics.get(str(int(qid)) if isinstance(qid, float) else str(qid))
            if raw_topic:
                topic = _normalize_topic(raw_topic)
                for ps in pass_strs:
                    topic_scores.setdefault(sid, {}).setdefault(topic, []).append(_pass_rate(ps))

            for r in responses:
                if isinstance(r, str):
                    all_code_lengths.setdefault(sid, []).append(len(r))
                    all_line_counts.setdefault(sid, []).append(r.count("\n") + 1)

            if len(responses) >= 2:
                for i in range(1, len(responses)):
                    td = (timestamps[i] - timestamps[i - 1]).total_seconds()
                    if td >= 0:
                        all_time_diffs.setdefault(sid, []).append(td)
                    r_prev = str(responses[i - 1])
                    r_curr = str(responses[i])
                    if len(r_prev) < 50000 and len(r_curr) < 50000:
                        ed = levenshtein_distance(r_curr, r_prev)
                        all_edit_dists.setdefault(sid, []).append(float(ed))

            if sid not in student_courses:
                cid = g["course_id"].iloc[0]
                if pd.notna(cid):
                    cid_str = str(int(cid)) if isinstance(cid, float) else str(cid)
                    student_courses[sid] = course_map.get(cid_str, cid_str)
                secid = g["section_id"].iloc[0] if "section_id" in g.columns else None
                if pd.notna(secid):
                    secid_str = str(int(secid)) if isinstance(secid, float) else str(secid)
                    student_sections[sid] = section_id_to_type.get(secid_str, "Regular")

        global_edit_dists = []
        for dists in all_edit_dists.values():
            global_edit_dists.extend(dists)
        median_edit_dist = float(np.median(global_edit_dists)) if global_edit_dists else 100.0

        student_avg_times = {}
        student_avg_edits = {}
        student_total_subs = {}
        for sid in all_sub_counts:
            student_avg_times[sid] = float(np.mean(all_time_diffs.get(sid, [0])))
            student_avg_edits[sid] = float(np.mean(all_edit_dists.get(sid, [0])))
            student_total_subs[sid] = all_sub_counts[sid]

        time_vals = list(student_avg_times.values())
        edit_vals = list(student_avg_edits.values())
        sub_vals = list(student_total_subs.values())
        time_p25, time_p75 = np.percentile(time_vals, 25), np.percentile(time_vals, 75) if time_vals else (0, 0)
        edit_p50 = np.percentile(edit_vals, 50) if edit_vals else 0
        sub_p50 = np.percentile(sub_vals, 50) if sub_vals else 0

        for sid in all_sub_counts:
            avg_time = student_avg_times[sid]
            avg_edit = student_avg_edits[sid]
            total_subs = student_total_subs[sid]
            n_questions = len(all_question_sets.get(sid, set()))
            pass_rates_list = all_pass_rates.get(sid, [0])
            precheck_count = all_precheck_counts.get(sid, 0)

            overall_pr = float(np.mean(pass_rates_list))
            recent_pr = float(np.mean(pass_rates_list[-10:])) if pass_rates_list else 0.0

            if len(pass_rates_list) >= 6:
                recent_3 = np.mean(pass_rates_list[-3:])
                prev_3 = np.mean(pass_rates_list[-6:-3])
                delta = recent_3 - prev_3
                if delta > 0.05:
                    trend = "improving"
                elif delta < -0.05:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            precheck_ratio = precheck_count / total_subs if total_subs > 0 else 0.0
            avg_attempts = total_subs / n_questions if n_questions > 0 else total_subs

            edit_dists = all_edit_dists.get(sid, [])
            rewrite_tendency = sum(1 for e in edit_dists if e > median_edit_dist) / len(edit_dists) if edit_dists else 0.5

            t_pass_rates = {}
            strongest = ""
            weakest = ""
            if sid in topic_scores:
                for topic, scores in topic_scores[sid].items():
                    t_pass_rates[topic] = float(np.mean(scores))
                if t_pass_rates:
                    strongest = max(t_pass_rates, key=t_pass_rates.get)
                    weakest = min(t_pass_rates, key=t_pass_rates.get)

            code_lens = all_code_lengths.get(sid, [])
            line_cnts = all_line_counts.get(sid, [])
            avg_clen = float(np.mean(code_lens)) if code_lens else 0.0
            avg_lcnt = float(np.mean(line_cnts)) if line_cnts else 0.0
            if avg_lcnt < 15:
                complexity = "concise"
            elif avg_lcnt > 40:
                complexity = "verbose"
            else:
                complexity = "moderate"

            archetype, arch_desc = _classify_archetype(
                avg_time, avg_edit, total_subs,
                time_p25, time_p75, edit_p50, sub_p50,
            )

            self._profiles[sid] = StudentProfile(
                student_id=sid,
                avg_time_between_submissions=avg_time,
                avg_edit_distance=avg_edit,
                total_submissions=total_subs,
                precheck_ratio=precheck_ratio,
                avg_attempts_per_question=avg_attempts,
                rewrite_tendency=rewrite_tendency,
                overall_pass_rate=overall_pr,
                recent_pass_rate=recent_pr,
                improvement_trend=trend,
                unique_questions_attempted=n_questions,
                topic_pass_rates=t_pass_rates,
                strongest_topic=strongest,
                weakest_topic=weakest,
                course_name=student_courses.get(sid, ""),
                section_type=student_sections.get(sid, "Regular"),
                weeks_into_course=0.0,
                archetype=archetype,
                archetype_description=arch_desc,
                avg_code_length=avg_clen,
                avg_line_count=avg_lcnt,
                code_complexity=complexity,
            )

    def build_profile(self, student_id: str) -> Optional[StudentProfile]:
        return self._profiles.get(str(student_id))

    def build_persona_text(self, student_id: str) -> Optional[str]:
        p = self.build_profile(student_id)
        if p is None:
            return None

        precheck_pct = p.precheck_ratio * 100
        submit_pct = 100 - precheck_pct

        topic_block = ""
        if p.topic_pass_rates:
            sorted_topics = sorted(p.topic_pass_rates.items(), key=lambda x: x[1], reverse=True)
            shown = sorted_topics[:8]
            topic_block = ", ".join(f"{t}: {r:.0%}" for t, r in shown)
            if len(sorted_topics) > 8:
                topic_block += f" (and {len(sorted_topics) - 8} more topics)"

        tendency_str = "Rewrites large portions of code" if p.rewrite_tendency > 0.5 else "Makes targeted, small patches"

        parts = [
            "You are simulating a specific university student solving C++ programming assignments. "
            "Your responses must authentically reflect this student's ability level, coding patterns, and problem-solving approach.",
            "",
            "=== STUDENT IDENTITY ===",
            "",
        ]

        bg_parts = []
        if p.course_name:
            bg_parts.append(f"a {p.course_name} student")
        if p.section_type:
            bg_parts.append(f"in a {p.section_type} section")
        bg_parts.append(f"who has attempted {p.unique_questions_attempted} distinct problems with {p.total_submissions} total submissions")
        parts.append("Background: " + " ".join(bg_parts) + ".")

        parts.append("")
        parts.append(f"Behavioral Style: {p.archetype_description}")
        parts.append(f"- Average time between submissions: {_format_time(p.avg_time_between_submissions)}")
        parts.append(f"- Average code change per attempt: {p.avg_edit_distance:.0f} characters (Levenshtein)")
        parts.append(f"- Precheck vs Submit ratio: {precheck_pct:.0f}% Precheck, {submit_pct:.0f}% Submit")
        parts.append(f"- Tendency: {tendency_str}")

        parts.append("")
        parts.append(f"Overall pass rate: {p.overall_pass_rate:.0%}")

        if topic_block:
            parts.append(f"Topic proficiency: {topic_block}")

        parts.append(
            f"Typical code length: {p.avg_code_length:.0f} characters ({p.avg_line_count:.0f} lines), {p.code_complexity} style. "
            f"Attempts per problem: {p.avg_attempts_per_question:.1f} on average."
        )

        parts.append("")
        parts.append("Embody this student fully. Your coding ability, problem-solving approach, "
                     "and submission behavior should be consistent with the profile above.")

        return "\n".join(parts)


def _classify_archetype(
    avg_time: float, avg_edit: float, total_subs: int,
    time_p25: float, time_p75: float, edit_p50: float, sub_p50: float,
) -> Tuple[str, str]:
    if avg_time < time_p25 and total_subs > sub_p50:
        return (
            "rapid_iterator",
            "Makes many quick, small changes. Submits frequently to test incrementally. Debugging style: trial-and-error.",
        )
    if time_p25 <= avg_time <= time_p75 and avg_edit > edit_p50:
        return (
            "deliberate_planner",
            "Takes moderate time between attempts but makes substantial code changes each time. Thinks before editing, then rewrites significant portions.",
        )
    if avg_time > time_p75 and total_subs < sub_p50:
        return (
            "careful_thinker",
            "Spends significant time between submissions. Makes fewer total attempts. Plans solutions carefully before coding.",
        )
    return (
        "balanced",
        "Shows a mix of strategies depending on the problem. Adapts approach based on difficulty.",
    )
