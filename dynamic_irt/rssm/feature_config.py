"""Feature group definitions and ablation configurations for Multi-Modal RSSM."""

from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for which feature groups to enable.

    Feature Groups:
        A - Performance (18 dims): testcase pass/fail vector, pass_rate, is_perfect, n_testcases
        B - Temporal (6 dims): time_since_last, attempt_num, cumulative_attempts, is_exam, week, days_since_start
        C - Code Structural (4 dims): code_length, line_count, edit_distance, code_length_ratio
        D - Student State (4 dims): running_avg_correctness, cumulative_ratio, improvement_trend, unique_questions
        E - Question (always on, 19 dims): question_id embedding (16) + difficulty + n_testcases + week
    """

    use_performance: bool = True
    use_temporal: bool = True
    use_code_struct: bool = True
    use_student_state: bool = True
    use_aux_loss: bool = True

    n_testcases: int = 15
    question_emb_dim: int = 16
    question_static_dim: int = 3  # difficulty, n_testcases_norm, week_norm

    @property
    def performance_dim(self) -> int:
        return self.n_testcases + 3  # tc_vector + pass_rate + is_perfect + n_testcases

    @property
    def temporal_dim(self) -> int:
        return 6

    @property
    def code_struct_dim(self) -> int:
        return 4

    @property
    def student_state_dim(self) -> int:
        return 4

    @property
    def answer_dim(self) -> int:
        dim = 0
        if self.use_performance:
            dim += self.performance_dim
        if self.use_temporal:
            dim += self.temporal_dim
        if self.use_code_struct:
            dim += self.code_struct_dim
        if self.use_student_state:
            dim += self.student_state_dim
        return dim

    @property
    def question_dim(self) -> int:
        return self.question_emb_dim + self.question_static_dim


CONFIGS = {
    "full": FeatureConfig(),
    "performance_only": FeatureConfig(
        use_temporal=False, use_code_struct=False, use_student_state=False
    ),
    "no_code": FeatureConfig(use_code_struct=False),
    "minimal": FeatureConfig(use_code_struct=False, use_student_state=False),
    "no_aux": FeatureConfig(use_aux_loss=False),
}
