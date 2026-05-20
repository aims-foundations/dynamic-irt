#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

OVERLEAF_FIGS=overleaf/figures

REGRADE=false
REBUILD_TRAJECTORIES=false
for arg in "$@"; do
    case "$arg" in
        --regrade) REGRADE=true ;;
        --rebuild-trajectories) REBUILD_TRAJECTORIES=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

if [ "$REGRADE" = true ]; then
    echo "Removing graded_pairs.csv cache..."
    rm -f results/llm_predictor/bug_comparison/graded_pairs.csv
fi

if [ "$REBUILD_TRAJECTORIES" = true ]; then
    echo "Removing trajectory_metrics.csv cache..."
    rm -f results/llm_predictor/behavioral/trajectory_metrics.csv
fi

echo "=== Learning curves ==="
python3 data_analysis/learning_curves_by_year.py
cp data_analysis/clustering_outputs/learning_curves_combined.png "$OVERLEAF_FIGS/"

echo "=== Response matrix heatmap + avg improvement ==="
python3 data_analysis/visualize_response_matrix.py --course dsa_hk231
cp results/student_eval/dsa_hk231/figures/attempt_progression_compact.png "$OVERLEAF_FIGS/"
cp results/student_eval/dsa_hk231/figures/avg_improvement_by_attempt.png "$OVERLEAF_FIGS/"

echo "=== Balanced accuracy ==="
python3 data_analysis/plot_filtered_accuracy.py
cp results/student_eval/dsa_hk231/accuracy_vs_attempt.png "$OVERLEAF_FIGS/"

echo "=== Error type flow ==="
PYTHONPATH=. python3 data_analysis/llm_bug_comparison.py
cp results/llm_predictor/bug_comparison/error_type_flow.png "$OVERLEAF_FIGS/"

echo "=== Kendall tau decomposition ==="
PYTHONPATH=. python3 data_analysis/kendall_tau_decomposition.py
cp results/llm_predictor/student_split/decomposition_test.png "$OVERLEAF_FIGS/"

echo "=== Behavioral comparison ==="
PYTHONPATH=. python3 data_analysis/llm_behavioral_comparison.py
cp results/llm_predictor/behavioral/behavioral_combined.png "$OVERLEAF_FIGS/"

echo "=== Problem patterns ==="
python3 data_analysis/problem_by_problem_analysis.py --all_courses
cp data_analysis/problem_outputs/aggregate_problem_patterns_left_all_courses.png "$OVERLEAF_FIGS/"

echo "=== Submission patterns ==="
python3 data_analysis/pace_analysis.py --all_courses
cp data_analysis/pace_outputs/aggregate_submission_patterns_left_all_courses.png "$OVERLEAF_FIGS/"

echo "=== Student behavior clustering ==="
python3 data_analysis/student_behavior_clustering.py
cp data_analysis/clustering_outputs/student_behavior_all_courses.png "$OVERLEAF_FIGS/"

echo ""
echo "All figures copied to $OVERLEAF_FIGS"
