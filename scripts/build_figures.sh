#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

OVERLEAF_FIGS=overleaf/figures

REGRADE=false
for arg in "$@"; do
    case "$arg" in
        --regrade) REGRADE=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

if [ "$REGRADE" = true ]; then
    echo "Removing graded_pairs caches..."
    rm -f results/llm_predictor/bug_comparison/graded_pairs*.csv
fi

echo "=== Platform figure (annotated) ==="
python3 data_analysis/platform_figure.py
cp data_analysis/schematic_outputs/codeinsights_platform_annotated.png "$OVERLEAF_FIGS/codeinsight/"

echo "=== Learning curves ==="
python3 data_analysis/learning_curves_by_year.py
cp data_analysis/clustering_outputs/learning_curves_combined.png "$OVERLEAF_FIGS/"

echo "=== Response matrix heatmap ==="
python3 data_analysis/visualize_response_matrix.py --course dsa_hk231
cp results/student_eval/dsa_hk231/figures/attempt_progression_compact.png "$OVERLEAF_FIGS/"

echo "=== Balanced accuracy ==="
python3 data_analysis/plot_filtered_accuracy.py --courses dsa_hk231 dsa_hk221

echo "=== Metrics table ==="
python3 data_analysis/model_metrics_table.py --write-tex

echo "=== LLM ablations ==="
PYTHONPATH=. python3 data_analysis/plot_llm_ablations.py --course dsa_hk231
cp results/llm_student_eval/dsa_hk231/llm_combined.png "$OVERLEAF_FIGS/llm_ablation_stacked.png"

echo "=== Error type flow ==="
PYTHONPATH=. python3 data_analysis/llm_bug_comparison.py --jsonl results/llm_student_eval/dsa_hk231/qwen_server_attempts10.jsonl
cp results/llm_predictor/bug_comparison/error_type_flow.png "$OVERLEAF_FIGS/"

echo "=== Kendall tau decomposition (per model) ==="
LLM_EVAL_DIR=results/llm_student_eval/dsa_hk231
PYTHONPATH=. python3 data_analysis/kendall_tau_decomposition.py --jsonl "$LLM_EVAL_DIR/claude_attempts10.jsonl" --output_dir "$LLM_EVAL_DIR/analysis_opus"
PYTHONPATH=. python3 data_analysis/kendall_tau_decomposition.py --jsonl "$LLM_EVAL_DIR/qwen_server_attempts10.jsonl" --output_dir "$LLM_EVAL_DIR/analysis_qwen"
PYTHONPATH=. python3 data_analysis/kendall_tau_decomposition.py --jsonl "$LLM_EVAL_DIR/gemma_server_attempts10.jsonl" --output_dir "$LLM_EVAL_DIR/analysis_gemma"

echo "=== Kendall tau decomposition grid ==="
PYTHONPATH=. python3 data_analysis/plot_llm_analysis_grid.py

echo "=== Problem patterns ==="
python3 data_analysis/problem_by_problem_analysis.py --all_courses
cp data_analysis/problem_outputs/aggregate_problem_patterns_left_all_courses.png "$OVERLEAF_FIGS/"

echo "=== Submission patterns ==="
python3 data_analysis/pace_analysis.py --all_courses
cp data_analysis/pace_outputs/aggregate_submission_patterns_left_all_courses.png "$OVERLEAF_FIGS/"

echo "=== Student behavior clustering ==="
python3 data_analysis/student_behavior_clustering.py
cp data_analysis/clustering_outputs/student_behavior_all_courses.png "$OVERLEAF_FIGS/"

echo "=== PF vs DSA summary ==="
PYTHONPATH=. python3 data_analysis/pf_vs_dsa_analysis.py
cp results/rssm_analysis/report/pf_vs_dsa_summary.png "$OVERLEAF_FIGS/"

echo ""
echo "All figures copied to $OVERLEAF_FIGS"
