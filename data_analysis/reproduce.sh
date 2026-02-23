#!/bin/bash
#
# reproduce.sh - Reproduce all data analysis results
#
# This script regenerates all figures and CSVs from the CodeInsights dataset.
# It runs the complete data analysis pipeline including EDA, clustering,
# pacing analysis, and problem-by-problem analysis.
#
# Usage:
#   bash reproduce.sh                    # Run all analyses
#   bash reproduce.sh --skip-eda         # Skip EDA (if already run)
#   bash reproduce.sh --course dsa_hk231 # Run for specific course only
#
# Outputs:
#   - eda_outputs/                       # EDA visualizations
#   - clustering_outputs/                # Student clustering results
#   - pace_outputs/                      # Pacing analysis results
#   - problem_outputs/                   # Problem-level analysis results
#

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
SKIP_EDA=false
COURSE_NAME=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-eda)
      SKIP_EDA=true
      shift
      ;;
    --course)
      COURSE_NAME="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: bash reproduce.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --skip-eda         Skip exploratory data analysis (if already run)"
      echo "  --course NAME      Run analysis for specific course (e.g., dsa_hk231)"
      echo "  --help, -h         Show this help message"
      echo ""
      echo "Examples:"
      echo "  bash reproduce.sh                    # Run all analyses on all courses"
      echo "  bash reproduce.sh --skip-eda         # Skip EDA step"
      echo "  bash reproduce.sh --course dsa_hk231 # Analyze only dsa_hk231"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run with --help for usage information"
      exit 1
      ;;
  esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."  # Go to CodeInsights root

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  CodeInsights Data Analysis Reproduction${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
python -c "import pandas, numpy, matplotlib, seaborn, scipy, sklearn, huggingface_hub" 2>/dev/null
if [ $? -ne 0 ]; then
  echo -e "${RED}Error: Missing required Python packages${NC}"
  echo "Install with: pip install pandas numpy matplotlib seaborn scipy scikit-learn huggingface-hub python-Levenshtein tueplots"
  exit 1
fi
echo -e "${GREEN}✓ All dependencies found${NC}"
echo ""

# Set course flag for scripts
COURSE_FLAG=""
if [ -n "$COURSE_NAME" ]; then
  COURSE_FLAG="--course_name $COURSE_NAME"
  echo -e "${YELLOW}Running analysis for course: $COURSE_NAME${NC}"
else
  echo -e "${YELLOW}Running analysis for all courses${NC}"
fi
echo ""

# Track timing
START_TIME=$(date +%s)

# ============================================================
# 1. Exploratory Data Analysis (EDA)
# ============================================================
if [ "$SKIP_EDA" = false ]; then
  echo -e "${BLUE}[1/4] Running Exploratory Data Analysis...${NC}"
  echo "-------------------------------------------------------"
  python data_analysis/eda_codeinsight.py
  echo -e "${GREEN}✓ EDA complete${NC}"
  echo ""
else
  echo -e "${YELLOW}[1/4] Skipping EDA (--skip-eda flag)${NC}"
  echo ""
fi

# ============================================================
# 2. Student Behavior Clustering
# ============================================================
echo -e "${BLUE}[2/4] Running Student Behavior Clustering...${NC}"
echo "-------------------------------------------------------"
if [ -n "$COURSE_NAME" ]; then
  python data_analysis/student_behavior_clustering.py --course_name "$COURSE_NAME"
else
  python data_analysis/student_behavior_clustering.py
fi
echo -e "${GREEN}✓ Clustering analysis complete${NC}"
echo ""

# ============================================================
# 3. Pace Analysis
# ============================================================
echo -e "${BLUE}[3/4] Running Pace Analysis...${NC}"
echo "-------------------------------------------------------"
if [ -n "$COURSE_NAME" ]; then
  python data_analysis/pace_analysis.py --course_name "$COURSE_NAME"
else
  python data_analysis/pace_analysis.py --all_courses
fi
echo -e "${GREEN}✓ Pace analysis complete${NC}"
echo ""

# ============================================================
# 4. Problem-by-Problem Analysis
# ============================================================
echo -e "${BLUE}[4/4] Running Problem-by-Problem Analysis...${NC}"
echo "-------------------------------------------------------"
if [ -n "$COURSE_NAME" ]; then
  python data_analysis/problem_by_problem_analysis.py --course_name "$COURSE_NAME"
else
  python data_analysis/problem_by_problem_analysis.py --all_courses
fi
echo -e "${GREEN}✓ Problem analysis complete${NC}"
echo ""

# ============================================================
# Summary
# ============================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  All analyses complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total elapsed time: ${MINUTES}m ${SECONDS}s"
echo ""
echo -e "${YELLOW}Outputs saved to:${NC}"
echo "  • data_analysis/eda_outputs/          - EDA visualizations (7 figures)"
echo "  • data_analysis/clustering_outputs/   - Student clustering results"
echo "  • data_analysis/pace_outputs/         - Pacing analysis results"
echo "  • data_analysis/problem_outputs/      - Problem-level analysis results"
echo ""
echo -e "${YELLOW}Figures used in paper:${NC}"
echo "  • student_behavior_clusters_all_courses.png"
echo "  • aggregate_submission_patterns_all_courses.png"
echo "  • aggregate_problem_patterns_all_courses.png"
echo ""
echo -e "${GREEN}To copy figures to overleaf:${NC}"
echo "  cp data_analysis/clustering_outputs/student_behavior_clusters_all_courses.png ../codeinsight-overleaf/figures/"
echo "  cp data_analysis/pace_outputs/aggregate_submission_patterns_all_courses.png ../codeinsight-overleaf/figures/"
echo "  cp data_analysis/problem_outputs/aggregate_problem_patterns_all_courses.png ../codeinsight-overleaf/figures/"
echo ""
