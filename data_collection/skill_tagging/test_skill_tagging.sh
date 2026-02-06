#!/bin/bash
# Test script for skill tagging
# This script tests the C++ skill tagging pipeline for CodeInsight questions

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== CodeInsight Skill Tagging Test ===${NC}"
echo ""

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
python -c "import pandas; import torch; import openai; import tqdm" 2>/dev/null || {
    echo -e "${RED}Missing dependencies. Install with:${NC}"
    echo "pip install pandas torch openai tqdm huggingface-hub together"
    exit 1
}
echo -e "${GREEN}Dependencies OK${NC}"
echo ""

# Check if C++ skills file exists
if [ -f "data/cpp_programming.json" ]; then
    echo -e "${GREEN}Found C++ skills hierarchy: data/cpp_programming.json${NC}"
    echo "Categories:"
    python -c "import json; data=json.load(open('data/cpp_programming.json')); print('  - ' + '\n  - '.join(data.keys()))"
else
    echo -e "${RED}Missing data/cpp_programming.json${NC}"
    exit 1
fi
echo ""

# Usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --vllm URL       Use vLLM server at URL (e.g., http://localhost:8080/v1)"
    echo "  --together       Use Together API (requires TOGETHER_API_KEY)"
    echo "  --dry-run        Test without making API calls"
    echo "  --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Test with local vLLM server"
    echo "  $0 --vllm http://localhost:8080/v1"
    echo ""
    echo "  # Test with Together API"
    echo "  export TOGETHER_API_KEY=your_key"
    echo "  $0 --together"
    echo ""
    echo "  # Dry run (no API calls)"
    echo "  $0 --dry-run"
}

# Parse arguments
DRY_RUN=false
MODEL_URL=""
USE_TOGETHER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --vllm)
            MODEL_URL="$2"
            shift 2
            ;;
        --together)
            USE_TOGETHER=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}=== DRY RUN MODE ===${NC}"
    echo "Testing skill hierarchy loading and prompt generation..."
    echo ""

    python << 'EOF'
import json

# Load C++ skills
with open("data/cpp_programming.json") as f:
    skills = json.load(f)

print("Loaded C++ skill categories:")
for category, data in skills.items():
    standard = data.get("standard", category)
    num_skills = sum(
        len(item.get("content", [])) if isinstance(item.get("content", []), list) else 0
        for item in data.get("data", [])
    )
    print(f"  - {category} ({standard}): ~{num_skills} skills")

# Test prompt generation
test_question = """
Implement a function to reverse a singly linked list.
The function should take the head of the list and return the new head after reversal.

struct Node {
    int data;
    Node* next;
};

Node* reverseList(Node* head);
"""

print("\n" + "="*50)
print("Test question:")
print(test_question[:100] + "...")
print("\nExpected skill tags might include:")
print("  - D.2.1: Singly linked list implementation")
print("  - D.2.4: Linked list operations (insert, delete, search)")
print("  - M.1.1: Pointer declaration and dereferencing")
print("  - A.5.2: Recursive problem solving (if recursive approach)")
print("\nDry run complete!")
EOF
    exit 0
fi

# Check API configuration
if [ "$USE_TOGETHER" = true ]; then
    if [ -z "$TOGETHER_API_KEY" ]; then
        echo -e "${RED}Error: TOGETHER_API_KEY not set${NC}"
        echo "Export your API key: export TOGETHER_API_KEY=your_key"
        exit 1
    fi
    echo -e "${GREEN}Using Together API${NC}"
    MODEL_URL="together"
elif [ -n "$MODEL_URL" ]; then
    echo -e "${GREEN}Using vLLM server at: $MODEL_URL${NC}"
    # Test connection
    curl -s --connect-timeout 5 "$MODEL_URL/models" > /dev/null 2>&1 || {
        echo -e "${RED}Cannot connect to vLLM server at $MODEL_URL${NC}"
        echo "Make sure the server is running:"
        echo "  vllm serve meta-llama/Llama-3.3-70B-Instruct --port 8080"
        exit 1
    }
    echo -e "${GREEN}vLLM server is reachable${NC}"
else
    echo -e "${RED}Error: No API backend specified${NC}"
    usage
    exit 1
fi

echo ""
echo -e "${YELLOW}Running skill tagging...${NC}"
echo "Note: run_tagging.py needs to be modified to use cpp_programming.json"
echo "Current version uses K-12 subjects. See run_tagging_cpp.py for C++ version."
echo ""

# Run the tagging script
# python run_tagging_cpp.py --model_url "$MODEL_URL" --dataset "codeinsight"
echo -e "${GREEN}Test setup complete!${NC}"
