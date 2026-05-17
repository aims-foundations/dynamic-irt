"""
Skill tagging for C++ programming questions (CodeInsight dataset).
Tags questions with required skills from the cpp_programming.json hierarchy.
"""
import argparse
import json
import os

import pandas as pd
from huggingface_hub import snapshot_download
from openai import OpenAI
from tqdm import tqdm

try:
    from together import Together
except ImportError:
    Together = None


# C++ skill categories
CPP_CATEGORIES = [
    "Fundamentals",
    "Memory Management",
    "Object-Oriented Programming",
    "Data Structures",
    "Algorithms",
    "Complexity Analysis",
    "STL and Templates",
]

PROMPT_CATEGORY = """You are a quality assurance specialist working in computer science education.
You are tasked with tagging the C++ programming skill categories that the following question tests.
The list of categories is:
(1) Fundamentals (variables, control flow, functions, I/O)
(2) Memory Management (pointers, dynamic memory, references)
(3) Object-Oriented Programming (classes, inheritance, operator overloading)
(4) Data Structures (arrays, linked lists, trees, graphs, hash tables)
(5) Algorithms (sorting, searching, recursion, DP, greedy)
(6) Complexity Analysis (time/space complexity)
(7) STL and Templates (containers, iterators, templates)

Please tag the categories that the following question tests and return them in a Python list, and nothing else.
Example: ['Data Structures', 'Algorithms'].
Question: {question}"""

PROMPT_SKILL = """You are a quality assurance specialist working in computer science education.
You are tasked with tagging the specific skills that the following C++ question tests.
The question must have at least one of the following skills.
The list of skills is:
{desc}

Please tag the skills by their index numbers and return them in a Python list, and nothing else.
Example: ['1', '2'].
Question: {question}"""


def generate(client, model, messages, **kwargs):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
    return completion.choices


def get_categories(question, client, model):
    """Get C++ skill categories for a question."""
    messages = [{"role": "user", "content": PROMPT_CATEGORY.format(question=question)}]

    for _ in range(5):
        try:
            completions = generate(client=client, model=model, messages=messages)
            categories = eval(completions[0].message.content)
            if len(categories) > 0 and all(cat in CPP_CATEGORIES for cat in categories):
                return categories
        except Exception:
            continue

    # Fallback: return empty or partial match
    try:
        return [x for x in categories if x in CPP_CATEGORIES]
    except:
        return []


def get_skills_for_level(question, skills_list, client, model):
    """Get specific skills from a list."""
    standards = [
        f"{idx + 1}. {skill}" for idx, skill in enumerate(skills_list)
    ]
    desc = "\n".join(standards)
    messages = [{"role": "user", "content": PROMPT_SKILL.format(question=question, desc=desc)}]

    for _ in range(5):
        try:
            completions = generate(client=client, model=model, messages=messages)
            indices = eval(completions[0].message.content)
            indices = [int(str(i).split(".")[0]) for i in indices]
            if len(indices) > 0 and all(1 <= i <= len(skills_list) for i in indices):
                return indices
        except Exception:
            continue

    try:
        return [i for i in indices if 1 <= i <= len(skills_list)]
    except:
        return []


def extract_skills_recursive(question, content, client, model, depth=0):
    """Recursively extract skills from hierarchical structure."""
    results = []

    if not content:
        return results

    # Check if this is the leaf level (has 'id' field)
    if isinstance(content, list) and len(content) > 0:
        if "id" in content[0]:
            # Leaf level - get specific skills
            skill_names = [item["content"] for item in content]
            indices = get_skills_for_level(question, skill_names, client, model)
            for idx in indices:
                results.append(content[idx - 1])
        else:
            # Intermediate level - has lv{n}_id
            level_key = f"lv{depth + 1}_id"
            if level_key in content[0]:
                skill_names = [item[level_key] for item in content]
                indices = get_skills_for_level(question, skill_names, client, model)
                for idx in indices:
                    item = content[idx - 1]
                    sub_results = extract_skills_recursive(
                        question, item.get("content", []), client, model, depth + 1
                    )
                    results.append({
                        "level": item[level_key],
                        "skills": sub_results
                    })

    return results


def tag_question(question, skills_by_category, client, model):
    """Tag a single question with C++ skills."""
    # Step 1: Get relevant categories
    categories = get_categories(question, client, model)
    print(f"  Categories: {categories}")

    if not categories:
        return {"categories": [], "skills": []}

    # Step 2: For each category, drill down to specific skills
    all_skills = []
    for category in categories:
        if category not in skills_by_category:
            continue

        category_data = skills_by_category[category]
        standard = category_data.get("standard", category)
        data = category_data.get("data", [])

        # Get skills from this category
        skills = extract_skills_recursive(question, data, client, model)

        all_skills.append({
            "category": category,
            "standard": standard,
            "skills": skills
        })

    return {"categories": categories, "skills": all_skills}


def main():
    parser = argparse.ArgumentParser(description="Tag C++ questions with skills")
    parser.add_argument("--model_url", type=str, default="http://localhost:8080/v1",
                        help="vLLM server URL or 'together' for Together API")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
                        help="Model to use for tagging")
    parser.add_argument("--dataset", type=str, default="codeinsight",
                        help="Dataset name (for output file naming)")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Input CSV with 'raw_question' column (optional)")
    parser.add_argument("--max_questions", type=int, default=None,
                        help="Maximum number of questions to process")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # Setup client
    if args.model_url == "together":
        if Together is None:
            raise ImportError("together package not installed. Run: pip install together")
        client = Together()
    else:
        client = OpenAI(
            base_url=args.model_url,
            api_key="token-abc123",
        )

    # Load C++ skills hierarchy
    script_dir = os.path.dirname(os.path.abspath(__file__))
    skills_path = os.path.join(script_dir, "data/cpp_programming.json")
    with open(skills_path) as f:
        skills_by_category = json.load(f)

    print(f"Loaded {len(skills_by_category)} C++ skill categories")

    # Load questions
    if args.input_file:
        dataset = pd.read_csv(args.input_file)
    else:
        # Try to load from HuggingFace
        try:
            data_folder = snapshot_download(
                repo_id="CodeInsightTeam/code_insights_csv", repo_type="dataset"
            )
            # Look for questions file
            questions_file = os.path.join(data_folder, "questions.csv")
            if os.path.exists(questions_file):
                dataset = pd.read_csv(questions_file)
            else:
                print("No questions file found. Please provide --input_file")
                return
        except Exception as e:
            print(f"Could not load from HuggingFace: {e}")
            print("Please provide --input_file with a CSV containing 'raw_question' column")
            return

    # Check for question column
    question_col = None
    for col in ["raw_question", "question", "problem", "content"]:
        if col in dataset.columns:
            question_col = col
            break

    if question_col is None:
        print(f"No question column found. Available columns: {list(dataset.columns)}")
        return

    print(f"Using column '{question_col}' for questions")
    print(f"Total questions: {len(dataset)}")

    # Load existing results if any
    output_file = f"results/{args.dataset}_cpp_skills.json"
    if os.path.exists(output_file):
        with open(output_file) as f:
            question_infos = json.load(f)
        start_idx = len(question_infos)
        print(f"Resuming from question {start_idx}")
    else:
        question_infos = []
        start_idx = 0

    # Process questions
    questions = dataset[question_col].tolist()
    if args.max_questions:
        questions = questions[:args.max_questions]

    for qi, question in enumerate(tqdm(questions[start_idx:], initial=start_idx, total=len(questions))):
        print(f"\nQuestion {qi + start_idx + 1}:")
        print(f"  {str(question)[:100]}...")

        result = tag_question(question, skills_by_category, client, args.model)

        question_infos.append({
            "index": qi + start_idx,
            "question": question[:500],  # Truncate for storage
            **result
        })

        # Save periodically
        if (qi + 1) % 10 == 0:
            with open(output_file, "w") as f:
                json.dump(question_infos, f, indent=2)
            print(f"  Saved {len(question_infos)} results")

    # Final save
    with open(output_file, "w") as f:
        json.dump(question_infos, f, indent=2)

    print(f"\nDone! Results saved to {output_file}")


if __name__ == "__main__":
    main()
