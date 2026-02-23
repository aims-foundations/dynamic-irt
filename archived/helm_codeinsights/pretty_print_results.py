#!/usr/bin/env python3
"""Pretty-print GLM test results as Markdown for VSCode preview."""

import json
import re
import sys
from pathlib import Path


def extract_code_blocks(text):
    """Extract C++ code blocks from completion text."""
    pattern = r"```(?:cpp|c\+\+)?\s*\n(.*?)```"
    blocks = re.findall(pattern, text, re.DOTALL)
    return blocks


def extract_think_and_code(text):
    """Split completion into thinking and code sections."""
    # Check for <think>...</think> tags
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        rest = text[think_match.end():].strip()
    else:
        # No explicit think tags — the model outputs reasoning then code
        code_blocks = extract_code_blocks(text)
        if code_blocks:
            first_block_start = text.find("```")
            thinking = text[:first_block_start].strip() if first_block_start > 0 else ""
            rest = text[first_block_start:].strip()
        else:
            thinking = text
            rest = ""

    code_blocks = extract_code_blocks(rest or text)
    return thinking, code_blocks


def to_markdown(results_path):
    """Convert results JSON to Markdown."""
    with open(results_path) as f:
        results = json.load(f)

    lines = []
    lines.append(f"# GLM-4.7-AWQ Results — {Path(results_path).stem}")
    lines.append(f"\n**{len(results)} instances** | Source: `{Path(results_path).name}`\n")

    for i, r in enumerate(results):
        lines.append(f"---\n")
        lines.append(f"## [{i+1}/{len(results)}] {r.get('question_name', 'N/A')}\n")
        lines.append(f"- **Question ID:** {r.get('question_id', 'N/A')}")
        if r.get("elapsed_time"):
            lines.append(f"- **Time:** {r['elapsed_time']:.1f}s | **Tokens:** {r.get('completion_tokens', 'N/A')}")
        hit_limit = r.get("completion_tokens", 0) == 4000
        if hit_limit:
            lines.append(f"- **Status:** Hit 4000 token limit (output may be truncated)")
        lines.append("")

        if r.get("error"):
            lines.append(f"> **ERROR:** {r['error']}\n")
            continue

        completion = r.get("completion", "")
        if not completion:
            lines.append("> *(No completion)*\n")
            continue

        thinking, code_blocks = extract_think_and_code(completion)

        # Chain of thought (collapsible)
        if thinking:
            lines.append("<details>")
            lines.append(f"<summary>Chain of Thought ({len(thinking)} chars)</summary>\n")
            lines.append(thinking)
            lines.append("\n</details>\n")

        # Code blocks
        if code_blocks:
            for j, block in enumerate(code_blocks):
                label = f"Code Block {j+1}/{len(code_blocks)}" if len(code_blocks) > 1 else "Code"
                lines.append(f"### {label}\n")
                lines.append("```cpp")
                lines.append(block.rstrip())
                lines.append("```\n")
        else:
            lines.append("### Raw Output (last 500 chars)\n")
            lines.append("```")
            lines.append(completion[-500:])
            lines.append("```\n")

    return "\n".join(lines)


if __name__ == "__main__":
    default_path = Path(__file__).parent / "glm_test_results" / "glm_s1_quick_test.json"
    path = sys.argv[1] if len(sys.argv) > 1 else str(default_path)

    md_content = to_markdown(path)

    # Write .md next to the JSON
    out_path = Path(path).with_suffix(".md")
    with open(out_path, "w") as f:
        f.write(md_content)

    print(f"Markdown written to {out_path}")
    print(f"Open in VSCode and press Ctrl+Shift+V to preview.")
