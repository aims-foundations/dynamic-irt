#!/usr/bin/env python3
"""Analyze truncation in GLM S1 results."""
import json, re, sys

results_path = sys.argv[1] if len(sys.argv) > 1 else "helm_results/evaluation/glm/S1/results.json"

with open(results_path) as f:
    results = json.load(f)

code_complete = 0
code_truncated = 0
still_thinking = 0
no_think_no_code = 0

for r in results:
    comp = r.get("completion", "") or ""

    # Check if still in <think> block (never got to code)
    has_think_open = "<think>" in comp
    has_think_close = "</think>" in comp

    if has_think_open and not has_think_close:
        still_thinking += 1
        continue

    # Extract text after </think> (or full text if no think tags)
    if has_think_close:
        after_think = comp[comp.index("</think>") + len("</think>"):]
    else:
        after_think = comp

    # Check for complete code blocks
    code_blocks = re.findall(r"```(?:cpp|c\+\+)?\s*\n(.*?)```", after_think, re.DOTALL)
    open_fences = len(re.findall(r"```(?:cpp|c\+\+)?", after_think))
    close_fences = len(re.findall(r"```\s*$", after_think, re.MULTILINE)) + after_think.count("```\n")

    if code_blocks:
        code_complete += 1
    elif "```cpp" in after_think or "```c++" in after_think:
        code_truncated += 1
    else:
        no_think_no_code += 1

print(f"Total: {len(results)}")
print(f"Has complete code block(s):    {code_complete} ({code_complete*100/len(results):.0f}%)")
print(f"Code truncated mid-block:      {code_truncated} ({code_truncated*100/len(results):.0f}%)")
print(f"Still in <think> (no code):    {still_thinking} ({still_thinking*100/len(results):.0f}%)")
print(f"No think, no code block:       {no_think_no_code} ({no_think_no_code*100/len(results):.0f}%)")

# Show examples of each category
print("\n--- Examples of 'still_thinking' (truncated in CoT) ---")
count = 0
for r in results:
    comp = r.get("completion", "") or ""
    if "<think>" in comp and "</think>" not in comp:
        print(f"  Instance {r['instance_id']}: {len(comp)} chars, ends with: ...{comp[-80:]!r}")
        count += 1
        if count >= 3:
            break

print("\n--- Examples of 'code_truncated' (truncated mid-code) ---")
count = 0
for r in results:
    comp = r.get("completion", "") or ""
    if "<think>" in comp and "</think>" not in comp:
        continue
    after = comp[comp.index("</think>") + 8:] if "</think>" in comp else comp
    blocks = re.findall(r"```(?:cpp|c\+\+)?\s*\n(.*?)```", after, re.DOTALL)
    if not blocks and ("```cpp" in after or "```c++" in after):
        print(f"  Instance {r['instance_id']}: {len(comp)} chars, ends with: ...{comp[-80:]!r}")
        count += 1
        if count >= 3:
            break

print("\n--- Completion length by category ---")
thinking_lens = []
complete_lens = []
for r in results:
    comp = r.get("completion", "") or ""
    if "<think>" in comp and "</think>" not in comp:
        thinking_lens.append(len(comp))
    else:
        complete_lens.append(len(comp))
if thinking_lens:
    print(f"  Still thinking: avg={sum(thinking_lens)/len(thinking_lens):,.0f} chars, min={min(thinking_lens):,}, max={max(thinking_lens):,}")
if complete_lens:
    print(f"  Got to code:    avg={sum(complete_lens)/len(complete_lens):,.0f} chars, min={min(complete_lens):,}, max={max(complete_lens):,}")
