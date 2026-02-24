"""
C++ code evaluator for testing generated code against test cases.
"""

import os
import platform
import shutil
import subprocess
import tempfile
from typing import Dict, List, Union

# Use gtimeout on macOS, timeout on Linux
TIMEOUT_CMD = "gtimeout" if platform.system() == "Darwin" else "timeout"


class CPPEvaluator:
    """Compiles and runs C++ code against test cases."""

    def __init__(self, template: str, testcases: List[Dict], timeout: int = 10):
        self.template = template
        self.testcases = testcases
        self.timeout = timeout
        self.formatted_testcases, self.std_inputs = self._format_testcases()

    def _format_testcases(self):
        formatted = []
        std_inputs = []
        for tc in self.testcases:
            formatted.append({"testcode": tc["input"], "expected_output": tc["output"]})
            std_inputs.append(tc.get("std_in", ""))
        return formatted, std_inputs

    def _compile(self, cpp_file: str, executable: str) -> bool:
        try:
            result = subprocess.run(
                [TIMEOUT_CMD, str(self.timeout), "g++", "-std=c++11", cpp_file, "-o", executable],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout + 2,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _run(self, executable: str, std_in: str) -> tuple:
        try:
            result = subprocess.run(
                [TIMEOUT_CMD, str(self.timeout), executable],
                input=std_in,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout + 2,
            )
            return result.returncode, result.stdout
        except Exception:
            return 1, ""

    def evaluate(self, student_answer: str) -> Dict[str, Union[float, List[int]]]:
        """Run test cases and return score and per-test results."""
        code = self.template.replace("{{ STUDENT_ANSWER }}", student_answer)
        start = code.find("{% for TEST in TESTCASES %}")
        end = code.find("{% endfor %}") + len("{% endfor %}")

        temp_dir = tempfile.mkdtemp()
        results = []

        try:
            for i, tc in enumerate(self.formatted_testcases):
                test_code = code[:start] + tc["testcode"] + code[end:]
                cpp_file = os.path.join(temp_dir, f"tc_{i}.cpp")
                executable = os.path.join(temp_dir, f"tc_{i}.out")

                with open(cpp_file, "w") as f:
                    f.write(test_code)

                if not self._compile(cpp_file, executable):
                    results.append(0)
                    continue

                returncode, output = self._run(executable, self.std_inputs[i])
                if returncode != 0:
                    results.append(0)
                elif output.strip() == self.testcases[i]["output"].strip():
                    results.append(1)
                else:
                    results.append(0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        score = sum(results) / len(results) if results else 0
        return {"score": score, "testcases": results}
