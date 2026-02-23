import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor


def compile_code(i, temp_dir):
    """
    Compiles the C++ file at temp_dir/tc_{i}.cpp and outputs to temp_dir/tc_{i}.out.

    Args:
        i (int): Index of the code to compile.
        temp_dir (str): Temporary directory where the C++ files are located.

    Returns:
        str or None: Path to the executable if compilation succeeds, else None.
    """
    executable = os.path.join(temp_dir, f"tc_{i}.out")
    cpp_file = os.path.join(temp_dir, f"tc_{i}.cpp")

    try:
        result = subprocess.run(
            ["g++", "-std=c++11", cpp_file, "-o", executable],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            return None
        return executable
    except Exception:
        return None


def parallel_compile(codes, temp_dir, max_workers=4):
    """
    Compiles multiple C++ codes in parallel.

    Args:
        codes (list): List of code snippets or identifiers.
        temp_dir (str): Directory containing the C++ files.
        max_workers (int): Maximum number of worker processes.

    Returns:
        list: List of paths to the compiled executables or None for failed compilations.
    """
    executables = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compile_code, i, temp_dir) for i in range(len(codes))]
        for future in futures:
            executables.append(future.result())
    return executables


def run_executable(executable, std_in, timeout=10):
    """
    Runs an executable with a timeout and captures its output.

    Args:
        executable (str): Path to the executable to run.
        std_in (str): Standard input to pass to the executable.
        timeout (int): Timeout for running the executable in seconds.

    Returns:
        tuple: (return_code, stdout)
    """
    if executable is None:
        return (1, "")

    try:
        result = subprocess.run(
            ["timeout", str(timeout), executable],
            input=std_in,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return (result.returncode, result.stdout)
    except Exception:
        return (1, "")


def parallel_run_executables(executables, std_inputs, max_workers=4):
    """
    Runs multiple executables in parallel with a timeout.

    Args:
        executables (list): List of paths to the executables.
        std_inputs (list): Standard inputs for each executable.
        max_workers (int): Maximum number of worker processes.

    Returns:
        list: List of (return_code, stdout) tuples.
    """
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_executable, executable, std_in)
            for std_in, executable in zip(std_inputs, executables)
        ]
        for future in futures:
            results.append(future.result())
    return results


class CPPEvaluator:
    def __init__(self, template, testcases, max_workers=8):
        """Initializes the CPPEvaluator class.

        Args:
            template (str): The template code with placeholders for the student's answer and test cases.
            testcases (Dict[str]): A list of test cases, each containing the input, output, and optional std_in.
            max_workers (int, optional): The maximum number of workers to use for parallel processing. Defaults to 8.
        """
        self.template = template
        self.testcases = testcases
        self.max_workers = max_workers
        self.formatted_testcases, self.std_inputs = self.format_testcases()

    def format_testcases(self):
        formatted_testcases = []
        std_inputs = []
        for testcase in self.testcases:
            formatted_testcases.append(
                {
                    "extra": "",
                    "testcode": testcase["input"],
                    "expected_output": testcase["output"],
                }
            )
            std_inputs.append(testcase.get("std_in", ""))
        return formatted_testcases, std_inputs

    def generate_code(self, student_answer):
        code = self.template.replace("{{ STUDENT_ANSWER }}", student_answer)

        start_index = code.find("{% for TEST in TESTCASES %}")
        end_index = code.find("{% endfor %}") + len("{% endfor %}")

        list_codes = []
        for testcase in self.formatted_testcases:
            testcode = code[:start_index] + testcase["testcode"] + code[end_index:]
            list_codes.append(testcode)

        return list_codes

    def write_and_compile_code(self, codes):
        temp_dir = tempfile.mkdtemp()
        for i, code in enumerate(codes):
            cpp_file = os.path.join(temp_dir, f"tc_{i}.cpp")
            with open(cpp_file, "w") as file:
                file.write(code)

        executables = parallel_compile(codes, temp_dir, max_workers=self.max_workers)
        return executables, temp_dir

    def evaluate_with_outputs(self, student_answer):
        """Like evaluate(), but also returns the actual stdout for each test case.

        Used by the iterative runner to build informative feedback prompts.

        Returns:
            Dict with 'score' (float), 'testcases' (List[int]), and
            'outputs' (List[str] — actual stdout per test, empty string on error).
        """
        codes = self.generate_code(student_answer)
        executables, temp_dir = self.write_and_compile_code(codes)

        list_result = []
        outputs = []

        execution_results = parallel_run_executables(
            executables, self.std_inputs, max_workers=self.max_workers
        )
        for i, testcase in enumerate(self.testcases):
            # None executable means compilation failed
            if executables[i] is None or execution_results[i][0] != 0:
                list_result.append(0)
                outputs.append("")
                continue

            expected_output = testcase["output"]
            student_output = execution_results[i][1]
            outputs.append(student_output)

            if expected_output.strip() != student_output.strip():
                list_result.append(0)
            else:
                list_result.append(1)

        try:
            shutil.rmtree(temp_dir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        if len(list_result) == 0:
            return {"score": 0, "testcases": list_result, "outputs": outputs}

        return {
            "score": sum(list_result) / len(list_result),
            "testcases": list_result,
            "outputs": outputs,
        }

    def evaluate(self, student_answer):
        """Evaluates the student's answer using the test cases.

        Returns:
            Dict with 'score' (float) and 'testcases' (List[int]).
        """
        codes = self.generate_code(student_answer)
        executables, temp_dir = self.write_and_compile_code(codes)
        list_result = []

        execution_results = parallel_run_executables(
            executables, self.std_inputs, max_workers=self.max_workers
        )
        for i, testcase in enumerate(self.testcases):
            if executables[i] is None or execution_results[i][0] != 0:
                list_result.append(0)
                continue

            expected_output = testcase["output"]
            student_output = execution_results[i][1]
            if expected_output.strip() != student_output.strip():
                list_result.append(0)
            else:
                list_result.append(1)

        try:
            shutil.rmtree(temp_dir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        if len(list_result) == 0:
            return {"score": 0, "testcases": list_result}

        return {
            "score": sum(list_result) / len(list_result),
            "testcases": list_result,
        }
