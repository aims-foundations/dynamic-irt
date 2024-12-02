import os
import shutil
import tempfile

from .utils import parallel_compile, parallel_run_executables


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
        """Formats the test cases into the required format for the grading engine.

        Returns:
            Tuple[List[Dict[str]], List[str]]: A tuple containing the formatted test cases and standard inputs.
        """
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
            if "std_in" not in testcase:
                std_inputs.append("")
            else:
                std_inputs.append(testcase["std_in"])
        return formatted_testcases, std_inputs

    def generate_code(self, student_answer):
        """Generates the C++ code with the student's answer and test cases.

        Args:
            student_answer (str): The student's answer to be inserted into the template.

        Returns:
            List[str]: A list of C++ code snippets with the student's answer and test cases inserted.
        """
        # Insert the student's answer and test cases into the template
        code = self.template.replace("{{ STUDENT_ANSWER }}", student_answer)

        # Find the for loop in the template
        start_index = code.find("{% for TEST in TESTCASES %}")
        end_index = code.find("{% endfor %}") + len("{% endfor %}")

        list_codes = []
        for testcase in self.formatted_testcases:
            # Insert the test case code into the template between the for loop
            testcode = code[:start_index] + testcase["testcode"] + code[end_index:]
            list_codes.append(testcode)

        return list_codes

    def write_and_compile_code(self, codes):
        """Writes and compiles the C++ code.

        Args:
            codes (List[str]): A list of C++ code snippets.

        Returns:
            Tuple[List[str], str]: A tuple containing the list of executable paths and the temporary directory.
        """
        # Write the C++ code to a temporary file
        temp_dir = tempfile.mkdtemp()
        for i, code in enumerate(codes):
            cpp_file = os.path.join(temp_dir, f"tc_{i}.cpp")
            with open(cpp_file, "w") as file:
                file.write(code)

        # Compile the C++ code
        executables = parallel_compile(codes, temp_dir, max_workers=self.max_workers)

        return executables, temp_dir

    def evaluate(self, student_answer):
        """Evaluates the student's answer using the test cases.

        Args:
            student_answer (str): The student's answer to be evaluated.

        Returns:
            Dict[str, Union[float, List[int]]]: A dictionary containing the score and test case results.
        """
        # Generate the C++ code with the student's answer
        codes = self.generate_code(student_answer)

        # Write and compile the C++ code
        executables, temp_dir = self.write_and_compile_code(codes)
        list_result = []

        executation_results = parallel_run_executables(
            executables, self.std_inputs, max_workers=self.max_workers
        )
        for i, testcase in enumerate(self.testcases):
            if executation_results[i][0] != 0:
                list_result.append(0)
                continue

            expected_output = testcase["output"]
            student_output = executation_results[i][1]
            if expected_output.strip() != student_output.strip():
                list_result.append(0)
            else:
                list_result.append(1)

        # Delete the temporary directory
        try:
            shutil.rmtree(temp_dir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        return_results = {
            "score": sum(list_result) / len(list_result),
            "testcases": list_result,
        }
        return return_results
