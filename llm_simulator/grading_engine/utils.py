import os
import subprocess
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
            text=True,  # Optional: to get output as string
        )
        if result.returncode != 0:
            # print(f"Compilation failed for {cpp_file}:\n{result.stderr}")
            return None
        return executable
    except Exception as e:
        # print(f"An error occurred while compiling {cpp_file}: {e}")
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
        # Submit all compilation tasks
        futures = [
            executor.submit(compile_code, i, temp_dir) for i in range(len(codes))
        ]

        # Retrieve results as they complete
        for future in futures:
            result = future.result()
            executables.append(result)

    return executables


def run_executable(executable, std_in, timeout=10):
    """
    Runs an executable with a timeout and captures its output.

    Args:
        executable (str): Path to the executable to run.
        timeout (int): Timeout for running the executable in seconds.

    Returns:
        tuple: (return_code, output) where return_code is 0 if successful, non-zero otherwise,
               and output is the stdout captured from the execution.
    """
    if executable is None:
        return (0, "")  # Return 0 and empty output for failed compilations

    try:
        result = subprocess.run(
            ["timeout", str(timeout), executable],
            input=std_in,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # To decode stdout and stderr as strings
        )
        return (result.returncode, result.stdout)
    except Exception as e:
        # print(f"An error occurred while running {executable}: {e}")
        return (1, "")  # Non-zero return code for errors


# Example usage
def parallel_run_executables(executables, std_inputs, max_workers=4):
    """
    Runs multiple executables in parallel with a timeout.

    Args:
        executables (list): List of paths to the executables.
        max_workers (int): Maximum number of worker processes.

    Returns:
        list: List of results containing the outputs from running each executable.
    """
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all executable running tasks
        futures = [
            executor.submit(run_executable, executable, std_in)
            for std_in, executable in zip(std_inputs, executables)
        ]

        # Retrieve results as they complete
        for future in futures:
            result_code, output = future.result()
            results.append((result_code, output))

    return results
