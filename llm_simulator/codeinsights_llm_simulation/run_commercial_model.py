import pandas as pd
import anthropic
import google.generativeai as genai
import openai
import time
import os
import logging
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from mistralai import Mistral
import functools

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set API keys from environment
os.environ['ANTHROPIC_API_KEY'] = 'APIKEY'
os.environ['GOOGLE_API_KEY'] = 'APIKEY'
os.environ['OPENAI_API_KEY'] = 'APIKEY'
os.environ["MISTRAL_API_KEY"] = 'APIKEY'

OUTPUT_FOLDER = "https://huggingface.co/datasets/Kazchoko/codeinsights_llm_simulation/resolve/main/"

@dataclass
class TestCase:
    """Structured representation of a test case."""
    input: str
    std_in: str
    output: str

@dataclass
class ProcessedQuestion:
    """Structured representation of a processed question."""
    question_id: str
    prompt: str
    test_cases: List[TestCase]
    student_id: Optional[str] = None

class TestCaseParser:
    """Utility class for parsing test cases from strings."""
    
    @staticmethod
    def parse_test_cases(test_string: str) -> List[TestCase]:
        """Parse test cases from unittest string."""
        test_cases = []
        
        for testcase_str in test_string.split("Unittest")[1:]:
            body = testcase_str[testcase_str.find(":") + 1:]
            
            input_idx = body.find("Input:")
            std_in_idx = body.find("STD input:")
            output_idx = body.find("Output:")
            
            if -1 in (input_idx, std_in_idx, output_idx):
                logger.warning("Failed to parse test case - missing required fields")
                return []  # Return empty list if parsing fails
            
            test_cases.append(TestCase(
                input=body[input_idx + 6:std_in_idx].strip(),
                std_in=body[std_in_idx + 10:output_idx].strip(),
                output=body[output_idx + 7:].strip()
            ))
        
        return test_cases

class LLMRunner:
    """Base class for LLM API runners with common functionality."""
    
    def __init__(self, api_key: str = None, delay: float = 1.0, max_workers: int = 3):
        self.api_key = api_key
        self.delay = delay
        self.max_workers = max_workers
    
    def run_parallel(self, prompt_list: List[str], **kwargs) -> List[str]:
        """Run prompts in parallel with rate limiting."""
        results = [""] * len(prompt_list)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_index = {
                executor.submit(self._single_request, prompt, **kwargs): i 
                for i, prompt in enumerate(prompt_list)
            }
            
            # Process completed futures
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                    logger.info(f"✓ Completed prompt {index + 1}/{len(prompt_list)}")
                except Exception as e:
                    logger.error(f"✗ Error on prompt {index + 1}: {e}")
                    results[index] = f"ERROR: {str(e)}"
                
                # Rate limiting
                time.sleep(self.delay / self.max_workers)
        
        return results
    
    def _single_request(self, prompt: str, **kwargs) -> str:
        """Override in subclasses for specific API calls."""
        raise NotImplementedError

class ClaudeRunner(LLMRunner):
    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-20250514", **kwargs):
        super().__init__(api_key, **kwargs)
        self.model = model
        if not self.api_key:
            self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Claude API key required")
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def _single_request(self, prompt: str, max_tokens: int = 4000, 
                       stop_sequences: List[str] = None) -> str:
        if stop_sequences is None:
            stop_sequences = ["\n```"]
            
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

class GeminiRunner(LLMRunner):
    def __init__(self, api_key: str = None, model_name: str = "gemini-1.5-flash", **kwargs):
        super().__init__(api_key, **kwargs)
        self.model_name = model_name
        if not self.api_key:
            self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key required")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def _single_request(self, prompt: str, max_output_tokens: int = 1024,
                       temperature: float = 0.0, stop_sequences: List[str] = None) -> str:
        config_params = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if stop_sequences:
            config_params["stop_sequences"] = stop_sequences
        
        response = self.model.generate_content(
            contents=[prompt],
            generation_config=genai.types.GenerationConfig(**config_params)
        )
        return response.text

class OpenAIRunner(LLMRunner):
    def __init__(self, api_key: str = None, model: str = "o1-mini", **kwargs):
        super().__init__(api_key, **kwargs)
        self.model = model
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        openai.api_key = self.api_key
    
    def _single_request(self, prompt: str, max_tokens: int = 4000,
                       temperature: float = 0.0, stop_sequences: List[str] = None) -> str:
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_sequences
        )
        return response.choices[0].message.content

class MistralRunner(LLMRunner):
    def __init__(self, api_key: str = None, model: str = "mistral-large-latest", **kwargs):
        super().__init__(api_key, **kwargs)
        self.model = model
        if not self.api_key:
            self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key required")
        self.client = Mistral(api_key=self.api_key)
    
    def _single_request(self, prompt: str, max_tokens: int = 4000,
                       temperature: float = 0.0, stop_sequences: List[str] = None) -> str:
        response = self.client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )
        
        text = response.choices[0].message.content
        
        # Manual trimming for stop sequences
        if stop_sequences:
            for seq in stop_sequences:
                idx = text.find(seq)
                if idx != -1:
                    text = text[:idx]
                    break
        
        return text

class PromptGenerator:
    """Centralized prompt generation for different scenarios."""
    
    @staticmethod
    def generate_base_cpp_prompt(question_name: str, question_text: str, 
                               question_template: str) -> str:
        """Generate base C++ coding prompt."""
        return (
            f"Question: {question_name} — {question_text}\n\n"
            "Template:\n"
            f"{question_template}\n\n"
            "Provide ONLY your C++ implementation that will replace the {{ STUDENT_ANSWER }} block in the template. "
            "– Do NOT reproduce any part of the template "
            "– Do NOT emit `int main()` (it's already declared) "
            "– Ensure your code is correct, efficient, handles all edge cases, and includes any needed class definitions "
            "IMPORTANT: "
            "Your entire response must be exactly one Markdown C++ code-block. "
            "1. The first line of your output must be: ```cpp "
            "2. The last line of your output must be: ``` "
            "3. No extra characters, whitespace, or text may appear before the opening ```cpp or after the closing ```."
        )
    
    @staticmethod
    def generate_student_profile_prompt(student_id: str, topic_performance: pd.DataFrame) -> str:
        """Generate student profile section."""
        prompt = f"Student {student_id} has the following performance across topics:\n"
        for _, row in topic_performance.iterrows():
            prompt += (
                f"- For topic '{row['topic']}', the unit test pass rate is "
                f"{row['pass_rate']:.2f}, and the rate of passing all tests is {row['perfect']:.2f}.\n"
            )
        return prompt

class DataProcessor:
    """Handles data loading and processing operations."""
    
    def __init__(self):
        self.base_url = "https://huggingface.co/datasets/Kazchoko/codeinsights_llm_simulation/resolve/main/"
        self._cache = {}
    
    @functools.lru_cache(maxsize=None)
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load and cache data files."""
        if filename not in self._cache:
            url = f"{self.base_url}{filename}"
            logger.info(f"Loading data from {url}")
            self._cache[filename] = pd.read_csv(url)
        return self._cache[filename]
    
    def process_scenario_1(self) -> List[ProcessedQuestion]:
        """Process scenario 1 data efficiently."""
        df = self.load_data("Scenario1_full_data.csv")
        processed = []
        
        for question_id, question_df in df.groupby("question_unittest_id"):
            target = question_df.iloc[0]
            
            # Parse test cases
            test_cases = TestCaseParser.parse_test_cases(target["question_unittests"])
            if not test_cases:
                continue
            
            prompt = PromptGenerator.generate_base_cpp_prompt(
                target['question_name'], 
                target['question_text'], 
                target['question_template']
            )
            
            processed.append(ProcessedQuestion(
                question_id=question_id,
                prompt=prompt,
                test_cases=test_cases
            ))
        
        return processed
    
    def process_scenario_2_3_4(self, scenario: str, prompt_func) -> List[ProcessedQuestion]:
        """Process scenario 2 with student profiling."""
        df = self.load_data(f"Scenario{scenario}_full_data.csv")
        student_topic = self.load_data("student_performace_by_topic.csv") 
        processed = []
        
        for student_id, student_df in df.groupby("student_id"):
            student_df = student_df.sort_values("timestamp")
            if len(student_df) < 4:
                continue
            
            attempts = student_df.iloc[:4]
            topic_performance = student_topic[student_topic["student_id"] == student_id]
            student_profile = PromptGenerator.generate_student_profile_prompt(student_id, topic_performance)
            
            # Process each attempt as target
            for idx in range(4):
                target = attempts.iloc[idx]
                examples = [attempts.iloc[i] for i in range(4) if i != idx]
                
                test_cases = TestCaseParser.parse_test_cases(target["question_unittests"])
                if not test_cases:
                    continue
                
                prompt = self.prompt_func(student_profile, target, examples)
                
                processed.append(ProcessedQuestion(
                    question_id=target.get("question_unittest_id"),
                    prompt=prompt,
                    test_cases=test_cases,
                    student_id=student_id
                ))
        
        return processed
    
    def _build_scenario_2_prompt(self, student_profile: str, target: pd.Series, examples: List[pd.Series]) -> str:
        """Build scenario 2 prompt with examples."""
        prompt = (
            "=== Student Profile ===\n"
            f"{student_profile}\n"
            f"Week: {target['week']}\n"
            f"Topic: {target['topic']}\n\n"
        )
        
        for n, ex in enumerate(examples, start=1):
            prompt += (
                f"Example {n}:\n"
                f"Question: {ex['question_name']} — {ex['question_text']}\n"
                f"Template:\n{ex['question_template']}\n"
                f"Your Code:\n{ex['response']}\n\n"
            )
        
        prompt += (
            "Now, using that same student style, attempt this:\n"
            f"Question: {target['question_name']} — {target['question_text']}\n"
            f"Template:\n{target['question_template']}\n\n"
            "Provide ONLY your C++ implementation that will replace the {{ STUDENT_ANSWER }} block in the template. "
            "– Do NOT reproduce any part of the template "
            "– Do NOT emit `int main()` (it's already declared) "
            "– Ensure your code mirrors the style of the previous examples and includes any necessary class definitions "
            "IMPORTANT: your entire response must be exactly one Markdown C++ code‐block:\n"
            "1. First line: ```cpp\n"
            "2. Last line: ```\n"
        )
        
        return prompt

    def _build_scenario_3_prompt(self, student_profile: str, target: pd.Series, examples: List[pd.Series]) -> str:
        """Build scenario 3 prompt with examples."""
        prompt = (
                    "=== Student Profile ===\n"
                    f"{student_level_prompt}\n"
                    "When students submit a code to the platform, it will be tested by number of unit tests, where\n"
                    "- Unit test pass rate = proportion of unit tests passed with the code\n"
                    "- Full pass rate   = proportion of code passing all unit tests\n\n"
                    "=== Past Mistake Examples ===\n"
        )
        for n, ex in enumerate(examples, start=1):
            prompt += (
                        f"Example {n} (Week {ex['week']}, Topic: {ex['topic']}):\n"
                        f"Question: {ex['question_name']} — {ex['question_text']}\n"
                        "Template:\n"
                        f"{ex['question_template']}\n"
                        "Student's Response Code with Error:\n"
                        f"{ex['response_mistake']}\n\n"
            )
        
        prompt += (
                    "=== New Target Problem ===\n"
                    f"Week: {target['week']}, Topic: {target['topic']}\n"
                    f"Question: {target['question_name']} — {target['question_text']}\n"
                    + "Template:\n"
                    f"{target['question_template']}\n\n"
                    "⚠**Instructions:**\n"
                    "1. Mimic your own coding style, naming conventions, indentation, and typical error patterns from the examples.\n"
                    "2. Introduce a mistake you are likely to make (e.g., off‑by‑one index, wrong initialization, missing edge case).\n"
                    "3. Do **not** produce a fully correct solution or add unfamiliar optimizations.\n\n"
                    "4. Include any needed class definitions, and make sure the code is compatible with the Unit Test Input.\n"
                    "5. Provide ONLY your C++ implementation that will replace the {{ STUDENT_ANSWER }} block in the template.\n"
                    "6. Do NOT reproduce any part of the template.\n"
                    "7. Do NOT emit `int main()` (it’s already declared).\n\n"
                    "IMPORTANT: your entire response must be exactly one Markdown C++ code‑block:\n"
                    "1. First line: ```cpp\n"
                    "2. Last line: ```\n"
                    "No extra characters, whitespace, or text before/after.\n")
        
        return prompt

    def _build_scenario_4_prompt(self, student_profile: str, target: pd.Series, examples: List[pd.Series]) -> str:
        """Build scenario 4 prompt with examples."""
        prompt = (
                    f"Week: {target['week']}\n"
                    f"Topic: {target['topic']}\n\n"
        )
        for n, ex in enumerate(examples, start=1):
            prompt += (
                        f"Example {n}:\n"
                        f"Question: {ex['question_name']} — {ex['question_text']}\n"
                        "Template:\n"
                        f"{ex['question_template']}\n"
                        "Your Code:\n"
                        f"{ex['response']}\n\n"
            )
        prompt += (
                    "Now, using that same student's coding style, attempt this:\n"
                    f"Question: {target['question_name']} — {target['question_text']}\n\n"
                    + "Template:\n"
                    f"{target['question_template']}\n\n"
                    "Provide ONLY your C++ implementation that will replace the {{ STUDENT_ANSWER }} block in the template.  "
                    "– Do NOT reproduce any part of the template  "
                    "– Do NOT emit `int main()` (it’s already declared)  "
                    "– Ensure your code is correct, handles all edge cases, and includes any needed class definitions  "
                    "– Match the student’s usual efficiency style.\n\n"
                    "IMPORTANT: your entire response must be exactly one Markdown C++ code‑block:\n"
                    "1. First line: ```cpp\n"
                    "2. Last line: ```\n"
                    "No extra whitespace or text before/after.\n"
        )

def run_scenario_with_all_models(scenario_name: str, prompts: List[str], 
                               question_ids: List[str], student_ids: List[str] = None) -> Dict[str, pd.DataFrame]:
    """Run a scenario with all available models and return results."""
    
    # Initialize all model runners
    model_runners = {
        'claude': ClaudeRunner(max_workers=2, delay=1.0),  # More conservative for Claude
        'gemini': GeminiRunner(max_workers=3, delay=0.5),
        'openai': OpenAIRunner(max_workers=2, delay=0.8),  # Conservative for OpenAI
        'mistral': MistralRunner(max_workers=3, delay=0.5)
    }
    
    results = {}
    
    for model_name, runner in model_runners.items():
        try:
            logger.info(f"Running {scenario_name} with {model_name.upper()}...")
            
            # Run the model on all prompts
            model_results = runner.run_parallel(prompts)
            
            # Create DataFrame based on scenario type
            if student_ids is not None:  # Scenarios 2, 3, 4 have student_ids
                df = pd.DataFrame({
                    "student_id": student_ids,
                    "question_id": question_ids,
                    "text": model_results,
                    "model": model_name
                })
            else:  # Scenario 1 only has question_ids
                df = pd.DataFrame({
                    "question_id": question_ids,
                    "text": model_results,
                    "model": model_name
                })
            
            results[model_name] = df
            logger.info(f"✓ {model_name.upper()} completed: {len(model_results)} results")
            
        except Exception as e:
            logger.error(f"✗ Failed to run {model_name.upper()}: {e}")
            # Create empty DataFrame with same structure for failed models
            if student_ids is not None:
                results[model_name] = pd.DataFrame({
                    "student_id": student_ids,
                    "question_id": question_ids,
                    "text": [f"ERROR: {str(e)}"] * len(prompts),
                    "model": model_name
                })
            else:
                results[model_name] = pd.DataFrame({
                    "question_id": question_ids,
                    "text": [f"ERROR: {str(e)}"] * len(prompts),
                    "model": model_name
                })
    
    return results

def main():
    """Main execution function - runs all scenarios with all models."""
    try:
        # Initialize data processor
        processor = DataProcessor()
        all_results = {}
        
        # Process Scenario 1
        logger.info("=" * 50)
        logger.info("PROCESSING SCENARIO 1")
        logger.info("=" * 50)
        
        scenario1_data = processor.process_scenario_1()
        scenario1_prompts = [item.prompt for item in scenario1_data]
        scenario1_ids = [item.question_id for item in scenario1_data]
        
        scenario1_results = run_scenario_with_all_models(
            "Scenario 1", 
            scenario1_prompts, 
            scenario1_ids
        )
        all_results['scenario1'] = scenario1_results
        
        # Process Scenario 2
        logger.info("=" * 50)
        logger.info("PROCESSING SCENARIO 2")
        logger.info("=" * 50)
        
        scenario2_data = processor.process_scenario_2_3_4("2", _build_scenario_2_prompt)
        scenario2_prompts = [item.prompt for item in scenario2_data]
        scenario2_question_ids = [item.question_id for item in scenario2_data]
        scenario2_student_ids = [item.student_id for item in scenario2_data]
        
        scenario2_results = run_scenario_with_all_models(
            "Scenario 2",
            scenario2_prompts,
            scenario2_question_ids,
            scenario2_student_ids
        )
        all_results['scenario2'] = scenario2_results

        # Process Scenario 3
        logger.info("=" * 50)
        logger.info("PROCESSING SCENARIO 3")
        logger.info("=" * 50)
        
        scenario3_data = processor.process_scenario_2_3_4("3", _build_scenario_3_prompt)
        scenario3_prompts = [item.prompt for item in scenario3_data]
        scenario3_question_ids = [item.question_id for item in scenario3_data]
        scenario3_student_ids = [item.student_id for item in scenario3_data]
        
        scenario3_results = run_scenario_with_all_models(
            "Scenario 3",
            scenario3_prompts,
            scenario3_question_ids,
            scenario3_student_ids
        )
        all_results['scenario3'] = scenario3_results

        # Process Scenario 4
        logger.info("=" * 50)
        logger.info("PROCESSING SCENARIO 4")
        logger.info("=" * 50)
        
        scenario4_data = processor.process_scenario_2_3_4("4", _build_scenario_4_prompt)
        scenario4_prompts = [item.prompt for item in scenario4_data]
        scenario4_question_ids = [item.question_id for item in scenario4_data]
        scenario4_student_ids = [item.student_id for item in scenario4_data]
        
        scenario4_results = run_scenario_with_all_models(
            "Scenario 4",
            scenario4_prompts,
            scenario4_question_ids,
            scenario4_student_ids
        )
        all_results['scenario4'] = scenario4_results
        
        # Print summary
        logger.info("=" * 50)
        logger.info("EXECUTION SUMMARY")
        logger.info("=" * 50)
        
        for scenario_name, scenario_results in all_results.items():
            logger.info(f"\n{scenario_name.upper()}:")
            for model_name, df in scenario_results.items():
                success_count = len(df[~df['text'].str.startswith('ERROR:')])
                total_count = len(df)
                logger.info(f"  {model_name.upper()}: {success_count}/{total_count} successful")
        
        return all_results
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    all_results = main()
    print("Processing complete!")
    print(f"Results available for {len(all_results)} scenarios")
    for scenario_name in all_results.keys():
        print(f"  - {scenario_name}: {len(all_results[scenario_name])} models")
        
    # Save csv
    for scenario_name, scenario_results in all_results.items():
         for model_name, df in scenario_results.items():
            target_dir = os.path.join(OUTPUT_FOLDER, "scenario_results", model_name)
            os.makedirs(target_dir, exist_ok=True)
            out_path = os.path.join(
                target_dir,
                f"{model_name}_{scenario_name}.csv"
            )
            df.to_csv(out_path, index=False)