import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from grading_engine import CPPEvaluator


def testcase1(evaluator):
    student_answer = """int buyCar(int* nums, int length, int k) { sort(nums, nums + length); int i = 0, cnt = 0; while(k > 0 && i < length){ k = k - nums[i]; cnt++; i++; } return cnt; }"""
    result = evaluator.evaluate(student_answer)
    print("Real results:", 0.2)
    print("Autograding:", result)
    assert result["score"] == 0.2


def testcase2(evaluator):
    student_answer = """int buyCar(int* nums, int length, int k) { sort(nums, nums + length); int i = 0, cnt = 0; while(k - nums[i] >= 0 && i < length){ k = k - nums[i]; cnt++; i++; } return cnt; }"""
    result = evaluator.evaluate(student_answer)
    print("Real results:", 0.9)
    print("Autograding:", result)
    assert result["score"] == 0.9


def testcase3(evaluator):
    student_answer = """int buyCar(int* nums, int length, int k) { int x; cin >> x; cout << x; sort(nums, nums + length); int i = 0, cnt = 0; while(k - nums[i] >= 0 && i < length){ k = k - nums[i]; cnt++; i++; } return cnt; }"""
    result = evaluator.evaluate(student_answer)
    print("Real results:", 0.1)
    print("Autograding:", result)
    assert result["score"] == 0.1


if __name__ == "__main__":
    # Example usage:
    template = """
    #include <iostream>
    #include <algorithm>
    using namespace std;
    #define SEPARATOR "#<ab@17943918#@>"
    {{ STUDENT_ANSWER }}
    int main() {
        {% for TEST in TESTCASES %}
        {
            {{ TEST.extra }};
            {{ TEST.testcode }};
        }
        {% if not loop.last %}
        cout << SEPARATOR << endl;
        {% endif %}
        {% endfor %}
        return 0;
    }
    """

    testcases = [
        {
            "input": r'int nums[] = {90,30,40,90,20}; int length = sizeof(nums)/sizeof(nums[0]); cout << buyCar(nums, length, 90) << "\n";',
            "output": "3",
        },
        {
            "input": r'int nums[] = {90,30,40,90,20,70,150,300}; int length = sizeof(nums)/sizeof(nums[0]); cout << buyCar(nums, length, 200) << "\n";',
            "std_in": "5",
            "output": "54",
        },
        {
            "input": r'int nums[] = {80,120,150,30,500,260,170,200,50}; int length = sizeof(nums)/sizeof(nums[0]); cout << buyCar(nums, length, 500) << "\n";',
            "output": "5",
        },
        {
            "input": r'int nums[] = {150,140,130,120,110,100,90,80,70,60}; int length = sizeof(nums)/sizeof(nums[0]); cout << buyCar(nums, length, 120) << "\n";',
            "output": "1",
        },
        {
            "input": r'int nums[] = {50,90,180,300,52,46,285,78,42,966,135,545,858,47,124}; int length = sizeof(nums)/sizeof(nums[0]); cout << buyCar(nums, length, 1000) << "\n";',
            "output": "10",
        },
        {
            "input": r'int nums[] = {50,90,180,300,52,46,285,78,42,966,135,545,858,47,124}; int length = sizeof(nums)/sizeof(nums[0]); cout << buyCar(nums, length, 10) << "\n";',
            "output": "0",
        },
        {
            "input": r'int nums[] = {52,123,465,85,494,71,58,123,64,824,712,64,85,741,123}; int length = sizeof(nums)/sizeof(nums[0]); cout << buyCar(nums, length, 800) << "\n";',
            "output": "9",
        },
        {
            "input": r'int nums[] = {2302, 2803, 142,122,256,157,523,148,125,444,505,1285}; int length = sizeof(nums)/sizeof(nums[0]); cout << buyCar(nums, length, 2702) << "\n";',
            "output": "9",
        },
        {
            "input": r'int nums[] = {10,50,90,140,20,30,50,40,50,10,20,30,35,80,50,90,120,110,120,130,200,210,24,250,150,180,182}; int length = sizeof(nums)/sizeof(nums[0]); cout << buyCar(nums, length, 2702) << "\n";',
            "output": "27",
        },
        {
            "input": r'int nums[] = {10,50,90,140,20,30,50,40,50,10,20,30,35,80,50,90,120,110,120,130,200,210,24,250,150,180,182}; int length = sizeof(nums)/sizeof(nums[0]); cout << buyCar(nums, length, 5) << "\n";',
            "output": "0",
        },
    ]

    evaluator = CPPEvaluator(template, testcases)

    testcase1(evaluator)
    testcase2(evaluator)
    testcase3(evaluator)
