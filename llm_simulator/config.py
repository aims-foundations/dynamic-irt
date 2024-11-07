START_WEEK = 2
MAX_QUES_PER_WEEK = 5
MAX_TRIALS_PER_QUES = 3

SYSTEM_PROMPT = "You are a student of a programming course."
QUESTION_INST = "Here are the exercise questions for practice."
EXAM_INST = "Here are the exam questions."
RESPONSE_INST = "\nPlease write your answer to the above question in C++. If you retry a question, only return the modifications to the prior answer."
FEEDBACK_INST = "Your score: {score}/1."
FIRST_ATTEMPT_PREFIX = "First answer for question "
RETRY_PREFIX = "Editted answer for question "

RETRY_START_TAG = "REPLACE"
RETRY_SEPARATOR = "WITH"
RETRY_END_TAG = "END"

STOP_ANSWER = ["<|eot_id|>", "Your score"]
USER_TAG = "<|start_header_id|>user<|end_header_id|>\n\n"


WEEK_FILES = {
    "dsa_hk231": {
        0: [
            [
                "W1_OOP_Review.json",
                "W1_Recursion.json",
                "W1_Array_List.json",
                "W1_Singly_Linked_List.json",
            ],
            [
                "W2_Doubly_Linked_List.json",
                "W2_Stack.json",
                "W2_Queue.json",
                "W2_Sorting_(Easy).json",
            ],
            [
                "W3_Sorting_(Advance).json",
                "W3_Binary_Tree.json",
                "W3_Binary_Search_Tree.json",
            ],
            ["W4_AVL_Tree.json", "W4_B-Tree.json", "W4_Splay_Tree.json"],
            ["W5_Heap.json", "W5_Search.json"],
            ["W6_Graph.json", "W6_Hash.json"],
        ],
        1: [
            [
                "W1_All.json",
            ],
            [
                "W2_All.json",
            ],
            [
                "W3_All.json",
            ],
            [
                "W4_All.json",
            ],
            [
                "W5_All.json",
            ],
            [
                "W6_All.json",
            ],
        ],
        2: [
            [
                "W1_OOP_Review.json",
                "W1_Recursion.json",
                "W1_Array_List.json",
                "W1_Singly_Linked_List.json",
            ],
            [
                "W2_Doubly_Linked_List.json",
                "W2_Stack.json",
                "W2_Queue.json",
                "W2_Sorting_(Easy).json",
            ],
            [
                "W3_Sorting_(Advance).json",
                "W3_Binary_Tree.json",
                "W3_Binary_Search_Tree.json",
            ],
            ["W4_AVL_Tree.json", "W4_B-Tree.json", "W4_Splay_Tree.json"],
            ["W5_All.json"],
            ["W6_All.json"],
        ],
        "exam": [
            "W2_Exam.json",
            "W3_Exam.json",
            "W4_Exam.json",
            "W5_Exam.json",
            "W6_Exam.json",
        ],
    }
}

CLASSES = {
    "dsa_hk231": [
        "CC01",
        # "CC02",
        "CC03",
        # # "CC05",
        # # "CC07",
        # # "CC08",
        "CN02",
        "DT01",
        "L02",
        # # "L03",
        "L04",
        # # "L05",
        "L06",
        "L07",
        "L08",
        "L09",
        # "L10",
        # "L13",
        # "L14"
    ]
    # CN01, CN02, CC04, CC05, CC06, L01, L03, L11, L12, L14, L15
}
