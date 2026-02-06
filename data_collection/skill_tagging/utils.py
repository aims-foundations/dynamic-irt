GRADE_MAP = {
    "Kindergarten": "kindergarten",
    "First": "1",
    "Second": "2",
    "Third": "3",
    "Fourth": "4",
    "Fifth": "5",
    "Sixth": "6",
    "Seventh": "7",
    "Eighth": "8",
    "Ninth": "9",
    "Tenth": "10",
    "Eleventh": "11",
    "Twelfth": "12",
    "High school": "10, 11, and 12",
}

REVERSE_GRADE_MAP = {v: k for k, v in GRADE_MAP.items()}


PROMPT_SUBJECT = """You are a quality assurance specialist working in education.
You are tasked with tagging the subjects that the following question belongs to.
The list of subjects is:
(1) Mathematics, (2) Language & Arts, (3) Science, and (4) Social Studies.
Please tag the subjects of the following question and return the subjects in a Python list, and nothing else.
Example: ['Mathematics', 'Science'].
Question: {question}"""
LIST_SUBJETCS = ["Mathematics", "Language & Arts", "Science", "Social Studies"]

PROMPT_GRADE = """You are a quality assurance specialist working in education.
You are tasked with tagging the grade levels that the following question belongs to.
The question must belong to at least one of the following grade levels.
The possible grade levels are {grade_desc}.
Please tag the grade levels of the following question and return the grade levels in a Python list, and nothing else.
Example: ['1', '2'].
Question: {question}"""

PROMPT_LV = """You are a quality assurance specialist working in education.
You are tasked with tagging the skills that the following question tests.
The question must have at least one of the following skills.
The list of skills is:
{desc}.
Please tag the skills of the following question and return the tags in a Python list, and nothing else.
Example: ['1', '2'].
Question: {question}"""

PROMPT_ELEMENT = """You are a quality assurance specialist working in education.
You are tasked with tagging the skills that the following question tests.
The list of skills is:
{desc}.
Please tag the skills of the following question and return the skills in a Python list, and nothing else.
Example: ['1', '2'].
Question: {question}"""
