"""
This module contains utility functions.
"""


def parse_score(name):
    """
    This function is used to parse student score from crawled text.
    """
    return float(name.split("\n")[1][1:])
