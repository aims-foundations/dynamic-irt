#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains configurations for crawling pipeline.
"""

LOGIN_URL = (
    "https://sso.hcmut.edu.vn/cas/login?service="
    "https%3A%2F%2Fe-learning.hcmut.edu.vn%2Flogin%2Findex.php%3FauthCAS%3DCAS"
)
LOGIN_USER = "010344"
LOGIN_PASSWD = "010344"

DATA_LINKS = {
    "DSA-HK231": {
        "L09": "https://e-learning.hcmut.edu.vn/course/view.php?id=106649",
        "DT01": "https://e-learning.hcmut.edu.vn/course/view.php?id=108885",
    }
}
