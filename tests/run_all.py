"""run all tests as once."""

#! IMPORTS

import sys
from os.path import join, dirname, abspath

sys.path += [join(dirname(abspath(__file__)))]

from single_tests import *

#! MAIN

if __name__ == "__main__":
    test_read_tdf.run_test()
    test_runtest.run_test()
