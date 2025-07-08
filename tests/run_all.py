"""run all tests as once."""

#! IMPORTS

import sys
from os.path import join, dirname, abspath

sys.path += [join(dirname(abspath(__file__)), "single_tests")]

from single_tests import test_read_tdf

#! MAIN

if __name__ == "__main__":
    test_read_tdf.run_test()
