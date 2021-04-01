"""
 Author: Matt Hanson
 Created: 1/04/2021 5:55 PM
 """
import os
import sys
import timeit


def timeit_test(py_file_path, function_names=('test_function', 'test_function2'), n=10):
    """
    run an automated timeit test, must be outside of the function definition, prints results in scientific notation
    units are seconds

    :param py_file_path: path to the python file that holds the functions,
                        if the functions are in the same script as call then  __file__ is sufficient.
                        in this case the function call should be protected by: if __name__ == '__main__':
    :param function_names: the names of the functions to test (iterable), functions must not have arguments
    :param n: number of times to test
    :return:
    """
    print(py_file_path)
    d = os.path.dirname(py_file_path)
    fname = os.path.basename(py_file_path).replace('.py', '')
    sys.path.append(d)

    out = {}
    for fn in function_names:
        print('testing: {}'.format(fn))
        t = timeit.timeit('{}()'.format(fn),
                          setup='from {} import {}'.format(fname, fn), number=n) / n
        out[fn] = t
        print('{0:e} seconds'.format(t))
    return out


if __name__ == '__main__':
    timeit_test(r'C:\Users\dumon\python_projects\SLMACC-2020-CSRA\Storylines\storyline_runs\run_random_suite.py',
                ('make_1_year_storylines',  # 16 stories 0.366s
                 # 'run_1year_basgra',  # 16 stories (full logical on dickie), 100 reals of 1 yr sim: 89.17002 seconds
                 'create_1y_pg_data',  # 16 stories 0.197s
                 ), n=100)  # 90s per 16 stories.
