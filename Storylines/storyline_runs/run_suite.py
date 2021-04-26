"""
 Author: Matt Hanson
 Created: 27/04/2021 11:41 AM
 """
import subprocess
import sys

if __name__ == '__main__':
    paths = [  # todo check that this worked, also IBASAL, aslo manage the todos in these files related to ibasal
        r'C:\Users\dumon\python_projects\SLMACC-2020-CSRA\Storylines\storyline_runs\lauras_autum_drought_1yr.py',
        r'C:\Users\dumon\python_projects\SLMACC-2020-CSRA\Storylines\storyline_runs\lauras_v2_1yr.py',
        r'C:\Users\dumon\python_projects\SLMACC-2020-CSRA\Storylines\storyline_runs\historical_quantified_1yr.py',
        r'C:\Users\dumon\python_projects\SLMACC-2020-CSRA\Storylines\storyline_runs\run_unique_events.py',

    ]
    for p in paths:
        result = subprocess.run([sys.executable, p],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise ChildProcessError('{}\n{}'.format(result.stdout, result.stderr))
