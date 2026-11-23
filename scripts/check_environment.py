"""Report every Phase 1 prerequisite, returning nonzero on missing items."""
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import shutil
import subprocess


def main():
    missing = []
    print('Python', platform.python_version())
    home = os.environ.get('SUMO_HOME')
    print('SUMO_HOME', home)
    if not home or not (Path(home) / 'tools').is_dir():
        missing.append('SUMO_HOME/tools')
    for command in ('sumo', 'sumo-gui', 'netconvert'):
        if shutil.which(command):
            print(subprocess.check_output([command, '--version'], text=True).splitlines()[0])
        else:
            missing.append(command)
    for package in ('sumo-rl', 'gymnasium', 'pettingzoo', 'traci', 'sumolib', 'numpy'):
        try:
            print(package, version(package))
        except PackageNotFoundError:
            missing.append(package)
    if missing:
        raise SystemExit('Missing: ' + ', '.join(missing))


if __name__ == '__main__':
    main()
