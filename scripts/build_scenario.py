"""Build the synthetic four-arm intersection from versioned geometry."""
from pathlib import Path
import subprocess

SCENE = Path(__file__).resolve().parents[1] / 'sumo' / 'demo'
NET = SCENE / 'junction.net.xml'
ROUTES = SCENE / 'traffic.rou.xml'


def build():
    subprocess.run([
        'netconvert', '--node-files', 'junction.nod.xml', '--edge-files', 'junction.edg.xml',
        '--connection-files', 'junction.con.xml', '--tls.default-type', 'static',
        '--tls.green.time', '25', '--tls.yellow.time', '3', '--no-turnarounds',
        '--output-file', NET.name,
    ], cwd=SCENE, check=True)


if __name__ == '__main__':
    build()
