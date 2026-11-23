"""Measure the fixed-time synthetic demo every simulation second."""
import argparse
import csv
from pathlib import Path

from build_scenario import NET, ROUTES, build


def run(gui, output, seed):
    import traci

    build()
    command = ['sumo-gui' if gui else 'sumo', '-n', str(NET), '-r', str(ROUTES), '--seed', str(seed)]
    if gui:
        command += ['--start', '--quit-on-end']
    output.parent.mkdir(parents=True, exist_ok=True)
    traci.start(command)
    try:
        if tuple(traci.trafficlight.getIDList()) != ('J',):
            raise RuntimeError('Expected one controlled junction J')
        lanes = sorted(set(traci.trafficlight.getControlledLanes('J')))
        if len(lanes) != 4:
            raise RuntimeError(f'Expected four incoming lanes, found {lanes}')
        departed = arrived = peak_queue = 0
        phases = set()
        with output.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['time_s', 'phase', 'signal', 'departed_total', 'arrived_total', *lanes])
            for _ in range(360):
                traci.simulationStep()
                departed += traci.simulation.getDepartedNumber()
                arrived += traci.simulation.getArrivedNumber()
                phase = traci.trafficlight.getPhase('J')
                phases.add(phase)
                queues = [traci.lane.getLastStepHaltingNumber(lane) for lane in lanes]
                peak_queue = max(peak_queue, sum(queues))
                writer.writerow([traci.simulation.getTime(), phase, traci.trafficlight.getRedYellowGreenState('J'),
                                 departed, arrived, *queues])
        pending = traci.simulation.getMinExpectedNumber()
        print(f'departed={departed} arrived={arrived} pending={pending} peak_queue={peak_queue} phases={sorted(phases)}')
        if arrived != 140 or departed != arrived or pending or not peak_queue or len(phases) < 2:
            raise RuntimeError('Synthetic flow, clearance, queue or phase check failed')
    finally:
        traci.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('results/demo.csv'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run(args.gui, args.output, args.seed)
