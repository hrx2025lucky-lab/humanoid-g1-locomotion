"""Show or run one explicitly budgeted training command in an external framework."""
from pathlib import Path
import argparse
import json
import shlex
import subprocess


def command_for(number, variant, python, directory, environments, iterations, seed):
    tasks = {
        1: 'Unitree-G1-29dof-Velocity',
        2: 'Unitree-G1-29dof-Velocity-Rough',
        4: 'Mjlab-VelocityHeight-Flat-Unitree-G1',
        5: 'Unitree-G1-29dof-Navigation-HRL-Baseline',
        6: 'Mjlab-Humanoid-HW6-Student-Action-Matching-G1',
        8: 'Unitree-G1-29dof-AMP-WalkToRun',
        9: 'Mjlab-Humanoid-HW6-Teacher-G1',
        10: 'Unitree-G1-29dof-Mimic-HOI_terrain-Perceptive-Raycast',
        11: 'Instinct-Parkour-Target-Amp-G1-v0',
    }
    variants = {
        (4, 'no-height-reward'): tasks[4] + '-NoHeightRew',
        (4, 'no-height-observation'): tasks[4] + '-BlindActor',
        (5, 'random-layout'): 'Unitree-G1-29dof-Navigation-HRL-RandomArena',
        (6, 'kl'): 'Mjlab-Humanoid-HW6-Student-KL-Matching-G1',
    }
    if number not in tasks:
        raise ValueError('This practice records deployment or retargeting, not a training job.')
    if environments < 1 or iterations < 1:
        raise ValueError('Environment count and update budget must be positive.')
    task = tasks[number] if variant == 'default' else variants.get((number, variant))
    if task is None:
        raise ValueError('This variant is not defined for the selected practice.')
    if number in (4, 6, 9):
        return [python, '-m', 'mjlab.scripts.train', task,
                f'--env.scene.num-envs={environments}', f'--agent.max-iterations={iterations}',
                f'--agent.seed={seed}', '--agent.logger=tensorboard']
    script = 'scripts/instinct_rl/train.py' if number == 11 else 'scripts/rsl_rl/train.py'
    return [python, str(Path(directory) / script), '--task', task, '--num_envs', str(environments),
            '--max_iterations', str(iterations), '--seed', str(seed), '--headless']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('practice', type=int, choices=range(1, 12))
    parser.add_argument('--variant', default='default')
    parser.add_argument('--framework-dir', required=True, type=Path)
    parser.add_argument('--python', required=True, help='Interpreter belonging to the selected framework.')
    parser.add_argument('--num-envs', required=True, type=int)
    parser.add_argument('--iterations', required=True, type=int, help='Explicit update budget for this new run.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--execute', action='store_true', help='Start the displayed command in the foreground.')
    args = parser.parse_args()
    directory = args.framework_dir.expanduser().resolve()
    try:
        cmd = command_for(args.practice, args.variant, args.python, directory, args.num_envs, args.iterations, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({'cwd': str(directory), 'command': shlex.join(cmd), 'execute': args.execute,
                      'mode': 'new run; existing checkpoints are not selected automatically'}, indent=2))
    if not args.execute:
        return
    if not directory.is_dir():
        parser.error('Framework directory does not exist.')
    result = subprocess.run(cmd, cwd=directory, check=False)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
