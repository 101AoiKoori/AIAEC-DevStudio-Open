"""Command line shared by task-specific scripts and copied distribution scripts."""
from pathlib import Path
import argparse
import json
import sys
from .config import load_config
from .runner import preflight, train_run, bounded_train, evaluate_run, tune_runs, compare_runs, tune_thresholds
from .bounded import BudgetExhausted


def main(default_root=None, default_config=None, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'train', 'evaluate', 'freeze', 'tune', 'compare'))
    parser.add_argument('--root', type=Path, default=default_root or Path.cwd())
    parser.add_argument('--config', default=default_config or 'config/train.yaml')
    parser.add_argument('--run-name')
    parser.add_argument('--resume', help='Root-relative incomplete last.pt checkpoint')
    parser.add_argument('--wall-seconds', type=float, help='External hard wall-clock limit for training')
    parser.add_argument('--weights', help='Root-relative checkpoint for explicit evaluation')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--run-dir', help='Root-relative existing run, for evaluation')
    parser.add_argument('--evaluation-name')
    parser.add_argument('--selection-reason', help='Validation-based reason for freezing a candidate')
    parser.add_argument('--runs', nargs='+', help='Root-relative completed runs to compare')
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve()
        config = load_config(root, args.config)
        if args.command == 'check':
            print(json.dumps(preflight(root, config), ensure_ascii=False, indent=2)); return 0
        if args.command == 'train':
            if config['version'].lower() == 'v4' and args.wall_seconds is None:
                raise ValueError('V4 training requires an explicit --wall-seconds limit')
            if args.wall_seconds is not None:
                if args.resume:
                    raise ValueError('Bounded resume is not implemented')
                run_dir = bounded_train(root, config, args.config, args.wall_seconds, args.run_name)
            else:
                run_dir = train_run(root, config, args.run_name, args.resume)
        elif args.command == 'evaluate':
            if not args.weights: parser.error('evaluate requires --weights')
            run_dir = evaluate_run(root, config, args.weights, args.split, args.run_dir, args.evaluation_name)
        elif args.command == 'freeze':
            if not args.weights or not args.run_dir or not args.evaluation_name or not args.selection_reason:
                parser.error('freeze requires --weights, --run-dir, --evaluation-name and --selection-reason')
            from .freeze_gate import freeze_selection
            selected = freeze_selection(root, config, args.run_dir, args.weights, args.evaluation_name, args.selection_reason)
            print('FROZEN_CHECKPOINT:', selected, flush=True)
            return 0
        elif args.command == 'tune':
            if config['version'].lower() == 'v4' and not args.weights:
                raise ValueError('V4 tuning controller is not implemented; use explicitly bounded individual runs')
            run_dir = tune_thresholds(root, config, args.weights, args.run_name) if args.weights else tune_runs(root, config, args.run_name)
        else:
            if not args.runs: parser.error('compare requires --runs')
            run_dir = compare_runs(root, config, args.runs, args.run_name)
        print('COMPLETED_RUN:', run_dir, flush=True)
        return 0
    except KeyboardInterrupt:
        print('Training interrupted; inspect run metadata and logs.', file=sys.stderr); return 130
    except BudgetExhausted as error:
        print('BUDGET_EXHAUSTED:', error, file=sys.stderr); return 124
    except Exception as error:
        print(f'ERROR: {type(error).__name__}: {error}', file=sys.stderr); return 1


if __name__ == '__main__':
    sys.exit(main())
