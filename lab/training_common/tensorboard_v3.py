"""Serve recorded V4 metrics in TensorBoard, independently of training."""
import argparse
import inspect
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace


def _configure_protobuf():
    from google.protobuf import json_format
    from tensorboard.plugins.custom_scalar import custom_scalars_plugin
    from tensorboard.plugins.hparams import hparams_plugin

    parameters = inspect.signature(json_format.MessageToJson).parameters
    if 'including_default_value_fields' in parameters or 'always_print_fields_with_no_presence' not in parameters:
        return

    def to_json(message, *, including_default_value_fields=False, **options):
        options.setdefault('always_print_fields_with_no_presence', including_default_value_fields)
        return json_format.MessageToJson(message, **options)

    # Keep the adapter inside the TensorBoard plugins that use the removed keyword.
    for plugin in (hparams_plugin, custom_scalars_plugin):
        plugin.json_format = SimpleNamespace(MessageToJson=to_json, Parse=json_format.Parse,
                                             ParseError=json_format.ParseError)


def _source(args, here):
    from launch_task import launcher_settings, targets
    from engine.config import load_config, resolve

    root = here.parents[2] if args.task else here.parents[1]
    config_root = root
    if args.package:
        if not args.task:
            raise ValueError('-Package requires a workspace task name')
        key = 'fire_sticker' if args.task == 'fire' else args.task
        config_root = resolve(root, 'training_packages/' + key)
    if args.task:
        settings = launcher_settings(here.with_name('launcher.yaml'), args.task)
    else:
        settings = launcher_settings(root / 'config/launcher.yaml', None)
    default = targets(root, args.task)[1]
    config_path = 'config/train.yaml' if args.package else settings['train_config'] or default
    config = load_config(config_root, config_path)
    source = resolve(config_root, config['output_root'])
    if args.run:
        value = Path(args.run)
        if value.name != args.run or args.run in {'.', '..'}:
            raise ValueError('-Run expects a run directory name')
        run = resolve(source, value)
        if not run.is_dir() or run.parent != source:
            raise ValueError(f'Run directory missing: {run}')
        source = run
    return source, root / 'runs/_tensorboard', settings['port'], settings['open_browser']


def serve(source, logdir, port, *, open_browser=False, single_run=False):
    from tensorboard import program
    from tensorboard_metrics import Projection

    _configure_protobuf()
    projection = Projection(source, logdir, single_run=single_run)
    try:
        projection.refresh()
        board = program.TensorBoard()
        board.configure(argv=[None, '--logdir', str(logdir), '--host', '127.0.0.1', '--port', str(port),
                              '--load_fast=false', '--reload_interval=3', '--samples_per_plugin=scalars=1000'])
        url = board.launch()
        print('TensorBoard:', url, flush=True)
        print('Reading:', source, flush=True)
        if open_browser and os.name == 'nt':
            try:
                os.startfile(url)
            except OSError as error:
                print(f'Open {url} manually ({error})', file=sys.stderr, flush=True)
        while True:
            time.sleep(3)
            projection.refresh()
    finally:
        projection.close()


def main(argv=None):
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parent))
    sys.path.insert(0, str(here.parent / 'common'))
    from launch_task import TASK_FEATURES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASK_FEATURES)
    parser.add_argument('--package', action='store_true')
    parser.add_argument('--run')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--logdir', type=Path)
    parser.add_argument('--port', type=int)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args(argv)
    if args.source:
        if not args.logdir:
            parser.error('--source requires --logdir')
        source, cache, port, browser = args.source, args.logdir.parent, 0, False
    else:
        source, cache, port, browser = _source(args, here)
    if args.port is not None:
        port = args.port
    if args.plan:
        print(f'Source: {source}\nCache: {cache}\nPort: {port}')
        return 0
    cache.mkdir(parents=True, exist_ok=True)
    if args.logdir:
        args.logdir.mkdir(parents=True, exist_ok=True)
        serve(source, args.logdir, port, open_browser=browser and not args.no_browser, single_run=bool(args.run))
    else:
        with TemporaryDirectory(prefix='view-', dir=cache) as temp:
            serve(source, Path(temp), port, open_browser=browser and not args.no_browser, single_run=bool(args.run))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
    except (OSError, ValueError, RuntimeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(1)
