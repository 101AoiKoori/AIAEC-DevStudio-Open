"""Verify TensorBoard reads real event protobufs from synthetic run records."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

COMMON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMMON))
from tensorboard_metrics import Projection, training_rows
from tensorboard_v3 import _configure_protobuf


class TensorBoardMetricsTests(unittest.TestCase):
    def setUp(self):
        scratch = COMMON.parents[1] / 'tmp'
        scratch.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix='tensorboard-fixture-', dir=scratch)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / 'source/v4_fixture'
        self.csv = self.run / 'training/results.csv'
        self.csv.parent.mkdir(parents=True)
        self.log = self.root / 'view'
        self.log.mkdir()
        self.projection = Projection(self.run.parent, self.log)
        self.addCleanup(self.projection.close)

    def write_results(self, rows):
        self.csv.write_text('epoch, time, train/box_loss, lr/pg0, metrics/mAP50(B), metrics/mAP50-95(P)\n'
                            + rows, encoding='utf-8')

    def test_fixed_budget_uses_named_losses_and_excludes_unmeasured_ap(self):
        config = self.run / 'config/effective.json'
        config.parent.mkdir()
        config.write_text('{"training_policy":{"mode":"fixed"}}', encoding='utf-8')
        self.write_results('1,10,0.5,0.001,0,0\n2,20,0.4,0.001,0,0')
        self.assertEqual(training_rows(self.run), {
            1: {'train/elapsed_seconds': 10.0, 'train/box_loss': 0.5, 'lr/pg0': 0.001}})

    def test_projection_preserves_one_run_and_replaces_rolled_back_steps(self):
        from tensorboard.backend.event_processing.plugin_event_accumulator import EventAccumulator
        from tensorboard.util.tensor_util import make_ndarray

        self.write_results('1,10,0.5,0.001,0.25,0.1\n2,20,0.4,0.001,0.5,0.2\n')
        self.projection.refresh()
        accumulator = EventAccumulator(str(self.log / 'source/v4_fixture/00 Overall')).Reload()
        self.assertEqual([item.step for item in accumulator.Tensors('Box/mAP50')], [1, 2])
        self.assertIn('Pose/mAP50-95', accumulator.Tags()['tensors'])
        self.write_results('1,10,0.5,0.001,0.25,0.1\n')
        self.projection.refresh()
        self.write_results('1,10,0.5,0.001,0.25,0.1\n2,20,0.4,0.001,0.75,0.2\n')
        self.projection.refresh()
        values = accumulator.Reload().Tensors('Box/mAP50')
        self.assertEqual([(item.step, float(make_ndarray(item.tensor_proto))) for item in values], [(1, .25), (2, .75)])

    def test_hparams_route_and_scalar_tags_with_installed_protobuf(self):
        from tensorboard import program
        from werkzeug.test import Client

        self.write_results('1,10,0.5,0.001,0.25,0.1\n')
        evaluation = self.run / 'evaluation/val_fixture'
        evaluation.mkdir(parents=True)
        (evaluation / 'summary.json').write_text(json.dumps({'split': 'val', 'metrics': {'metrics/mAP50(B)': .5}}),
                                                 encoding='utf-8')
        (evaluation / 'per_class_metrics.csv').write_text('class_id,name,box_ap50\n0,target,0.5\n', encoding='utf-8')
        self.projection.refresh()
        _configure_protobuf()
        board = program.TensorBoard()
        board.configure(argv=[None, '--logdir', str(self.log), '--host', '127.0.0.1', '--port', '0',
                              '--load_fast=false', '--reload_interval=0', '--reload_task=blocking'])
        server = board._make_server()
        self.addCleanup(server.server_close)
        client = Client(server.app)
        experiment = '/experiment/defaultExperimentId/data'
        self.assertEqual(client.get(experiment + '/plugin/hparams/experiment',
                                    query_string={'request': '{}'}).status_code, 200)
        response = client.get('/data/plugin/scalars/tags')
        self.assertEqual(response.status_code, 200)
        tags = {name.replace('\\', '/'): values for name, values in response.get_json().items()}
        overall = 'source/v4_fixture/00 Overall'
        self.assertIn('Box/mAP50', tags[overall])
        self.assertIn('Class 0 target/box_ap50', tags['source/v4_fixture/Evaluation/val_fixture'])
        self.assertIn(overall, [name.replace('\\', '/') for name in client.get(experiment + '/runs').get_json()])

    def test_legacy_attempts_have_readable_chronological_names_and_keep_their_own_points(self):
        from tensorboard.backend.event_processing.plugin_event_accumulator import EventAccumulator
        from tensorboard.compat.proto import event_pb2, summary_pb2
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        from tensorboard.util.tensor_util import make_ndarray

        metadata = self.run / 'metadata'
        metadata.mkdir()
        attempts = [('train', '2026-11-16T07:59:51+00:00', [1, 2]),
                    ('resume_' + 'a' * 32, '2026-11-16T08:25:58+00:00', [2, 3]),
                    ('resume_20261116T083500_000000Z', '2026-11-16T08:35:00+00:00', [4])]
        for phase, started, steps in attempts:
            (metadata / f'{phase}_status.json').write_text(json.dumps({'started_at_utc': started}), encoding='utf-8')
            directory = 'train_20261116T075951_000000Z' if phase == 'train' else phase
            writer = EventFileWriter(str(self.run / 'tensorboard' / directory))
            for step in steps:
                event = event_pb2.Event(step=step, wall_time=100 + step, summary=summary_pb2.Summary(
                    value=[summary_pb2.Summary.Value(tag='train/loss_0', simple_value=step)]))
                writer.add_event(event)
            writer.close()
        self.write_results('1,10,0.5,0.001,0.25,0.1\n2,20,0.4,0.001,0.5,0.2\n')
        self.projection.refresh()
        self.projection.refresh()
        title = self.log / 'source/v4_fixture'
        names = [item.name for item in sorted(title.iterdir())]
        self.assertEqual(names, ['00 Overall', '01 Train 2026-11-16 07.59.51 UTC',
                                 '02 Resume 2026-11-16 08.25.58 UTC', '03 Resume 2026-11-16 08.35.00 UTC'])
        for (_, _, steps), name in zip(attempts, names[1:]):
            reader = EventAccumulator(str(title / name)).Reload()
            data = reader.Tensors('train/box_loss')
            self.assertEqual([(point.step, float(make_ndarray(point.tensor_proto))) for point in data],
                             [(step, float(step)) for step in steps])


if __name__ == '__main__':
    unittest.main()
