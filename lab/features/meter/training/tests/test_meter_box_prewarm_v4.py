"""Check the restricted box-only adapter, without running training."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[5]
PATH=ROOT/'lab/features/meter/training/scripts/train_meter_box_prewarm_v4.py'
spec=importlib.util.spec_from_file_location('meter_box_prewarm',PATH)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class BoxPrewarmEncoding(unittest.TestCase):
    def test_box_and_all_unknown_points_are_separate(self):
        values=module.encode_box([160,120,480,360]).split()
        self.assertEqual(len(values),29)
        self.assertEqual(values[:5],['0','0.50000000','0.50000000','0.50000000','0.50000000'])
        self.assertEqual(values[5:],['0']*24)

    def test_invalid_box_is_rejected(self):
        for box in ([0,0,0,1],[-1,0,20,10],[0,0,641,480],[0,0,float('nan'),10],None):
            with self.subTest(box=box),self.assertRaises(ValueError):module.encode_box(box)

    def test_source_must_mark_every_point_unknown_and_box_evidenced(self):
        row={'keypoint_visibility':dict.fromkeys(module.POINTS,'unknown'),
             'dial_raster_evidence':{'bbox_status':'candidate_from_isolated_raster_and_visible_fraction',
                                     'bbox_xyxy_pixels':[10,20,100,110]}}
        self.assertEqual(len(module.assert_unknown_points(row).split()),29)
        row['keypoint_visibility']['humidity_tip']='visible'
        with self.assertRaisesRegex(ValueError,'unknown'):module.assert_unknown_points(row)
        row['keypoint_visibility']['humidity_tip']='unknown';row['dial_raster_evidence']['bbox_status']='unknown'
        with self.assertRaisesRegex(ValueError,'raster evidence'):module.assert_unknown_points(row)

    def test_specialized_trainer_uses_the_shared_fixed_policy(self):
        class FixedTrainer:
            def validate(self):return {}, None
            def final_eval(self):return None
            def save_model(self):return None

        counters={'validation_calls_skipped':0,'final_eval_calls_skipped':0,'checked_batches':0}
        cls=module.make_box_only_trainer(FixedTrainer,None,counters)
        trainer=object.__new__(cls)
        self.assertEqual(trainer.validate(),({},None))
        self.assertIsNone(trainer.final_eval())
        self.assertEqual(counters['validation_calls_skipped'],1)
        self.assertEqual(counters['final_eval_calls_skipped'],1)
        self.assertIs(cls.save_model, FixedTrainer.save_model)

    def test_legacy_checkpoint_without_optimizer_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no saved optimizer state and cannot resume'):
            module.validate_box_resume({'epoch': 0, 'optimizer': None}, {'epochs': 3}, None)

    def test_resume_requires_matching_fixed_policy_and_unfinished_epoch_budget(self):
        config={'epochs': 3}
        signature=module.prewarm_run_signature(config,'fixture-data')
        resume={'schema_version':1,'task':'pose','policy':module.BOX_POLICY,'epochs':3,
                'run_signature':signature,'trained_epochs':1,'terminal':False,
                'model':{'weight':0},'ema':{'weight':0},'scheduler':{'last_epoch':0},
                'rng':dict.fromkeys(('python','numpy','torch','cuda')),'stopper':{}}
        checkpoint={'epoch':0,'optimizer':{'state':{},'param_groups':[{}]},'scaler':{},
                    'updates':1,'policy_resume':resume}
        self.assertEqual(module.validate_box_resume(checkpoint,config,signature),1)
        with self.assertRaisesRegex(ValueError,'signature differs'):
            module.validate_box_resume(checkpoint,config,'different')
        resume['terminal']=True
        with self.assertRaisesRegex(ValueError,'finished training run'):
            module.validate_box_resume(checkpoint,config,signature)

    def test_resume_dispatches_original_budget_without_preparing_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            base=root/'runs/meter'
            run=base/'copied'
            checkpoint=run/'training/weights/last.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'fixture')
            metadata=run/'metadata'
            module.write_json(metadata/'config.json',{'epochs':3,'device_requested':'cpu','samples':40})
            with patch.multiple(module,ROOT=root,RUNS=base):
                def fixture_child(command, deadline, log, cwd):
                    self.assertEqual(cwd,root)
                    self.assertEqual(command[command.index('--epochs')+1],'3')
                    self.assertEqual(command[command.index('--device')+1],'cpu')
                    self.assertEqual(root/command[command.index('--resume')+1],checkpoint)
                    attempt=command[command.index('--attempt')+1]
                    module.write_json(metadata/f'result_{attempt}.json',
                                      {'status':'completed','optimizer_updates':4,'checkpoint':str(checkpoint)})

                with patch.object(module,'prepare',side_effect=AssertionError('Cannot prepare on resume')), \
                        patch.object(module,'run_bounded',side_effect=fixture_child) as controller:
                    module.main(['--resume',str(checkpoint.relative_to(root))])
                    controller.assert_called_once()


if __name__=='__main__':unittest.main()
