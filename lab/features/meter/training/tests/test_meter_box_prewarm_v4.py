"""Check the restricted box-only adapter, without running training."""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

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

    def test_specialized_trainer_skips_validation_and_writes_only_last(self):
        class Parent:
            def validate(self):raise AssertionError('base validator must never run')
            def final_eval(self):raise AssertionError('base final evaluation must never run')
            def save_model(self):raise AssertionError('base best-model selection must never run')
        class Model:
            def half(self):return self
        writes=[]
        fake_torch=SimpleNamespace(save=lambda obj,path:writes.append((obj,path)))
        counters={'validation_calls_skipped':0,'final_eval_calls_skipped':0,'checked_batches':0}
        cls=module.make_box_only_trainer(Parent,fake_torch,'test',counters)
        trainer=object.__new__(cls);trainer.epoch=0;trainer.ema=SimpleNamespace(ema=Model(),updates=2)
        trainer.args=SimpleNamespace(pose=0.,kobj=0.);trainer.last=Path('last.pt');trainer.best=Path('best.pt')
        self.assertEqual(trainer.validate(),({},0.));trainer.final_eval();trainer.save_model()
        self.assertEqual(counters['validation_calls_skipped'],1);self.assertEqual(counters['final_eval_calls_skipped'],1)
        self.assertEqual(len(writes),1);checkpoint,path=writes[0]
        self.assertEqual(path,Path('last.pt'));self.assertIsNone(checkpoint['best_fitness'])
        self.assertEqual(checkpoint['train_metrics'],{});self.assertIsNone(checkpoint['optimizer'])


if __name__=='__main__':unittest.main()
