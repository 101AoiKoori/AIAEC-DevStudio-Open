import unittest
import numpy as np
from lab.features.e_bike.training.scripts.source_contact_audit import roles, mirror_digest


class SourceContactAuditTest(unittest.TestCase):
    def test_mirror_hash_preserves_geometry_and_color(self):
        image=np.arange(24,dtype=np.uint8).reshape(2,4,3)
        self.assertEqual(mirror_digest(image),mirror_digest(image[:,::-1]))
        self.assertNotEqual(mirror_digest(image),mirror_digest(image[:,:,::-1]))

    def test_foreground_role_uses_record_lineage(self):
        row={'source_record':{'objects':[{'foreground_source':{'path':'crop.png','source_video':'vehicle.mp4'}}],
                             'background':{'image':'bg.png','parent_group':'scene.blend'}}}
        found=roles(row)
        self.assertEqual(found['foreground_crop'],['crop.png'])
        self.assertEqual(found['foreground_parent'],['vehicle.mp4'])
        self.assertEqual(found['background_parent'],['scene.blend'])

    def test_focus_only_crop_reference_can_resolve_selected_parent(self):
        row={'source_record':{'research_details':{'generation_state':{'target':{'vehicles':[{'source':'crop.png'}]}}}}}
        self.assertEqual(roles(row,{'crop.png':{'source_video':'vehicle.mp4'}})['foreground_parent'],['vehicle.mp4'])


if __name__=='__main__':unittest.main()
