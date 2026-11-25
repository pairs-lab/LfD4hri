"""Small real-decoder regression fixtures; no robot/model imports or GPU."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from .export import FFMPEG,THREAD_ENV,VIEWS,CAMERAS,check_png,independent_count,png_encode_command,publish_dir,raw_samples,sha,space


class ExportTests(unittest.TestCase):
    def test_only_requested_four_views(self):
        self.assertEqual(VIEWS,(2,3,4,6))
        self.assertEqual(CAMERAS,('view2','view3','view4','view6'))
        self.assertTrue(set(CAMERAS).isdisjoint({'view1','view5','handeye'}))

    def test_png_crc_dimensions_and_truncation(self):
        with tempfile.TemporaryDirectory(prefix='vla_frame_test_') as t:
            p=Path(t)/'a.png';a=np.arange(32*32*3,dtype=np.uint8).reshape(32,32,3)
            cv2.imwrite(str(p),a);check_png(p,32,32)
            with self.assertRaises(ValueError):check_png(p,31,32)
            original=p.read_bytes();bad=bytearray(original);bad[45]^=1;p.write_bytes(bad)
            with self.assertRaises(ValueError):check_png(p,32,32)
            p.write_bytes(original[:-4])
            with self.assertRaises(ValueError):check_png(p,32,32)

    def test_every_frame_native_RGB_and_numbering(self):
        with tempfile.TemporaryDirectory(prefix='vla_frame_decode_') as t:
            root=Path(t);video=root/'video.mp4';frames=root/'frames';frames.mkdir()
            a=np.zeros((7,32,32,3),np.uint8)
            for i in range(7):a[i]=[i*31,200-i*20,i*17]
            command=[FFMPEG,'-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','32x32','-r','30',
                '-i','pipe:0','-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(video)]
            subprocess.run(command,input=a.tobytes(),check=True,env=THREAD_ENV)
            subprocess.run(png_encode_command(video,frames),check=True,env=THREAD_ENV)
            self.assertEqual(independent_count(video),7)
            self.assertEqual([p.name for p in sorted(frames.iterdir())],[f'frame{i:06d}.png' for i in range(1,8)])
            decoded=raw_samples(video,list(range(7)),32,32)
            for i,expected in enumerate(decoded,1):
                p=frames/f'frame{i:06d}.png';check_png(p,32,32)
                np.testing.assert_array_equal(cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB),expected)

    def test_atomic_directory_never_overwrites_user_folder(self):
        with tempfile.TemporaryDirectory(prefix='vla_frame_publish_') as t:
            root=Path(t);stage=root/'stage';stage.mkdir();target=root/'target';target.mkdir()
            with self.assertRaises(OSError):publish_dir(stage,target)
            self.assertTrue(stage.is_dir());self.assertTrue(target.is_dir())

    def test_duplicate_frames_are_independent_copies(self):
        with tempfile.TemporaryDirectory(prefix='vla_frame_copy_') as t:
            root=Path(t);a=root/'a.png';b=root/'b.png'
            cv2.imwrite(str(a),np.zeros((8,8,3),np.uint8));original=sha(a);shutil.copy2(a,b)
            self.assertNotEqual(a.stat().st_ino,b.stat().st_ino)
            b.write_bytes(b'annotation changed this file')
            self.assertEqual(sha(a),original)

    def test_disk_reserve_gate(self):
        with patch('VLA.frame_export_r61.export.shutil.disk_usage',return_value=shutil._ntuple_diskusage(100,80,20)):
            with self.assertRaises(RuntimeError):space(Path('/tmp'))


if __name__=='__main__':unittest.main()
