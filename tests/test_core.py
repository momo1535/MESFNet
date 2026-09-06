import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import infer
import train
from model.dataset_load import RemoteSensingDataset
from model.losses4 import TotalLoss
from model.mesfnet import MESFNet
from model.postprocess import fill_small_holes
from tools.migrate_legacy_checkpoint import migrate_state_dict

torch.set_num_threads(1)


class PipelineTests(unittest.TestCase):
    def test_checkpoint_key_migration(self):
        tensor = torch.randn(2, 3)
        state = {"dec4.bn.weight": tensor, "dec4.sk.convs.0.bn.running_mean": tensor,
                 "scp1.detail.1.weight": tensor, "s1.0.layer_scale": tensor}
        migrated = migrate_state_dict(state)
        self.assertEqual(list(migrated), ["decoder_stage4.norm.weight",
            "decoder_stage4.sk.convs.0.norm.running_mean", "detail_enhance1.detail.1.weight",
            "encoder_stage1.0.layer_scale"])
        self.assertTrue(all(value is tensor for value in migrated.values()))
        self.assertEqual(list(migrate_state_dict(migrated)), list(migrated))
        with self.assertRaises(ValueError):
            migrate_state_dict({"dec4.bn.weight": tensor, "decoder_stage4.norm.weight": tensor})
        with self.assertRaises(ValueError):
            migrate_state_dict({"epoch": 150, "model": state})

    def test_d4_inverse_and_average(self):
        image = torch.randn(1, 3, 13, 17)
        for k, flip in infer.MODES["d4"]:
            self.assertTrue(torch.equal(image, infer.inverse_transform(infer.transform(image, k, flip), k, flip)))
        class PixelModel(torch.nn.Module):
            def forward(self, x):
                return {"mask": x[:, :1]}
        single = infer.predict_probability(PixelModel(), image, "none")
        for tta in ("flip4", "d4"):
            self.assertTrue(torch.allclose(single, infer.predict_probability(PixelModel(), image, tta), atol=1e-6))

    def test_holes_preserve_border_and_large_components(self):
        mask = np.ones((16, 16), dtype=bool)
        mask[3:5, 3:5] = False
        mask[8:12, 8:12] = False
        mask[0:3, 10:12] = False
        result = fill_small_holes(mask, 4)
        self.assertTrue(result[3:5, 3:5].all())
        self.assertFalse(result[8:12, 8:12].any())
        self.assertFalse(result[0:3, 10:12].any())

    def test_dataset_and_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("image", "mask"):
                (root / name).mkdir()
            Image.new("RGB", (64, 64), (100, 150, 200)).save(root / "image" / "sample.jpg")
            mask = np.zeros((64, 64), np.uint8)
            mask[10:40, 10:40] = 255
            Image.fromarray(mask).save(root / "mask" / "sample_mask.png")
            image, target, edge, morph = RemoteSensingDataset(root, img_size=(64, 64))[0]
            self.assertEqual(tuple(image.shape), (3, 64, 64))
            self.assertEqual(target.sum().item(), 900)
            self.assertEqual(tuple(edge.shape), (1, 64, 64))
            self.assertTrue(torch.isfinite(morph).all())
            args = argparse.Namespace(config=train.ROOT / "configs" / "train.json", data_root=root,
                save_dir=root / "run", gpu="1", split_dir=None, init_checkpoint=None,
                hard_case_manifest=None, resume=False)
            with patch.dict(os.environ, {"ABLATIONS": "no_sk", "FULL_TRAIN": "1"}):
                config, env = train.build_environment(args)
            self.assertEqual(config["ABLATIONS"], "groupnorm_decoder")
            self.assertEqual(env["FULL_TRAIN"], "0")
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
            args.save_dir.mkdir()
            (args.save_dir / "important.txt").touch()
            with self.assertRaises(FileExistsError):
                train.build_environment(args)
            self.assertEqual({p.name for p in (train.ROOT / "configs").iterdir()}, {"train.json", "infer.json"})
            for config_file in (train.ROOT / "configs").glob("*.json"):
                self.assertIsInstance(json.loads(config_file.read_text()), dict)

    def test_model_forward_and_loss_backward(self):
        model = MESFNet(pretrained=False, backbone_variant="base", ablations={"groupnorm_decoder"}).eval()
        for layer in (model.decoder_stage4.norm, model.decoder_stage3.norm, model.segmentation_fusion[1]):
            self.assertIsInstance(layer, torch.nn.GroupNorm)
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(tuple(outputs["mask"].shape), (1, 1, 64, 64))
        self.assertEqual(tuple(outputs["edge"].shape), (1, 1, 64, 64))
        self.assertEqual(tuple(outputs["morph"].shape), (1, 4))
        outputs = {k: v.detach().requires_grad_() for k, v in outputs.items()}
        target = torch.zeros_like(outputs["mask"])
        target[:, :, 20:40, 20:40] = 1
        loss = TotalLoss()(outputs, target, torch.zeros_like(target), torch.zeros(1, 4))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(outputs["mask"].grad).all())


if __name__ == "__main__":
    unittest.main()
