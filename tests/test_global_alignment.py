import unittest

import torch

from stroke_model.global_alignment import alignment_matrices, predict_with_global_alignment


class ReferenceClassifier(torch.nn.Module):
    """Known reference halves: spatial predictions must follow the inverse warp."""
    def forward(self, image, reference, strokes):
        self.seen = image.clone()
        p = torch.zeros_like(strokes)
        p[:, 0, :, :image.shape[-1]//2] = 1
        p[:, 1] = 1 - p[:, 0]
        return dict(probabilities=p, masks=image*p)


class AlignmentTests(unittest.TestCase):
    def test_small_off_center_input_is_aligned_and_restored(self):
        image = torch.zeros(1, 1, 64, 64)
        image[:, :, 7:15, 3:11] = 1
        ref = torch.zeros_like(image)
        ref[:, :, 16:48, 16:48] = 1
        model = ReferenceClassifier()
        output = predict_with_global_alignment(model, image, ref, ref.expand(-1, 2, -1, -1), True)
        # The model sees a centered, enlarged glyph, not the original corner.
        self.assertGreater(float(model.seen[0, 0, 20:44, 20:44].mean()), .99)
        p = output['probabilities']
        self.assertGreater(float(p[0, 0, 9, 4]), .99)
        self.assertGreater(float(p[0, 1, 9, 9]), .99)
        torch.testing.assert_close(p.sum(1), torch.ones_like(image[:, 0]))
        torch.testing.assert_close(output['masks'].sum(1), image[:, 0])
        self.assertEqual(int(output['masks'][0, 0, 30, 30]), 0)

    def test_identity_and_disabled_paths(self):
        image = torch.zeros(1, 1, 32, 32)
        image[:, :, 4:28, 10:20] = 1
        strokes = image.expand(-1, 2, -1, -1)
        model = ReferenceClassifier()
        identity = model(image, image, strokes)
        for enabled in (False, True):
            result = predict_with_global_alignment(model, image, image, strokes, enabled)
            torch.testing.assert_close(result['masks'], identity['masks'])
        self.assertNotIn('global_alignment_theta', predict_with_global_alignment(model, image, image, strokes))

    def test_uniform_scaling_and_inverse_matrices_on_rectangular_canvas(self):
        image = torch.zeros(2, 1, 40, 64)
        image[:, :, 3:11, 4:20] = 1
        reference = torch.zeros_like(image)
        reference[:, :, 12:36, 16:48] = 1
        theta, inverse = alignment_matrices(image, reference)
        torch.testing.assert_close(theta[:, 0, 0], theta[:, 1, 1])
        bottom = torch.tensor([0., 0., 1.]).expand(2, 1, 3)
        product = torch.cat((theta, bottom), 1) @ torch.cat((inverse, bottom), 1)
        torch.testing.assert_close(product, torch.eye(3).expand(2, -1, -1))

    def test_empty_input_rejected(self):
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            alignment_matrices(torch.zeros(1, 1, 32, 32), torch.ones(1, 1, 32, 32))


if __name__ == '__main__':
    unittest.main()
