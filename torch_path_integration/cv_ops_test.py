import unittest

import numpy as np
import torch

from torch_path_integration import cv_ops


class TransformationMatrixTestCase(unittest.TestCase):
    def test_decompose(self):
        batch_shape = (4, 3)

        scale_x = torch.rand(*batch_shape)
        scale_y = torch.rand(*batch_shape)
        trans_x = torch.rand(*batch_shape)
        trans_y = torch.rand(*batch_shape)
        rotation = torch.rand(*batch_shape) * np.pi

        T = cv_ops.affine_transform_2d(
            scale_x=scale_x,
            scale_y=scale_y,
            trans_x=trans_x,
            trans_y=trans_y,
            rotation=rotation,
        )

        rot, sx, sy, sh, tx, ty = cv_ops.decompose_transformation_matrix(T)

        self.assertTrue((scale_x - sx).sum() < 1e-6)
        self.assertTrue((scale_y - sy).sum() < 1e-6)
        self.assertTrue((rotation - rot).sum() < 1e-6)
        self.assertTrue((trans_x - tx).sum() < 1e-6)
        self.assertTrue((trans_y - ty).sum() < 1e-6)


if __name__ == '__main__':
    unittest.main()
