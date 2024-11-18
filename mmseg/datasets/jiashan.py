import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class JiaShanDataset(CustomDataset):
    """JiaShan dataset.

    Args:
        split (str): Split txt file for JiaShan.
    """

    CLASSES = ('background', 'edge')

    # PALETTE = [[0, 0, 0], [255, 255, 255]]
    PALETTE = [[255, 255, 255], [0, 0, 0]]

    def __init__(self, split, **kwargs):
        super(JiaShanDataset, self).__init__(
            img_suffix='.png',      #底图影像
            seg_map_suffix='.png',  #标签影像
            split=split, **kwargs
        )
        assert osp.exists(self.img_dir) and self.split is not None
