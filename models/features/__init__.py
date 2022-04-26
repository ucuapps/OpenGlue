from models.features.hardnet import GFTTAffNetHardNet
from models.features.opencv import methods as OPENCV_METHODS
from models.features.sift import SIFT
from models.features.superpoint import methods as SUPERPOINT_METHODS
from models.features.iterative_features_extractor import IterativeLocalFeature
from functools import partial

methods = {
    'SIFT': SIFT,
    'GFTTAffNetHardNet': GFTTAffNetHardNet,
    **SUPERPOINT_METHODS,
    **OPENCV_METHODS
}


def get_feature_extractor(model_name):
    """
    Create method form configuration
    """
    if model_name not in methods:
        raise NameError('{} module was not found among local descriptors. Please choose one of the following '
                        'methods: {}'.format(model_name, ', '.join(methods.keys())))

    return methods[model_name]


__all__ = ['get_feature_extractor']
