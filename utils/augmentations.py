import kornia.augmentation as KA


def get_augmentation_transform(config):
    method_name = config['name']
    allowed_methods = ['none', 'weak_color_aug']
    if method_name not in allowed_methods:
        raise NameError('{} module was not found among local descriptors. Please choose one of the following '
                        'methods: {}'.format(method_name, ', '.join(allowed_methods)))
    elif method_name == 'none':
        return KA.AugmentationSequential(same_on_batch=True)
    elif method_name == 'weak_color_aug':
        return KA.AugmentationSequential(
            KA.RandomEqualize(p=0.25),
            KA.RandomSharpness(p=0.25),
            KA.RandomSolarize(p=0.25),
            KA.RandomGaussianNoise(p=0.5, mean=0., std=0.05)
        )


if __name__ == '__main__':
    import torch

    aug = KA.RandomEqualize()
    image = torch.rand(1, 1, 5, 6)
    print(aug(image).shape)