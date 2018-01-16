import imgaug
from imgaug import augmenters as iaa

seq_affine = iaa.Sequential([
    iaa.Affine(translate_px=(-2, 2),
               scale=(0.9, 1.1),
               rotate=(-5, 5),
               shear=(-2, 2),
               mode='edge')
])

seq_color = iaa.Sequential([
    iaa.GaussianBlur(sigma=(0, 0.5)),
    # iaa.Add((-16, 16), per_channel=0.2),
    iaa.Multiply((0.8, 1.2)),
    iaa.Multiply((0.9, 1.1), per_channel=0.4),
])

seq = iaa.Sequential([
    seq_affine,
    seq_color
])