import imgaug
from imgaug import augmenters as iaa

seq_affine = iaa.Sequential([
    iaa.Affine(translate_px=(-2, 2),
               rotate=(-8, 8),
               scale=(0.95, 1.05),
               mode='edge')
])

seq_color = iaa.Sequential([
    # iaa.GaussianBlur(sigma=(1.0, 1.0)),
    iaa.Multiply((0.5, 1.5)),
    iaa.Add((-10, 10), per_channel=1.0),
])

seq_brighten = iaa.Sequential([
    iaa.Multiply((1.4, 1.4))
])

seq = iaa.Sequential([
    seq_affine,
    seq_color
])