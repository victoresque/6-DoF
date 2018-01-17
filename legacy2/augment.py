import imgaug
from imgaug import augmenters as iaa

seq_affine = iaa.Sequential([
    iaa.Affine(translate_px=(-2, 2),
               rotate=(-12, 12),
               scale=(0.95, 1.05),
               shear=(-4, 4),
               mode='edge')
])

seq_color = iaa.Sequential([
    iaa.GaussianBlur(sigma=(1.0, 1.0)),
    iaa.Multiply((0.5, 1.1)),
    iaa.Multiply((0.8, 1.2), per_channel=0.4),
])

seq_brighten = iaa.Sequential([
    iaa.Multiply((1.4, 1.4))
])

seq = iaa.Sequential([
    seq_affine,
    seq_color
])