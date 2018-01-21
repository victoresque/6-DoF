import imgaug
from imgaug import augmenters as iaa

seq_affine = iaa.Sequential([
    iaa.Affine(translate_px=(-2, 2),
               rotate=(-1, 1),
               scale=(0.98, 1.02),
               shear=(-1, 1),
               mode='edge')
])

seq_color = iaa.Sequential([
    iaa.GaussianBlur(sigma=(0.8, 1.4)),
    iaa.Multiply((0.5, 1.2)),
    iaa.Multiply((0.9, 1.1), per_channel=0.4),
])

seq = iaa.Sequential([
    seq_affine,
    seq_color
])