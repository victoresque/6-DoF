from imgaug import augmenters as iaa


seq_affine = iaa.Sequential([
    iaa.Affine(translate_px=(-2, 2),
               rotate=(-1, 1),
               scale=(0.98, 1.02),
               shear=(-1, 1),
               mode='edge')
])

seq_color = iaa.Sequential([
    iaa.GaussianBlur(sigma=(1.0, 2.0)),
    iaa.AddToHueAndSaturation((-32, 32), channels=[0]),
    iaa.AddToHueAndSaturation((-16, -16), channels=[1]),
    iaa.Multiply((0.5, 1.3)),
])

seq = iaa.Sequential([
    seq_affine,
    seq_color
])