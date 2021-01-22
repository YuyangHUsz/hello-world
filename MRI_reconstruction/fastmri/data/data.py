import pathlib
from fastmri.data import subsample
from fastmri.data import transforms, mri_data

# Create a mask function
mask_func = subsample.RandomMaskFunc(
    center_fractions=[0.08],
    accelerations=[4]
)


def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
    # Transform the data into appropriate format
    # Here we simply mask the k-space and return the result
    kspace = transforms.to_tensor(kspace)
    masked_kspace,_ = transforms.apply_mask(kspace, mask_func)
    return masked_kspace



dataset = mri_data.SliceDataset(
    root=pathlib.Path(
      '/Users/huyuyang/singlecoil_val'
    ),
    transform=data_transform,
    challenge='singlecoil'
)

for masked_kspace in dataset:
    # Do reconstruction
    pass