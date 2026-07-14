import os
import sys
from tqdm import tqdm
import h5py
import numpy as np
import SimpleITK as sitk
sep = '\\' if sys.platform[:3] == 'win' else '/'


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


data_path = 'DATA_FOLDER/'
slice_num = 0
relabel_dict = {200: 2, 500: 3, 600: 1}
c0_image_files = sorted([os.path.join(data_path, x, y)
                                for x in os.listdir(data_path) if x.endswith('images')
                                for y in os.listdir(os.path.join(data_path, x)) if y.endswith('C0.nii.gz')
                                ])
lge_image_files = sorted([os.path.join(data_path, x, y)
                                for x in os.listdir(data_path) if x.endswith('images')
                                for y in os.listdir(os.path.join(data_path, x)) if y.endswith('LGE.nii.gz')
                                ])
t2_image_files = sorted([os.path.join(data_path, x, y)
                                for x in os.listdir(data_path) if x.endswith('images')
                                for y in os.listdir(os.path.join(data_path, x)) if y.endswith('T2.nii.gz')
                                ])
assert len(c0_image_files) == len(lge_image_files) == len(t2_image_files)
print('length: C0: {}, LGE: {},  T2: {}'.format(len(c0_image_files), len(lge_image_files), len(t2_image_files)))
for item in tqdm(range(len(c0_image_files))):
    # Load #
    c0_image_file = c0_image_files[item]
    c0_label_file = c0_image_file.replace('images', 'masks')
    c0_label_file = c0_label_file[:-7] + '_manual' + '.nii.gz'
    lge_image_file = c0_image_file.replace('C0', 'LGE')
    lge_label_file = lge_image_file.replace('images', 'masks')
    lge_label_file = lge_label_file[:-7] + '_manual' + '.nii.gz'
    # C0
    c0_image_itk = sitk.ReadImage(c0_image_file)
    c0_label_itk = sitk.ReadImage(c0_label_file)
    c0_image_origin = c0_image_itk.GetOrigin()
    c0_image_spacing = c0_image_itk.GetSpacing()
    c0_image = sitk.GetArrayFromImage(c0_image_itk)
    c0_image = np.flip(c0_image, axis=1)
    c0_label = sitk.GetArrayFromImage(c0_label_itk)
    c0_label = np.flip(c0_label, axis=1)
    # LGE
    lge_image_itk = sitk.ReadImage(lge_image_file)
    lge_label_itk = sitk.ReadImage(lge_label_file)
    lge_image_origin = lge_image_itk.GetOrigin()
    lge_image_spacing = lge_image_itk.GetSpacing()
    lge_image = sitk.GetArrayFromImage(lge_image_itk)
    lge_image = np.flip(lge_image, axis=1)
    lge_label = sitk.GetArrayFromImage(lge_label_itk)
    lge_label = np.flip(lge_label, axis=1)

    # Normalize #
    # C0
    c0_image= (c0_image - c0_image.min()) / (c0_image.max() - c0_image.min())
    c0_label[c0_label == 200] = 2
    c0_label[c0_label == 500] = 3
    c0_label[c0_label == 600] = 1
    # LGE
    lge_image = (lge_image - lge_image.min()) / (lge_image.max() - lge_image.min())
    lge_label[lge_label == 200] = 2
    lge_label[lge_label == 500] = 3
    lge_label[lge_label == 600] = 1
    
    '''
    # Save Slice h5py #
    '''
    # C0
    for slice_idx in range(c0_image.shape[0]):
        image_name = c0_image_file.split(sep)[-1].split('.')[0]
        save_path = data_path.replace('MS_CMR', 'MS_CMR_OriSize_h5py')
        ensure_dir(save_path)
        f = h5py.File(
            save_path + image_name + '_slice_{}.h5'.format(slice_idx), 'w')
        f.create_dataset(
            'image', data=c0_image[slice_idx], compression="gzip")
        f.create_dataset('label', data=c0_label[slice_idx], compression="gzip")
        f.close()
    # lGE
    for slice_idx in range(lge_image.shape[0]):
        image_name = lge_image_file.split(sep)[-1].split('.')[0]
        save_path = data_path.replace('MS_CMR', 'MS_CMR_OriSize_h5py')
        ensure_dir(save_path)
        f = h5py.File(
            save_path + image_name + '_slice_{}.h5'.format(slice_idx), 'w')
        f.create_dataset(
            'image', data=lge_image[slice_idx], compression="gzip")
        f.create_dataset('label', data=lge_label[slice_idx], compression="gzip")
        f.close()

    '''
    # Save Slice h5py #
    '''
    # C0
    image_name = c0_image_file.split(sep)[-1].split('.')[0]
    save_path = data_path.replace('MS_CMR', 'MS_CMR_OriSize_h5py_3D')
    ensure_dir(save_path)
    f = h5py.File(save_path + image_name + '.h5', 'w')
    f.create_dataset('image', data=c0_image, compression="gzip")
    f.create_dataset('label', data=c0_label, compression="gzip")
    f.close()
    # lGE
    image_name = lge_image_file.split(sep)[-1].split('.')[0]
    save_path = data_path.replace('MS_CMR', 'MS_CMR_OriSize_h5py_3D')
    ensure_dir(save_path)
    f = h5py.File(save_path + image_name + '.h5', 'w')
    f.create_dataset('image', data=lge_image, compression="gzip")
    f.create_dataset('label', data=lge_label, compression="gzip")
    f.close()