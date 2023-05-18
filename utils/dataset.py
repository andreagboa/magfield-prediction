import os
import numpy as np
import torch
import h5py


class MagneticFieldDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, scaling, volume=None):
        super(MagneticFieldDataset, self).__init__()
        self.db_path = datapath
        self.scaling = scaling
        self.volume = volume
        self.size = h5py.File(self.db_path, mode='r')['field'].shape[0]
 
    def open_hdf5(self):
        self.db = h5py.File(self.db_path, mode='r')

    def __getitem__(self, idx):
        if not hasattr(self, 'db'):
            self.open_hdf5()
    
        f = self.db['field'][idx]
        if self.volume is not None:
            f_t = torch.from_numpy(f.astype('float32'))
            grad_z = torch.tensor([0])
        elif len(f.shape) == 4:
            fx_z = np.gradient(f[0], axis=2)[:,:,1]
            fy_z = np.gradient(f[1], axis=2)[:,:,1]
            fz_z = np.gradient(f[2], axis=2)[:,:,1]
            grad_z = np.stack((fx_z, fy_z, fz_z), axis=0)
            grad_z = torch.from_numpy(grad_z.astype('float32'))
            f_t = torch.from_numpy(f[:,:,:,1].astype('float32'))
        else:
            f_t = torch.from_numpy(f.astype('float32'))
            grad_z = torch.tensor([0])

        return f_t * self.scaling, grad_z

    def __len__(self):
        return self.size

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, scaling, image_shape=None, aux=False, with_subfolder=False):
        super(Dataset, self).__init__()        
        self.samples = self._find_samples_in_subfolders(data_path) if with_subfolder \
            else [x for x in os.listdir(data_path)]
        self.data_path = data_path
        self.scaling = scaling
        self.image_shape = image_shape
        self.aux = aux

    def __getitem__(self, index):
        path = os.path.join(self.data_path, self.samples[index])
        img = np.load(path)
        if self.image_shape is not None:
            img = img[:, :self.image_shape[1], :self.image_shape[2]]
        if self.aux:
            img_top = np.load(f'{self.data_path}/../IntermagMMM_aux_top_256/' + self.samples[index])
            img_bottom = np.load(f'{self.data_path}/../IntermagMMM_aux_bottom_256/' + self.samples[index])
            img = np.concatenate((np.expand_dims(img_top, axis=-1), img, np.expand_dims(img_bottom, axis=-1)), axis=-1)
        img = torch.from_numpy(img.astype('float32'))
        img_scaled = img * self.scaling

        return img_scaled

    def _find_samples_in_subfolders(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        samples = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    # item = (path, class_to_idx[target])
                    # samples.append(item)
                    samples.append(path)
        return samples


    def __len__(self):
        return len(self.samples)
