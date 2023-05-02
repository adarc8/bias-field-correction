import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, folds_path, fold_amounts, valid_fold, validation_flag=False):
        self.image_files = []
        self.folds_path = folds_path
        self.val_flag = validation_flag
        self.fold_amounts = fold_amounts
        self.valid_fold = valid_fold
        train_folds = [fold for fold in range(fold_amounts) if fold != valid_fold]
        if self.val_flag:
            self.list_files = os.listdir(os.path.join(self.folds_path, f'fold{valid_fold}'))
            self.image_files = [f'fold{valid_fold}/' + path for path in self.list_files]
        else:
            self.list_files = [os.listdir(os.path.join(self.folds_path, f'fold{fold_idx}'))
                               for fold_idx in range(fold_amounts) if fold_idx != valid_fold]
            for idx, listdirs in enumerate(self.list_files):
                fold_idx = train_folds[idx]
                if fold_idx == self.valid_fold:
                    fold_idx += 1  # skip it
                for listdir in listdirs:
                    self.image_files.append(f'fold{fold_idx}/' + listdir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_file = self.image_files[index]
        img_path = os.path.join(self.folds_path, img_file)
        image = np.float32(Image.open(img_path))  # (256,256,4)
        image_width = image.shape[-1]
        w = image_width // 4

        dirty = image[:, :w]
        clean_ref = image[:, w:2 * w]
        bias_ref = image[:, 2 * w:3 * w]
        seg = image[:, 3 * w:4 * w]

        dirty = np.float32(dirty / np.max(dirty))
        clean_ref = np.float32(clean_ref / np.max(clean_ref))
        bias_ref = np.float32(bias_ref / np.max(bias_ref))
        seg = seg / 10
        # bias_by_n4itk = np.float32(bias_by_n4itk/np.max(bias_by_n4itk))

        if self.val_flag:
            augmentation = config.augmentation_val(image=dirty,
                                                   image0=clean_ref,
                                                   image1=bias_ref,
                                                   image2=seg)

        else:
            augmentation = config.augmentation_train(image=dirty,
                                                     image0=clean_ref,
                                                     image1=bias_ref,
                                                     image2=seg)

        dirty = augmentation['image']
        clean_ref, bias_ref = augmentation['image0'], augmentation['image1']
        seg = augmentation['image2']

        # dirty = np.expand_dims(dirty,0)
        # clean_ref = np.expand_dims(clean_ref,0)
        # bias_ref = np.expand_dims(bias_ref,0)
        # clean_by_n4itk = np.expand_dims(clean_by_n4itk,0)
        # bias_by_n4itk = np.expand_dims(bias_by_n4itk,0)

        return dirty, clean_ref, bias_ref, seg


if __name__ == "__main__":

    dataset = MyDataset(config.FOLDS_PATHS, fold_amounts=6, valid_fold=0, validation_flag=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for dirty, clean_ref, bias_ref, seg in loader:
        print(dirty.shape)
