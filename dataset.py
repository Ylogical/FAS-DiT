"""
Dataset loaders for FASDiT.

Binary datasets (GlaS / MoNuSeg / PH2 / IMID / TNBC) follow the layout below,
relative to the project root (see README.md):

  processed_glas/ | processed_monuseg/ | processed_ph2/ | processed_imid/
    train/images/*.png   train/masks/*.png
    val/images/*.png     val/masks/*.png
    test/images/*.png    test/masks/*.png

An image and its mask are matched by file name stem, so their extensions may
differ.

Synapse is multi-class (background + 8 organs = 9 classes) and keeps the
TransUNet layout instead:

  Synapse/
    train_npz/caseXXXX_sliceYYY.npz   2D slices, image + label, 512x512
    test_vol_h5/caseXXXX.npy.h5       3D volumes of the 12 test cases
    split.json                        written by prepare_synapse_split.py

Ranges after loading:
    image     -> (3, H, W) float in [-1, 1]
    mask      -> (1, H, W) float in {-1, +1}                        binary
              -> (K, H, W) float in {-1, +1}, signed one-hot        multi-class
    label_idx -> (H, W) int64, the multi-class integer label (for the CE term)
"""
import os
import json
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

try:
    import h5py
except ImportError:                     # only needed for the Synapse test split
    h5py = None


# Dataset root directories (relative to the project root)
_DATASET_ROOT = {
    'glas':    'processed_glas',
    'monuseg': 'processed_monuseg',
    'ph2':     'processed_ph2',
    'imid':    'processed_imid',
    # Evaluation only: target domain of the MoNuSeg -> TNBC transfer experiment
    'tnbc':    'processed_tnbc',
    # Multi-class abdominal CT (TransUNet layout, not processed_*)
    'synapse': 'Synapse',
}

# Mask channels per dataset: 1 for binary, K (incl. background) for multi-class
_DATASET_NUM_CLASSES = {
    'glas': 1, 'monuseg': 1, 'ph2': 1, 'imid': 1, 'tnbc': 1,
    'synapse': 9,
}


def get_num_classes(dataset_name: str) -> int:
    return _DATASET_NUM_CLASSES.get(dataset_name.lower(), 1)


def is_multiclass(dataset_name: str) -> bool:
    return get_num_classes(dataset_name) > 1


def get_dataset_root(dataset_name: str) -> str:
    root = _DATASET_ROOT.get(dataset_name.lower())
    if root is None:
        raise ValueError(f"Unknown dataset: {dataset_name}, "
                         f"supported: {list(_DATASET_ROOT)}")
    return root


class MedicalSegmentationDataset(Dataset):
    """Binary medical image segmentation dataset (image + mask pairs)."""

    def __init__(self, root_dir: str, split: str = 'train', image_size: int = 256):
        """
        Args:
            root_dir  : dataset root, e.g. 'processed_glas'
            split     : 'train' | 'val' | 'test'
            image_size: images and masks are resized to image_size x image_size
        """
        self.root_dir   = root_dir
        self.split      = split
        self.image_size = image_size
        self.is_train   = (split == 'train')

        self.image_dir = os.path.join(root_dir, split, 'images')
        self.mask_dir  = os.path.join(root_dir, split, 'masks')

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: {self.image_dir}\n"
                f"Expected {root_dir}/{{train,val,test}}/{{images,masks}}/ "
                f"- see the 'Data' section of README.md"
            )

        self.filenames = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ])
        if len(self.filenames) == 0:
            raise RuntimeError(f"Empty dataset: {self.image_dir}")

        # Match each image to its mask by file name stem
        _exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        mask_by_stem = {
            os.path.splitext(m)[0]: m
            for m in os.listdir(self.mask_dir) if m.lower().endswith(_exts)
        }
        self.mask_filenames = []
        for f in self.filenames:
            stem = os.path.splitext(f)[0]
            if stem not in mask_by_stem:
                raise FileNotFoundError(
                    f"No mask found for image {f} in {self.mask_dir} (stem '{stem}')"
                )
            self.mask_filenames.append(mask_by_stem[stem])

        self.resize      = transforms.Resize((image_size, image_size))
        self.resize_mask = transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.NEAREST
        )
        self.to_tensor = transforms.ToTensor()

        if self.is_train:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            )

    def __len__(self):
        return len(self.filenames)

    def apply_augmentation(self, image: Image.Image, mask: Image.Image):
        """Synchronized augmentation for image and mask (train split only)."""
        # 1. Random 90-degree rotation
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            image = TF.rotate(image, angle)
            mask  = TF.rotate(mask,  angle)

        # 2. Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        # 3. Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)

        # 4. Random scale + crop (0.8-1.2x)
        if random.random() > 0.3:
            scale    = random.uniform(0.8, 1.2)
            new_size = int(self.image_size * scale)
            image = TF.resize(image, new_size)
            mask  = TF.resize(mask,  new_size,
                              interpolation=transforms.InterpolationMode.NEAREST)
            if scale > 1.0:
                i, j, h, w = transforms.RandomCrop.get_params(
                    image, (self.image_size, self.image_size))
                image = TF.crop(image, i, j, h, w)
                mask  = TF.crop(mask,  i, j, h, w)
            else:
                image = TF.resize(image, self.image_size)
                mask  = TF.resize(mask,  self.image_size,
                                  interpolation=transforms.InterpolationMode.NEAREST)

        # 5. Color jitter (image only)
        if random.random() > 0.3:
            image = self.color_jitter(image)

        # 6. Random Gaussian blur (image only)
        if random.random() > 0.7:
            image = TF.gaussian_blur(image, random.choice([3, 5]))

        return image, mask

    def __getitem__(self, idx):
        fname     = self.filenames[idx]
        img_path  = os.path.join(self.image_dir, fname)
        mask_path = os.path.join(self.mask_dir,  self.mask_filenames[idx])

        image = Image.open(img_path).convert('RGB')
        mask  = Image.open(mask_path).convert('L')

        image = self.resize(image)
        mask  = self.resize_mask(mask)

        if self.is_train:
            image, mask = self.apply_augmentation(image, mask)

        image = self.to_tensor(image)          # (3, H, W) in [0, 1]
        mask  = self.to_tensor(mask)           # (1, H, W) in [0, 1]

        return {
            'image':    image * 2.0 - 1.0,                    # [-1, +1]
            'mask':     (mask > 0.5).float() * 2.0 - 1.0,     # {-1, +1}
            'filename': fname,
        }


class SynapseDataset(Dataset):
    """Synapse multi-organ CT segmentation (background + 8 organs = 9 classes).

    train / val: 2D slices from Synapse/train_npz/, filtered by the case lists
                 in split.json, so slices of one case never straddle the splits.
    test       : the 12 volumes in Synapse/test_vol_h5/, unrolled into 2D slices
                 named "caseXXXX_sliceYYY". Each test sample also carries
                 'label_full', the label at its native 512x512 resolution, so the
                 evaluation can restore the prediction and score case-wise 3D
                 volumes at that resolution (the protocol used by TransUNet and
                 by the diffusion baselines).

    Augmentation matches MedicalSegmentationDataset.apply_augmentation step by
    step (same operations, same probabilities); it runs on tensors instead of
    PIL images because the labels are integer class maps.
    """

    NUM_CLASSES = 9
    CLASS_NAMES = ['background', 'aorta', 'gallbladder', 'kidney_left',
                   'kidney_right', 'liver', 'pancreas', 'spleen', 'stomach']
    # Organ short names used in the reports
    ORGAN_NAMES = {1: 'Aorta', 2: 'GB', 3: 'KL', 4: 'KR',
                   5: 'Liver', 6: 'PC', 7: 'SP', 8: 'SM'}

    # CT window (TransUNet): clip to [-125, 275] HU, then min-max to [0, 1]
    CT_CLIP_MIN = -125.0
    CT_CLIP_MAX = 275.0

    def __init__(self, root_dir: str, split: str = 'train', image_size: int = 224,
                 split_json: str = None, native_label: bool = None,
                 allow_empty: bool = False):
        """
        Args:
            root_dir    : Synapse root (train_npz / test_vol_h5 / split.json)
            split       : 'train' | 'val' | 'test'
            image_size  : network input resolution (224 in the paper)
            split_json  : defaults to <root_dir>/split.json
            native_label: also return the native-resolution label; None turns it
                          on for the test split only
            allow_empty : tolerate an empty split (val is empty when the split
                          was built with --val_cases 0)
        """
        if split == 'test' and h5py is None:
            raise ImportError('The Synapse test split needs h5py (pip install h5py)')

        self.root_dir     = root_dir
        self.split        = split
        self.image_size   = image_size
        self.is_train     = (split == 'train')
        self.native_label = (split == 'test') if native_label is None else native_label

        if split_json is None:
            split_json = os.path.join(root_dir, 'split.json')
        if not os.path.isfile(split_json):
            raise FileNotFoundError(
                f"Synapse split.json not found: {split_json}\n"
                f"Run `python prepare_synapse_split.py` first - see the 'Data' "
                f"section of README.md"
            )
        with open(split_json, 'r', encoding='utf-8') as f:
            split_info = json.load(f)
        if split not in split_info:
            raise ValueError(f"Unknown split: {split} (train | val | test)")

        self.cases = set(split_info[split].get('cases', []))

        # (file path, case id, slice index or None, sample name)
        if split in ('train', 'val'):
            npz_dir = os.path.join(root_dir, 'train_npz')
            if not os.path.isdir(npz_dir):
                raise FileNotFoundError(f"Dataset directory not found: {npz_dir}")
            self.samples = [
                (os.path.join(npz_dir, f), f.split('_')[0], None, f[:-4])
                for f in sorted(os.listdir(npz_dir))
                if f.endswith('.npz') and f.split('_')[0] in self.cases
            ]
        else:
            h5_dir = os.path.join(root_dir, 'test_vol_h5')
            self.samples = []
            for fname in sorted(split_info[split].get('files', [])):
                fpath = os.path.join(h5_dir, fname)
                with h5py.File(fpath, 'r') as f:
                    n_slices = f['image'].shape[0]
                case_id = fname.split('.')[0]
                self.samples += [(fpath, case_id, s, f'{case_id}_slice{s:03d}')
                                 for s in range(n_slices)]

        if len(self.samples) == 0 and not allow_empty:
            raise RuntimeError(
                f"Empty Synapse {split} split - check split.json and the data "
                f"directory (val is empty by design after --val_cases 0)"
            )

        if self.is_train:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            )

    def __len__(self):
        return len(self.samples)

    @property
    def n_cases(self):
        return len({s[1] for s in self.samples})

    @classmethod
    def _normalize_ct(cls, image):
        img = np.clip(image.astype(np.float32), cls.CT_CLIP_MIN, cls.CT_CLIP_MAX)
        return (img - cls.CT_CLIP_MIN) / (cls.CT_CLIP_MAX - cls.CT_CLIP_MIN)

    def _load_sample(self, fpath, slice_idx):
        """Returns (image (H, W) float in [0, 1], label (H, W) int64), native size."""
        if slice_idx is None:
            data  = np.load(fpath)
            image = data['image'].astype(np.float32)
            label = data['label'].astype(np.int64)
        else:
            with h5py.File(fpath, 'r') as f:
                image = f['image'][slice_idx].astype(np.float32)
                label = f['label'][slice_idx].astype(np.int64)
        # The released npz/h5 files are already windowed to [0, 1]; raw HU is not
        if image.max() > 1.5 or image.min() < -0.5:
            image = self._normalize_ct(image)
        return image, label

    @staticmethod
    def _resize_pair(image_t, label_t, size):
        """Bilinear for the image, nearest for the label."""
        if image_t.shape[-2:] == (size, size):
            return image_t, label_t
        image_t = TF.resize(image_t, [size, size],
                            interpolation=transforms.InterpolationMode.BILINEAR,
                            antialias=True)
        label_t = TF.resize(label_t.unsqueeze(0).float(), [size, size],
                            interpolation=transforms.InterpolationMode.NEAREST
                            ).squeeze(0).long()
        return image_t, label_t

    def _geometric_augmentation(self, image_t, label_t):
        """Steps 1-4 of MedicalSegmentationDataset.apply_augmentation.

        image_t: (1, H, W) float in [0, 1]   label_t: (H, W) int64
        """
        H = self.image_size

        # 1. Random 90-degree rotation
        if random.random() > 0.5:
            k = {90: 1, 180: 2, 270: 3}[random.choice([90, 180, 270])]
            image_t = torch.rot90(image_t, k, dims=[1, 2])
            label_t = torch.rot90(label_t, k, dims=[0, 1])

        # 2. Random horizontal flip
        if random.random() > 0.5:
            image_t = torch.flip(image_t, dims=[2])
            label_t = torch.flip(label_t, dims=[1])

        # 3. Random vertical flip
        if random.random() > 0.5:
            image_t = torch.flip(image_t, dims=[1])
            label_t = torch.flip(label_t, dims=[0])

        # 4. Random scale + crop (0.8-1.2x)
        if random.random() > 0.3:
            scale    = random.uniform(0.8, 1.2)
            new_size = int(H * scale)
            image_t, label_t = self._resize_pair(image_t, label_t, new_size)
            if scale > 1.0:
                i, j, h, w = transforms.RandomCrop.get_params(image_t, (H, H))
                image_t = TF.crop(image_t, i, j, h, w)
                label_t = TF.crop(label_t.unsqueeze(0), i, j, h, w).squeeze(0)
            else:
                image_t, label_t = self._resize_pair(image_t, label_t, H)

        return image_t.contiguous(), label_t.contiguous()

    def _photometric_augmentation(self, image_3ch):
        """Steps 5-6, applied after the channel copy and before the [-1, 1] map.

        CT is grayscale, so the three channels are equal and saturation / hue are
        identity here; brightness and contrast are what actually act.
        """
        if random.random() > 0.3:
            image_3ch = self.color_jitter(image_3ch)
        if random.random() > 0.7:
            image_3ch = TF.gaussian_blur(image_3ch, random.choice([3, 5]))
        return image_3ch.clamp(0.0, 1.0)

    def __getitem__(self, idx):
        fpath, case_id, slice_idx, fname = self.samples[idx]
        image, label = self._load_sample(fpath, slice_idx)

        # Native-resolution label, kept before any resizing
        label_native = (torch.from_numpy(label.copy()).long()
                        if self.native_label else None)

        image_t = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0)
        label_t = torch.from_numpy(np.ascontiguousarray(label)).long()
        image_t, label_t = self._resize_pair(image_t, label_t, self.image_size)
        if self.is_train:
            image_t, label_t = self._geometric_augmentation(image_t, label_t)

        # Single CT channel -> 3 channels, so the image encoder is unchanged
        image_3ch = image_t.repeat(3, 1, 1).clamp(0.0, 1.0)
        if self.is_train:
            image_3ch = self._photometric_augmentation(image_3ch)

        label_t = label_t.clamp(0, self.NUM_CLASSES - 1)
        onehot  = torch.nn.functional.one_hot(label_t, self.NUM_CLASSES)
        onehot  = onehot.permute(2, 0, 1).float()       # (K, H, W) in {0, 1}

        out = {
            'image':     image_3ch * 2.0 - 1.0,         # (3, H, W) in [-1, +1]
            'mask':      onehot * 2.0 - 1.0,            # (K, H, W) in {-1, +1}
            'label_idx': label_t,                       # (H, W) int64
            'filename':  fname,
            'case_id':   case_id,
        }
        if label_native is not None:
            out['label_full'] = label_native.clamp(0, self.NUM_CLASSES - 1)
        return out


def synapse_has_val(root_dir: str = None, split_json: str = None) -> bool:
    """Whether split.json defines a non-empty Synapse validation split."""
    if split_json is None:
        split_json = os.path.join(root_dir or _DATASET_ROOT['synapse'], 'split.json')
    try:
        with open(split_json, 'r', encoding='utf-8') as f:
            return len(json.load(f).get('val', {}).get('cases', [])) > 0
    except (OSError, ValueError):
        return False


def _make_dataset(dataset_name: str, split: str, image_size: int):
    """Build one split of dataset_name, binary or multi-class."""
    root = get_dataset_root(dataset_name)
    if is_multiclass(dataset_name):
        return SynapseDataset(root, split, image_size,
                              allow_empty=(split == 'val'))
    return MedicalSegmentationDataset(root, split, image_size)


def _make_loader(dataset, batch_size, shuffle, num_workers, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def get_dataloader(dataset_name: str, batch_size: int = 8, image_size: int = 256,
                   num_workers: int = 4):
    """Train and test DataLoaders. Returns (train_loader, test_loader)."""
    return (
        _make_loader(_make_dataset(dataset_name, 'train', image_size),
                     batch_size, True, num_workers, drop_last=True),
        _make_loader(_make_dataset(dataset_name, 'test', image_size),
                     batch_size, False, num_workers),
    )


def get_dataloader_with_val(dataset_name: str, batch_size: int = 8,
                            image_size: int = 256, num_workers: int = 4):
    """Train / val / test DataLoaders (the splits are fixed on disk).

    Returns (train_loader, val_loader, test_loader). val_loader is None when the
    split defines no validation set, which is the case for Synapse under the
    TransUNet protocol (18 training cases, 12 test cases, no val).
    """
    val_ds = _make_dataset(dataset_name, 'val', image_size)
    return (
        _make_loader(_make_dataset(dataset_name, 'train', image_size),
                     batch_size, True, num_workers, drop_last=True),
        (_make_loader(val_ds, batch_size, False, num_workers)
         if len(val_ds) > 0 else None),
        _make_loader(_make_dataset(dataset_name, 'test', image_size),
                     batch_size, False, num_workers),
    )
