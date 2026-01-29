import random

from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json
import torchaudio
import torch.nn.functional as F

# import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np

torch.multiprocessing.set_sharing_strategy('file_system')


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms


def get_audio_transforms(size, isize, sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                         n_mels=128, f_min=0, f_max=None, duration=0.3, in_chans=1,
                         mean_train=None, std_train=None):
    mean_train = [0.0] if mean_train is None else mean_train
    std_train = [1.0] if std_train is None else std_train
    if in_chans == 3 and len(mean_train) == 1:
        mean_train = mean_train * 3
        std_train = std_train * 3
    spec_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train, std=std_train)
    ])
    audio_transform = LogMelSpectrogramTransform(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
    )
    target_length = int(sample_rate * duration)
    return audio_transform, spec_transforms, target_length


class LogMelSpectrogramTransform:
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=128,
                 f_min=0, f_max=None):
        if f_max is None:
            f_max = sample_rate // 2
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

    def __call__(self, waveform):
        mel = self.mel(waveform)
        return self.to_db(mel)


def pad_or_trim_waveform(waveform, target_length):
    current_length = waveform.shape[-1]
    if current_length == target_length:
        return waveform
    if current_length > target_length:
        return waveform[..., :target_length]
    pad_amount = target_length - current_length
    return F.pad(waveform, (0, pad_amount))

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class RealIADDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase):
        self.img_path = os.path.join(root, 'realiad_1024', category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase

        json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons', category + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []

        data_set = class_json[phase]
        for sample in data_set:
            self.img_paths.append(os.path.join(root, 'realiad_1024', category, sample['image_path']))
            label = sample['anomaly_class'] != 'OK'
            if label:
                self.gt_paths.append(os.path.join(root, 'realiad_1024', category, sample['mask_path']))
            else:
                self.gt_paths.append(None)
            self.labels.append(label)
            self.types.append(sample['anomaly_class'])

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.types = np.array(self.types)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.phase == 'train':
            return img, label

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class AudioAnomalyDataset(torch.utils.data.Dataset):
    def __init__(self, root, phase, audio_transform, spec_transform, target_length, sample_rate=16000,
                 in_chans=1):
        self.root = root
        self.phase = phase
        self.audio_transform = audio_transform
        self.spec_transform = spec_transform
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.in_chans = in_chans
        self.file_paths, self.labels = self.load_dataset()

    def load_dataset(self):
        file_paths = []
        labels = []
        if self.phase == 'train':
            ok_dir = os.path.join(self.root, 'train', 'ok')
            ok_files = glob.glob(os.path.join(ok_dir, '*.wav'))
            file_paths.extend(ok_files)
            labels.extend([0] * len(ok_files))
        else:
            ok_dir = os.path.join(self.root, 'test', 'ok')
            ng_dir = os.path.join(self.root, 'test', 'ng')
            ok_files = glob.glob(os.path.join(ok_dir, '*.wav'))
            ng_files = glob.glob(os.path.join(ng_dir, '*.wav'))
            file_paths.extend(ok_files)
            labels.extend([0] * len(ok_files))
            file_paths.extend(ng_files)
            labels.extend([1] * len(ng_files))
        return np.array(file_paths), np.array(labels)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = int(self.labels[idx])
        waveform, sr = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = pad_or_trim_waveform(waveform, self.target_length)
        spec = self.audio_transform(waveform)
        spec = self.spec_transform(spec)
        if self.in_chans == 3 and spec.shape[0] == 1:
            spec = spec.repeat(3, 1, 1)
        if self.phase == 'train':
            return spec, label
        gt = torch.zeros([1, spec.size(-2), spec.size(-1)])
        if label == 1:
            gt = torch.ones_like(gt)
        return spec, gt, label, file_path


