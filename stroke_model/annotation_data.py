"""Actual input-stroke annotation loader for data inspection and overlap checks."""
import numpy as np
import torch
from torch.utils.data import Dataset


class ActualStrokeDataset(Dataset):
    def __init__(self,rows,size,shuffle_strokes=False,reference_jitter=0):
        self.rows,self.size=rows,size
        self.shuffle_strokes=shuffle_strokes
        self.reference_jitter=reference_jitter
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        with np.load(self.rows[index]['path'],allow_pickle=False) as arrays:
            if 'target_masks' not in arrays:
                raise ValueError('Need ACTUAL input target_masks; exclusive labels/template strokes cannot recover overlap annotations')
            image,reference,strokes,target=[torch.tensor(arrays[k].copy(),dtype=torch.float32)
                                           for k in ('image','reference','strokes','target_masks')]
            valid=torch.tensor(arrays['valid_pixels'].copy(),dtype=torch.float32) if 'valid_pixels' in arrays else torch.ones_like(image)
        if image.shape!=(1,self.size,self.size) or reference.shape!=image.shape or valid.shape!=image.shape:
            raise ValueError('image/reference/valid_pixels must be 1,S,S')
        if strokes.ndim!=3 or strokes.shape[-2:]!=image.shape[-2:] or not len(strokes) or target.shape!=strokes.shape:
            raise ValueError('strokes and actual target_masks must be matching K,S,S')
        for value in (image,reference,strokes,target,valid):
            if not torch.isfinite(value).all() or (value<0).any() or (value>1).any():raise ValueError('Arrays must be ink [0,1]')
        if not (strokes.flatten(1).amax(1)>.5).all():raise ValueError('Unpadded expected reference slots must be nonempty')
        if valid.sum()==0:raise ValueError('Need supervised pixels')
        if self.reference_jitter:
            dy,dx=torch.randint(-self.reference_jitter,self.reference_jitter+1,(2,)).tolist()
            def shift(value):
                result=torch.zeros_like(value)
                y0,y1=max(0,dy),self.size+min(0,dy)
                x0,x1=max(0,dx),self.size+min(0,dx)
                result[...,y0:y1,x0:x1]=value[...,y0-dy:y1-dy,x0-dx:x1-dx]
                return result
            moved=shift(strokes)
            if (moved.flatten(1).amax(1)>.5).all():
                reference,strokes=shift(reference),moved
        if self.shuffle_strokes:
            order=torch.randperm(len(strokes));strokes,target=strokes[order],target[order]
        return image,reference,strokes,target,valid
