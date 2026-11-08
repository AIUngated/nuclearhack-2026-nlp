"""Fold-local feature cleaning, persisted with every fitted model."""
import numpy as np
from sklearn.impute import SimpleImputer


class FoldPreprocessor:
    def __init__(self,min_prevalence=0,descriptor_clip=None,correlation=None):
        self.min_prevalence=min_prevalence
        self.descriptor_clip=descriptor_clip
        self.correlation=correlation

    def _clip(self,x):
        a=np.asarray(x,np.float32).copy()
        a[~np.isfinite(a)]=np.nan
        if self.descriptor_clip is not None:
            a[:,4096:]=np.clip(a[:,4096:],self.lower_,self.upper_)
        return a

    def fit(self,x):
        x=np.asarray(x,np.float32)
        self.mask_=np.ones(x.shape[1],dtype=bool)
        if self.min_prevalence:
            self.mask_[:4096]=(x[:,:4096]>0).sum(axis=0)>=self.min_prevalence
        if self.descriptor_clip is not None:
            q=self.descriptor_clip
            self.lower_=np.nanquantile(x[:,4096:],q,axis=0)
            self.upper_=np.nanquantile(x[:,4096:],1-q,axis=0)
        self.imputer_=SimpleImputer(strategy='median',keep_empty_features=True).fit(self._clip(x))
        if self.correlation is not None:
            desc=self.imputer_.transform(self._clip(x))[:,4096:]
            variable=np.std(desc,axis=0)>0
            with np.errstate(divide='ignore',invalid='ignore'): corr=np.abs(np.corrcoef(desc,rowvar=False))
            retained=[]
            for i in range(desc.shape[1]):
                if variable[i] and not any(corr[i,j]>self.correlation for j in retained): retained.append(i)
            self.mask_[4096:]=False
            self.mask_[4096+np.asarray(retained,int)]=True
        if self.mask_.sum()==0: raise ValueError('No features survived filtering')
        return self

    def transform(self,x):return self.imputer_.transform(self._clip(x))[:,self.mask_]

    def fit_transform(self,x):return self.fit(x).transform(x)
