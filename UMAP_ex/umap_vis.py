
import umap.plot
import umap # .umap_ as umap

from category_encoders import OneHotEncoder
from category_encoders import OrdinalEncoder
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from typing import Dict,List
import pandas as pd , re 
import numpy as np

class Empty(object) :
    def __init__(self,) :
        pass
    def fit(self, x) :
        return x
    def transform(self, x) :
        return x
    def fit_transform(self,x) :
        return x
    def inverse_transform(self,x) :
        return x



def auto_type_assign(df:pd.DataFrame) :
    cat_cols = df.sample(10).select_dtypes(include=["object","category"]).columns.tolist()
    num_cols = df.sample(10).select_dtypes(include=["number"]).columns.tolist()
    print("cat_cols : ", cat_cols , "num_cols : " , num_cols)
    return {"num_cols" : num_cols , "cat_cols" : cat_cols}
    
        
class CombineUmap(object) :
    trans_cat_names = None
    numeric_mapper = None
    cat_mapper = None
    intersection_mapper = None 
    union_mapper = None 
    contrast_mapper = None 
    
    def __init__(self, num_info:Dict[str,str], cat_info:Dict[str,str], target:str,**kwargs ) -> None:
        super().__init__()
        self.num_info = num_info
        self.cat_info = cat_info
        self.target = target
        self.assign_num_scaler()
        self.assign_cat_scaler()
    def assign_num_scaler(self,) :
        self.num_method = self.num_info.get("method", None)
        self.num_cols = self.num_info.get("cols", [])
        if self.num_method is None : 
            self.num_scaler = Empty() 
        elif self.num_method == "RobustScaler" :
            self.num_scaler = RobustScaler()
        else :
            raise NotImplementedError("아직 나머지 구현 안함")
    def assign_cat_scaler(self,) :    
        self.cat_method = self.cat_info.get("method", None)
        self.cat_cols = self.cat_info.get("cols", [])
        if self.cat_method is None : 
            self.cat_encoder = Empty() 
        elif self.cat_method == "OrdinalEncoder" :
            self.cat_encoder = OrdinalEncoder(cols = self.cat_cols)
        elif self.cat_method == "OneHotEncoder" :
            self.cat_encoder = OneHotEncoder(cols = self.cat_cols)
        else :
            raise NotImplementedError("아직 나머지 구현 안함")
    def fit(self, df:pd.DataFrame, 
            num_kwargs={"n_neighbors":15, "random_state":42,"n_jobs":20},
            cat_kwargs={"metric":"dice", "n_neighbors" : 150, "random_state" : 42,"n_jobs":20}) :
        if self.num_cols != [] :
            df = self.scale_num(df)
            self.numeric_mapper = umap.UMAP(**num_kwargs).fit(df[self.num_cols])
        if self.cat_cols != [] :
            df = self.encode_cat(df)
            self.cat_mapper = umap.UMAP(**cat_kwargs).fit(df[self.trans_cat_names])
        return self 
    def transform(self,df:pd.DataFrame) :
        result = [None , None]
        if self.num_cols != [] :
            result[0] = self.num_transform(df)
        if self.cat_cols != [] :
            result[1] = self.cat_transform(df)
        return result
    def make_new_mapper(self,) :
        if (self.numeric_mapper is not None) & (self.cat_mapper is not None) :
            self.intersection_mapper = self.numeric_mapper * self.cat_mapper
            self.union_mapper = self.numeric_mapper + self.cat_mapper
            self.contrast_mapper = self.numeric_mapper - self.cat_mapper
            print("make new mapper 1.intersection_mapper 2.union_mapper 3.contrast_mapper")
        return self
    def num_transform(self,df:pd.DataFrame) :
        df = self.scale_num(df)
        return self.numeric_mapper.transform(df[self.num_cols])
    def cat_transform(self, df:pd.DataFrame) :
        df = self.encode_cat(df)
        return self.cat_mapper.transform(df[self.trans_cat_names])
    def vis_basic(self,embedding:np.array,target=None,classes=None) :
        _, ax = plt.subplots(1, figsize=(14, 10))
        plt.scatter(*embedding.T, s=0.3, c=target, cmap='Spectral', alpha=1.0)
        plt.setp(ax, xticks=[], yticks=[])
        if classes is not None :
            num_class = len(classes)
            cbar = plt.colorbar(boundaries=np.arange(num_class+1)-0.5)
            cbar.set_ticks(np.arange(num_class))
            cbar.set_ticklabels(classes)
        plt.title('Embedded via UMAP');
        return self
    def vis_diagnostic(self, mapper,diagnostic_type="vq") :
        """[summary]

        Args:
            mapper ([type]): [description]
            diagnostic_type (str, optional): [description]. Defaults to "vq".

        Returns:
            [type]: [description]
        example 
        diagnostic_type= ["vq","local_dim","neighborhood","pca"],
        """
        return umap.plot.diagnostic(mapper, diagnostic_type=diagnostic_type)
    
    def vis_interactive(self,mapper,**kwargs) :
        umap.plot.output_notebook()
        p = umap.plot.interactive(mapper,**kwargs)
        umap.plot.show(p)
        return self
    
    def vis_connectivity(self,mapper ,**kwargs) :
        #edge_bundling='hammer'
        return umap.plot.connectivity(mapper, show_points=True,**kwargs)
    
    def vis_points(self, mapper, values=None,**kwargs) :
        """[summary]

        Args:
            mapper ([type]): [description]
            values ([type]): [description]
            
        Returns:
            [type]: [description]
        kwargs example
        {"theme" : "fire","background":"black"}
        """
        return umap.plot.points(mapper, values=values,   **kwargs)
        
    def scale_num(self, df:pd.DataFrame)  :
        df[self.num_cols]= self.num_scaler.fit_transform(df[self.num_cols])
        return df 
    
    def encode_cat(self,df:pd.DataFrame)  :
        df = self.cat_encoder.fit_transform(df)
        if self.cat_method == "OneHotEncoder" :
            self.trans_cat_names = self.get_onehot_names()
        elif self.cat_method == "OrdinalEncoder" :
            self.trans_cat_names = self.get_label_names()    
        return df 
    
    def get_label_names(self,) : 
        return self.cat_encoder.cols
    
    def get_onehot_names(self,) :
        onehot_names = [feature_name for feature_name in self.cat_encoder.feature_names if any([ re.search( f"^{i}_", feature_name) for i in self.cat_encoder.cols])]
        return onehot_names 
        
    

        
"""
import seaborn as sns
diamonds = sns.load_dataset('diamonds')
diamonds.head()

cat_cols = diamonds.sample(10).select_dtypes(include=["object","category"]).columns.tolist()
num_cols = diamonds.sample(10).select_dtypes(include=["number"]).columns.tolist()
print(cat_cols , num_cols)

combine_umap = CombineUmap(
    num_info={"method":"RobustScaler","cols":num_cols},
    cat_info = {"method":"OneHotEncoder","cols":cat_cols},
    target="")
combine_umap.fit(diamonds.sample(1000),
num_kwargs={"n_neighbors":15, "random_state":42,"n_jobs":20},
cat_kwargs={"metric":"dice", "n_neighbors" : 30, "random_state" : 42,"n_jobs":20})
## visualize
combine_umap.vis_interactive(combine_umap.cat_mapper)
combine_umap.vis_connectivity(combine_umap.numeric_mapper)
combine_umap.make_new_mapper()
combine_umap.vis_points(combine_umap.intersection_mapper)
combine_umap.vis_connectivity(combine_umap.intersection_mapper)
## transform
num_emb , cat_emb = combine_umap.transform(diamonds.sample(1000))
combine_umap.vis_basic(cat_emb)
"""