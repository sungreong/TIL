import tensorflow as tf
import numpy as np
from copy import deepcopy

def tf_mish(x) :
    return x * tf.nn.tanh(tf.nn.softplus(x))

class Embedding(object) :
    def __init__(self,size, emb_dim, name , emb_istraining= True) :
        self.emb_dim = emb_dim
        self.embeddings = tf.get_variable(name, [size, self.emb_dim], 
                                          trainable= True,dtype=tf.float32)
        tf.add_to_collection("Embedding", self.embeddings)
    def __call__(self, split) :
        Cat = tf.nn.embedding_lookup(self.embeddings, split)
        Cat = tf.reshape(Cat, shape=(-1, self.emb_dim))
        return Cat

def EmbeddingGraph(Cats, id) :
    with tf.variable_scope(f"subemb_{id}") :
        x = tf.concat(Cats, axis =1 )
        X_DIM = x.get_shape().as_list()[1]
        layer = tf.layers.dense(x, 12 , activation= tf.nn.relu)
        layer = tf.nn.dropout(layer , 0.5)
        layer = tf.layers.dense(layer ,12 , activation= tf.nn.relu)
        layer = tf.nn.dropout(layer , 0.5)
        logit = tf.layers.dense(layer ,1 , activation= None)
    return logit

class LayerForEnsemble(object) :
    def __init__(self , fac_var,category_info) : 
        self.fac_var = fac_var
        self.info = category_info
    def run(self, X , select_var, name="ens", type={}) :
        """
        Parameters
        ----------
        X
        select_var
        name
        type = { type , emb_dim , emb_use}
        Returns
        -------
        """
        inputs = []
        Cats = []
        embs = {}
        for idx in select_var :
            split = tf.slice(X, [0, idx], [-1, 1])
            if idx in list(self.info.keys()) :
                split =tf.cast(split, dtype=tf.int32)
                split = tf.reshape(split,(-1,))
                size = max(list(self.info[idx].values())) + 1
                if type["type"] in ["embedding", "group2vec"] :
#                     Cat = tf.keras.layers.Embedding(size,emb_dim,)(split)
#                     Cat = tf.keras.layers.Reshape(target_shape=(emb_dim,))(Cat)
                    div = embedding_n(type, size )
                    emb_name = f"{name}_emb_{idx}"
                    Cat = Embedding(size,div,emb_name)(split)
                    Cats.append(Cat)
                    embs[idx] = Cat
                elif type["type"] == "onehot" :
                    Cat = tf.one_hot(split ,depth=size)
                else :
                    raise Exception(f"No Valid Type : {type}, Please Change the type to onehot or embedding or group2vec")
                if type["type"] == "group2vec":
                    if type["emb_use"] : inputs.append(Cat)
                    else : pass
                else: inputs.append(Cat)
            else :
                inputs.append(split)
        if type["type"] == "group2vec" :
            if len(embs) > 1:
                group_matrix = self.group2vec(embs=embs ,name=name )
                inputs.append(group_matrix)
        x_input = tf.concat(inputs , axis = 1, name=name)
        return x_input , Cats

    def group2vec(self, embs , name = "group2vec"):
        group_key = deepcopy(list(embs.keys()))
        group_vec = []
        if len(group_key) == 2 :
            two_var = list(embs.keys())
            group_matrix = tf.keras.layers.Dot(axes=1)([embs[two_var[0]],
                                                embs[two_var[1]]])
        else :
            for g_k in group_key:
                new_group_key = group_key[:]
                new_group_key.remove(g_k)
                dots = [tf.keras.layers.Dot(axes=1)([embs[k],
                                                     embs[g_k]]) for k in new_group_key]
                dot_product = tf.keras.layers.Average()(dots)
                group_vec.append(dot_product)
            group_matrix = tf.concat(group_vec, axis=1, name=name)
            group_matrix = tf.nn.dropout(group_matrix, 0.5)

        # group_matrix = tf.layers.batch_normalization(group_matrix,
        #                                              center=True,
        #                                              scale=True,
        #                                              training=True)
        return group_matrix


def embedding_n(emb , size) :
    emb_dim , type = emb["emb_dim"] , emb["type"]
    if type == "group2vec" :
        div = emb_dim
    else :
        if emb_dim == size :
            if size == 3 :
                div = 1
            else :
                div = int(emb_dim / 2)
        else :
            div = emb_dim
    return div

class DataRepresenation(object) :
    def __init__(self , fac_var,category_info) :
        self.fac_var = fac_var
        self.info = category_info

    def __call__(self, X , select_var, name="representation", type={}) :
        inputs = []
        embs = {}
        for idx in select_var :
            split = tf.slice(X, [0, idx], [-1, 1])
            if idx in list(self.info.keys()) :
                split =tf.cast(split, dtype=tf.int32)
                split = tf.reshape(split,(-1,))
                size = max(list(self.info[idx].values())) + 1
                # out_of_ck = min(self.info[idx].values())
                if type["type"] in ["embedding", "group2vec"] :
                    # Cat = tf.keras.layers.Embedding(size,emb_dim,)(split)
                    # Cat = tf.keras.layers.Reshape(target_shape=(emb_dim,))(Cat)
                    div = embedding_n(type , size )
                    emb_name = f"{name}_emb_{idx}"
                    Cat = Embedding(size,div,emb_name)(split)
                    embs[idx] = Cat
                elif type["type"] == "onehot" :
                    Cat = tf.one_hot(split ,depth=size)
                else :
                    raise Exception(f"No Valid Type : {type}, Please Change the type to onehot or embedding")
                if type["type"] == "group2vec" : pass
                else : inputs.append(Cat)
            else :
                inputs.append(split)
        if type["type"] == "group2vec" :
            if len(embs) > 1 :
                group_matrix = self.group2vec(embs=embs ,name=name )
                inputs.append(group_matrix)
        x_input = tf.concat(inputs , axis = 1, name=f"final_concat_{name}")
        return x_input

    def group2vec(self, embs , name = "group2vec"):
        group_key = deepcopy(list(embs.keys()))
        group_vec = []
        if len(group_key) == 2:
            two_var = list(embs.keys())
            group_matrix = tf.keras.layers.Dot(axes=1)([embs[two_var[0]],
                                                        embs[two_var[1]]])
        else:
            for g_k in group_key:
                new_group_key = group_key[:]
                new_group_key.remove(g_k)
                dots = [tf.keras.layers.Dot(axes=1)([embs[k],
                                                     embs[g_k]]) for k in new_group_key]
                dot_product = tf.keras.layers.Average()(dots)
                group_vec.append(dot_product)
            group_matrix = tf.concat(group_vec, axis=1, name=name)
            group_matrix = tf.nn.dropout(group_matrix, 0.5)
        # group_matrix = tf.layers.batch_normalization(group_matrix,
        #                                              center=True,
        #                                              scale=True,
        #                                              training=True)
        return group_matrix

def Layer(X , hdims, name , activation,
          bn_istraining,DropoutRate,reuse = False) :
    with tf.variable_scope(f"EnsLayer_{name}",reuse=reuse):
        for idx2 , h_dim in enumerate(hdims) :
            if idx2 == 0 :
                W = tf.get_variable(f"w_{idx2}",shape=[X.get_shape()[1],h_dim])
                B = tf.get_variable(f"b_{idx2}",shape=[h_dim])
                layer = tf.matmul(X, W) + B
            else :
                W = tf.get_variable(f"w_{idx2}",shape=[hdims[idx2-1] ,h_dim ])
                B = tf.get_variable(f"b_{idx2}",shape=[h_dim])    
                layer = tf.matmul(layer, W) + B
            if len(hdims) == idx2 + 1 :
                logit = tf.identity(layer, name="Final_Logit")
            else :
                layer = tf.layers.\
                batch_normalization(layer, center=True, 
                                    scale=True, training=bn_istraining)
                layer = activation(layer)
                layer = tf.nn.dropout(layer, keep_prob=DropoutRate)
    return logit

 
def EnsembleNN(X , hidden = [[],[]], bn_istraining=None,
               DropoutRate=None, Combs=None , RepLayer= None ,
               activate_candidate = None, cat_type = {}) :
    Ensembles = []
    EnsembleCats = []
    with tf.variable_scope("NNEnsemble") :
        for idx , Comb in enumerate(Combs) :
            x_input , Cats = RepLayer.run(X, Comb, f"nnTree_{idx}",cat_type)
            X_DIM = x_input.get_shape().as_list()[1]
            dims = hidden[idx]
            dims = [X_DIM] + dims
            SELECT = np.random.randint(0 , len(activate_candidate) , 1)[0]
            activation = activate_candidate[SELECT]
            LAYER = Layer(x_input , dims, idx , activation,
                          bn_istraining,DropoutRate)
            Ensembles.append(LAYER)
            if Cats != [] :
                EnsembleCats.append(EmbeddingGraph(Cats,idx))
            print(f"No.{idx} nnTree : {str(dims)}")
    return Ensembles , EnsembleCats

def EmbeddingSolver(NEmbeddings, label_y) :
    embedding_solver = []
    sl_not_train = []
    for idx , embedding_logit in enumerate(NEmbeddings) :
        var1 = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            f"NNEnsemble/nnTree_{idx}")
        var2 = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            f"NNEnsemble/subemb_{idx}")
        emb_loss = tf.nn.weighted_cross_entropy_with_logits(
            labels = label_y , logits=embedding_logit,
            pos_weight=1.2) 
        emb_loss = tf.reduce_mean(emb_loss)
        solver2 = tf.train.AdamOptimizer(learning_rate= 1e-5).minimize(emb_loss ,
                                                                       var_list = var1+var2)
        sl_not_train.append(var2)
        embedding_solver.append(solver2)
        embedding_solver.append(emb_loss)
    embedding_solver = embedding_solver[1::2] + embedding_solver[0::2] 
    return embedding_solver , sl_not_train
    
