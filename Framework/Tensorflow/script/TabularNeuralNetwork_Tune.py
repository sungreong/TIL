class LayerUtils:
    import tensorflow as tf
    def tf_mish(self, x):
        import tensorflow as tf
        return x * tf.nn.tanh(tf.nn.softplus(x))

    def spectral_norm(self, w, iteration=1, name=None):
        import tensorflow as tf
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])
        u = tf.get_variable(name, [1, w_shape[-1]],
                            initializer=tf.random_normal_initializer(), trainable=False)
        u_hat = u
        v_hat = None
        for i in range(iteration):
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = tf.nn.l2_normalize(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = tf.nn.l2_normalize(u_)

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, w_shape)
        return w_norm

    def LayerByType(self, layer, Type, activation):
        import tensorflow as tf
        if Type == "Self_Normal":
            layer = activation(layer)
            layer = tf.contrib.nn.alpha_dropout(layer, 0.8)
        elif Type == "Batch_Normalization":
            layer = tf.contrib.layers.batch_norm(layer,
                                                 center=True, scale=True,
                                                 is_training=True,  # phase
                                                 scope='bn')
        elif Type == "Instance_Normalization":
            layer = tf.contrib.layers.instance_norm(layer)
        else:
            pass
        if Type == "Self_Normal":
            pass
        else:
            layer = activation(layer)
        return layer

    def UseBias(self, input, W, usebias, shape, init):
        import tensorflow as tf
        layer = tf.matmul(input, W)
        if usebias:
            bias = tf.get_variable("Bias",
                                   shape=[shape[1]], dtype=tf.float32,
                                   initializer=init)
            # tf.constant_initializer(0.0)
            layer += bias
        else:
            pass
        return layer

    def select_init(self, activation):
        import tensorflow as tf
        select_w_init = np.random.randint(0, 2, size=1)[0]
        seed_n = np.random.randint(1, 2000, size=1)[0]
        relu_w_init = [tf.keras.initializers.he_uniform(seed=seed_n),
                       tf.keras.initializers.he_normal(seed=seed_n)][select_w_init]
        tanh_w_init = [tf.keras.initializers.glorot_normal(seed=seed_n),
                       tf.keras.initializers.glorot_uniform(seed=seed_n)][select_w_init]
        s_elu_w_init = [tf.keras.initializers.lecun_normal(seed=seed_n),
                        tf.keras.initializers.lecun_uniform(seed=seed_n)][select_w_init]
        nomal_w_init = tf.keras.initializers.truncated_normal(seed=seed_n)
        if activation in [tf.nn.leaky_relu, tf.nn.relu]:
            init = relu_w_init
        elif activation in [tf.nn.tanh, tf.nn.softmax]:
            init = tanh_w_init
        elif activation in [tf.nn.selu, tf.nn.elu, self.tf_mish]:
            init = s_elu_w_init
        else:
            init = nomal_w_init
        return init

    def fully_connected_layer(self, input, shape=None,
                              name=None, activation=tf.nn.leaky_relu,
                              usebias=True, final=False,
                              SN=True, Type=None):
        import tensorflow as tf
        with tf.variable_scope(name):
            input_size = input.get_shape().as_list()[1]
            shape = [input_size, shape]
            init = self.select_init(activation)
            W1 = tf.get_variable(f"Weight", dtype=tf.float32,
                                 shape=shape, initializer=init)
            tf.add_to_collection('weight_variables', W1)
            if SN:
                W1 = self.spectral_norm(W1, name=f"SN_Weight")
            if final:
                layer = self.UseBias(input=input, W=W1, usebias=usebias, shape=shape,
                                     init=tf.constant_initializer(0.0))
            else:
                layer = self.UseBias(input=input, W=W1, usebias=usebias,
                                     shape=shape, init=tf.constant_initializer(0.0))
                layer = self.LayerByType(layer, Type, activation)
            #             print(f"{name:10} output : {layer.get_shape().as_list()}")
            return layer


class Utils(LayerUtils):
    def emb(self, split=None, _len_=None, n=None, ratio=None, key=None):
        import tensorflow as tf
        #         print(f"Category : {key}")
        split = tf.reshape(split, shape=(-1,))
        split = tf.to_int32(split)
        if _len_ < n:
            first = _len_
            to = _len_
            Cat = tf.one_hot(split, depth=_len_)
        #             print(f"[{key}] Onehot Shape : [{first}]")
        else:
            first = _len_
            to = int(_len_ / 2)
            # 2/_len_
            embeddings = tf.Variable(tf.truncated_normal([first, to],
                                                         stddev=0.1),
                                     dtype=tf.float32, name=key)
            mod = sys.modules[__name__]
            setattr(mod, f'embedding_{key}', embeddings)
            Cat = tf.nn.embedding_lookup(embeddings, split)
            Cat = tf.nn.dropout(Cat, ratio)
        #             print(f"[{key}] Onehot Shape : [{first}] --> Embedding Shape : [{to}] ")
        return Cat

    def EmbeddingLayer(self, X, objdict, inputratio, totalcol):
        import tensorflow as tf
        inputs = []
        for idx, key in enumerate(totalcol):
            split = tf.slice(X, [0, idx], [-1, 1])
            if idx in objdict:
                category_n = objdict[idx]
                Cat = self.emb(split, category_n, 4, inputratio, key)
                inputs.append(Cat)
            else:
                inputs.append(split)
        concatenated_layer = tf.concat(inputs, axis=1, name='concatenate')
        return concatenated_layer

    def tf_feature(self, objcol, objdict, InputdropoutRate, totalcol):
        import tensorflow as tf
        with tf.name_scope(f"FeatureX"):
            if objcol == []:
                featureX = tf.nn.dropout(self.X, InputdropoutRate)
            else:
                featureX = self.EmbeddingLayer(self.X, objdict, InputdropoutRate, totalcol)
        return featureX


class TabularNN(Utils, LayerUtils):
    def __init__(self, total_col, cat_col, objdict, target_dim, classbalanced, log):
        import tensorflow as tf
        tf.reset_default_graph()
        self.batch_size = tf.placeholder(tf.int64, name="Batchsize")
        self.X = tf.placeholder(tf.float32, shape=[None, len(total_col)])
        self.y = tf.placeholder(tf.float32, [None])
        self.regularizer = tf.placeholder(tf.float32, name="regularizer")
        self.dropoutRate = tf.placeholder(tf.float32, name="dropoutRate")
        self.InputdropoutRate = tf.placeholder(tf.float32, name="InputdropoutRate")
        self.cat_col = cat_col
        self.objdict = objdict
        self.total_col = total_col
        self.target_dim = target_dim
        self.class_balanced_number = classbalanced
        log = logging.getLogger('tune')
        log.setLevel(logging.DEBUG)
        fileHandler = logging.FileHandler(
            os.path.join(ResultPath, 'tune_log.txt'))
        log.addHandler(fileHandler)
        self.log = log

    def Network(self, X, dims, dropoutRate, activation):
        import tensorflow as tf
        with tf.variable_scope(f"Network", reuse=tf.AUTO_REUSE):
            for idx, h_dim in enumerate(dims):
                if len(dims) == idx + 1:
                    Layer = self.fully_connected_layer(X, shape=h_dim,
                                                       name=f"Weight{idx}",
                                                       activation=activation,
                                                       usebias=True, final=True,
                                                       SN=True, Type="Self_Normal")
                else:
                    Layer = self.fully_connected_layer(X, shape=h_dim,
                                                       name=f"Weight{idx}",
                                                       activation=activation,
                                                       usebias=True, final=False,
                                                       SN=True, Type="Self_Normal")
        return Layer

    def create_model(self, config):
        import tensorflow as tf
        nodes_1 = config["nodes_1"]
        nodes_2 = config["nodes_2"]
        nodes_3 = config["nodes_3"]
        activate = \
            [tf.nn.selu, self.tf_mish, tf.nn.leaky_relu, tf.nn.elu, tf.nn.relu]
        activation_candidate = ["selu", "mish", "leaky_relu", "elu", "relu"]
        activation_dict = dict(zip(activation_candidate, activate))
        selected = config["activate_candidate"]
        activation = activation_dict[selected]
        lr = config["lr"]
        alpha = config["alpha"]
        str0 = f"Trail : {self.Trial_ID} \n"
        str1 = \
            f"nodes_1 : {nodes_1}, nodes_2 : {nodes_2}, nodes_3 : {nodes_3} \n"
        str2 = \
            f"activation : {selected}, smoothLabel : {alpha}, learning_rate :{lr}"
        self.log.info(str1 + str2)
        TransformX = self.tf_feature(self.cat_col, self.objdict, self.InputdropoutRate, self.total_col)
        dims = [nodes_1, nodes_2, nodes_3, self.target_dim]
        Logit = self.Network(X=TransformX,
                             dims=dims,
                             dropoutRate=self.dropoutRate,
                             activation=activation)
        self.Probs = tf.nn.softmax(Logit)
        y_one_hot = tf.one_hot(tf.cast(self.y, tf.int32), depth=self.target_dim)
        weight = tf.constant([self.class_balanced_number])  #
        WCE = tf.nn.weighted_cross_entropy_with_logits(targets=y_one_hot,
                                                       logits=Logit,
                                                       pos_weight=weight)
        self.Loss = tf.reduce_mean(WCE)
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f"Network")
        WEIGHTS = tf.get_collection("weight_variables")
        L2 = []
        for v in WEIGHTS:
            L2.append(tf.nn.l2_loss(v))
            tf.summary.histogram(v.name, v)
        self.Loss += tf.add_n(L2) * self.regularizer
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, WEIGHTS)
        self.Loss += self.regularizer * regularization_penalty
        kwargs = {}
        kwargs['learning_rate'] = lr
        optimizer = tf.train.AdamOptimizer(**kwargs)
        self.solver = optimizer.minimize(self.Loss, var_list=vars)
        prediction = tf.argmax(self.Probs, 1)
        correct = tf.argmax(y_one_hot, 1)
        equality = tf.equal(prediction, correct)
        accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
        tf.summary.scalar(f"Accuracy", accuracy)
        tf.summary.scalar(f"Loss", self.Loss)
        tf.summary.histogram(f"Probability", self.Probs)
        return print("Crate Model")

    def Projector(self, train_writer, cat_col, LabelEncoding, Trial_ID):
        import tensorflow.contrib.tensorboard.plugins.projector as projector
        import tensorflow as tf
        config = projector.ProjectorConfig()
        mod = sys.modules[__name__]
        for cat in cat_col:
            if len(LabelEncoding[cat].classes_) < 5:
                continue
            index2word_map = dict(zip(np.arange(len(LabelEncoding[cat].classes_)).tolist(),
                                      LabelEncoding[cat].classes_))
            metadata_file = os.path.join(ResultPath, f'train', Trial_ID,
                                         f'metadata_{cat}.tsv')
            with open(metadata_file, "w") as metadata:
                metadata.write('Name\tClass\n')
                for k, v in index2word_map.items():
                    metadata.write('%s\t%d\n' % (v, k))
            embedding = config.embeddings.add()
            embedding.tensor_name = getattr(mod, f'embedding_{cat}').name
            # Link this tensor to its metadata file (e.g. labels).
            embedding.metadata_path = metadata_file
        projector.visualize_embeddings(train_writer, config)

    def regularizer_control(self, epoch):
        if epoch < 100:
            regularizer_rate = 1e-5
        elif epoch < 200:
            regularizer_rate = 1e-4
        elif epoch < 400:
            regularizer_rate = 1e-3
        elif epoch < 800:
            regularizer_rate = 1e-2
        else:
            regularizer_rate = 1e-1
        return regularizer_rate

    def train(self, LabelEncoding, Train, Valid, mb_size, Epoch, config):
        import tensorflow as tf
        tf.logging.set_verbosity(tf.logging.ERROR)
        self.Trial_ID = Trial.generate_id()
        Train_X, Train_y = Train
        Test_X, Test_y = Valid
        self.create_model(config)
        merged = tf.summary.merge_all()
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        train_writer = tf.summary.FileWriter(
            os.path.join(ResultPath, f'train', self.Trial_ID), sess.graph)
        saver = tf.train.Saver()
        self.Projector(train_writer, self.cat_col, LabelEncoding, self.Trial_ID)
        sess.run(tf.global_variables_initializer())
        _Loss_, _Epoch_ = [], []
        _Epoch2_, _trAUC_, _teAUC_ = [0], [0], [0]

        for epoch in range(Epoch):
            regularizer_rate = self.regularizer_control(epoch)
            idx = list(np.random.permutation(len(Train_X)))
            XX = Train_X.iloc[idx, :].values
            YY = Train_y[idx]
            batch_iter = int(len(XX) / mb_size)
            _Loss2_ = []
            for idx in range(batch_iter):
                X_mb = XX[idx * mb_size:(idx + 1) * mb_size]
                Y_mb = YY[idx * mb_size:(idx + 1) * mb_size]
                Feed = {self.X: X_mb,
                        self.y: Y_mb,
                        self.regularizer: regularizer_rate,
                        self.InputdropoutRate: 1.0,
                        self.dropoutRate: 1.0}
                _, LOSS = sess.run([self.solver, self.Loss], feed_dict=Feed)
                _Loss2_.append(LOSS)
            _Loss_.append(np.mean(_Loss2_))
            _Epoch_.append(epoch)
            if epoch % 100 == 0:
                if epoch == 0:
                    meta_graph_bool = True
                else:
                    meta_graph_bool = False
                saver.save(sess,
                           os.path.join(ResultPath, f'train', self.Trial_ID, "model.ckpt"),
                           global_step=epoch, write_meta_graph=meta_graph_bool)
                Feed = {self.X: Test_X.values,
                        self.y: Test_y.values,
                        self.regularizer: regularizer_rate,
                        self.InputdropoutRate: 1.0,
                        self.dropoutRate: 1.0,
                        }
                try:
                    summary = sess.run(merged, feed_dict=Feed)
                    train_writer.add_summary(summary, epoch)
                except Exception as e:
                    self.log.error(e)
        train_writer.close()
        Feed = {self.X: Test_X.values,
                self.InputdropoutRate: 1.0,
                self.dropoutRate: 1.0
                }
        te_probs = sess.run(self.Probs, feed_dict=Feed)
        te_real_target = np.squeeze(Test_y.values)
        AUC = roc_auc_score(te_real_target, te_probs[:, 1])
        global best_auc
        if best_auc < AUC:
            self.log.info("{} : {} < {}".format(self.Trial_ID, best_auc, AUC))
            best_auc = AUC

        tune.track.log(auc=AUC, done=False)
        clear_output()


init_candidate =\
["xavier_uniform","xavier_normal", "he_normal", "he_uniform","caffe_uniform"]
activation_dict = ["selu","mish","leaky_relu","elu","relu"]
best_auc = 0.0
log = logging.getLogger('tune')
log.setLevel(logging.DEBUG)
fileHandler = logging.FileHandler(
    os.path.join(ResultPath,'tune_log.txt') , mode= "w")
log.addHandler(fileHandler)


def train_tune(config) :
    import tensorflow as tf
    print(config)
    TNN = TabularNN(total_col=totalcol ,
          cat_col = cat_col ,  objdict = objdict,
          target_dim = 2 , classbalanced = balanced,
          log = log)
    mb_size = 2000
    TNN.train(LabelEncoding, (Train_X,Train_y) , (Test_X , Test_y),
              mb_size , 100, config)


ray.shutdown()
if __name__ == '__main__':
    logging.getLogger().info("Start optimization.")
    from ray.tune.schedulers import AsyncHyperBandScheduler
    from ray.tune.suggest.hyperopt import HyperOptSearch
    from hyperopt import hp
    from ray.tune.logger import DEFAULT_LOGGERS, NoopLogger
    ray.init()
    space = {
    "nodes_1" : hp.randint("nodes_1",50,100),
    "nodes_2" : hp.randint("nodes_2",15,50),
    "nodes_3" : hp.randint("nodes_3",5,15),
    "init_candidate" :  hp.choice("init_candidate", init_candidate),
    "activate_candidate" : hp.choice("activate_candidate", activation_dict),
    "alpha" : hp.uniform("alpha", 0.7,1.0),
    "lr" : hp.loguniform("lr", 1e-5, 1e-1)}
    algo = HyperOptSearch(
            space,
            max_concurrent=2,
            metric="auc",
            mode="max",)
    # points_to_evaluate=current_best_params
    sched = AsyncHyperBandScheduler(
        metric="auc",
        mode="max",
        max_t=100,
        grace_period=2)
    analysis = tune.run(
        train_tune,
        config = space ,
        name="tune",
        search_alg= algo,
        stop={
            "auc": 0.99
        },
        num_samples=100,
        resources_per_trial={
            "cpu": 2,
            "gpu": 0},
        scheduler=sched,
        loggers=[NoopLogger]
    )
