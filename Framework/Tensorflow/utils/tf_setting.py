import tensorflow as tf
from SetWeight import get_weight_variable

class TF_Template(object):

    def __init__(self):
        self.tensorboard_store = []

    def open_session(self, stock_name, **kwargs):
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver()
        if "tensor_dir" in kwargs:
            print("use tensorboard")
            self.open_tensorboard(kwargs["tensor_dir"])
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init)
        self.store = {"epoch": [], "gloss": [], "dloss": []}
        self.i = 0
        self.information = {"epoch": None, "save_png_path": None,
                            "date": None, "split": None, "stcok_name": None,
                            "predict_day": None}
        default_info = {"stock_name": stock_name, "predict_day": self.predict_day}
        self.information.update(default_info)

        #         self.log = loggingmaker(stock_name, f"{self.ResultPath}/log.txt")
        print("open session & init session and then run")

    def run_test(self, list, feed_dict):
        return self.sess.run(list, feed_dict=feed_dict)

    def GradClipMethod(self, loss, optimizer, var, type):
        if type == "grad_clip":
            gvs = optimizer.compute_gradients(loss, var)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var_1) for grad, var_1 in gvs]
            optimizer = optimizer.apply_gradients(capped_gvs)
        elif type == "grad_clip_by_global_norm":
            gradients = tf.gradients(loss, var)
            max_gradient_norm = 5.0
            clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, max_gradient_norm)
            optimizer = optimizer.apply_gradients(zip(clipped_gradients, var))
        elif type == "grad_clip_by_norm":
            gradients, variables = zip(*optimizer.compute_gradients(loss, var))
            gradients = [
                None if gradient is None else tf.clip_by_norm(gradient, 5.0)
                for gradient in gradients]
            optimizer = optimizer.apply_gradients(zip(gradients, variables))
        else:
            optimizer = optimizer.minimize(loss, var_list=var)
        return optimizer

    def Conv1d(self, X, name, weights=[6, 1, 1], bn={}, act=tf.nn.selu, sn=True):
        # W = tf.get_variable(name, weights)
        W = get_weight_variable(weights, name=name , type='he_uniform')
        if sn:
            W = self.spectral_norm(W)
        LAYER1 = tf.nn.conv1d(X, W, 1, "VALID")
        if bn["bool"]:
            LAYER1 = tf.layers. \
                batch_normalization(LAYER1,
                                    center=True, scale=True,
                                    training=bn["is_training"])
        LAYER1 = act(LAYER1)
        return LAYER1

    def UseDropout(self, logit, act, p):
        if act == tf.nn.selu:
            logit = act(logit)
            result = tf.contrib.nn.alpha_dropout(x=logit, keep_prob=p)
        else:
            logit = act(logit)
            result = tf.nn.dropout(logit, keep_prob=p)
        return result

    def FCLayer(self, prev_layer, hiddens, sn=True,
                bias=True, act=tf.nn.selu, name="w", **kwargs):
        logit = prev_layer
        for idx, hidden in enumerate(hiddens):
            if "residual" in kwargs:
                logit = tf.concat([logit, kwargs["residual"]], axis=1)
            logit = self.build_fc(f"{name}_{idx}", logit, hidden, sn=sn, bias=bias)
            if idx == len(hiddens) - 1:
                pass
            else:
                if act is not None:
                    if "keep_prob" in kwargs:
                        logit = self.UseDropout(logit, act, kwargs["keep_prob"])
                    else:
                        logit = act(logit)
        return logit

    def build_fc(self, name, prev, hidden, sn=True, bias=True):
        # w = tf.get_variable(f"fc_w_{name}", [prev.get_shape()[-1], hidden])
        w = get_weight_variable([prev.get_shape()[-1], hidden],
                                name=f"fc_w_{name}", type='he_uniform')
        rank = len(prev.get_shape().as_list())
        if sn:
            w = self.spectral_norm(w)
        if rank > 2:
            logit = tf.tensordot(prev, w, [[rank - 1], [0]])
        else:
            logit = tf.matmul(prev, w)
        if bias:
            b = tf.get_variable(f"fc_b_{name}", [hidden])
            logit = tf.nn.bias_add(logit, b)
        return logit

    def summary_scalar(self, name, value):
        s = tf.summary.scalar(name, value)
        self.tensorboard_store.append(s)

    def summary_scalar_list(self, lists=[[]]):
        for name, value in lists:
            s = tf.summary.scalar(name, value)
            self.tensorboard_store.append(s)

    def summary_hist(self, t_vars):
        for var in t_vars:
            s = tf.summary.histogram(var.op.name, var)
            self.tensorboard_store.append(s)

    @staticmethod
    def early_stopping(epoch, minimum, current, count,
                       threshold=30, start_n=1000):
        if epoch > start_n:
            if min(minimum) < current:
                count += 1
                if count == threshold:
                    raise Exception("Early Stopping")
            else:
                count = 0
        else:
            pass
        return count

    def open_tensorboard(self, suumary_dir):
        self.merged_summary = tf.summary.merge(self.tensorboard_store)
        self.writer = tf.summary.FileWriter(suumary_dir)
        self.writer.add_graph(self.sess.graph)

    #         self.merged_summary = tf.summary.merge_all()

    def spectral_norm(self, w, iteration=1):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])
        u = tf.Variable(tf.random.normal([1, w_shape[-1]]), trainable=False)
        u_hat = u
        v_hat = None
        for i in range(iteration):
            """
            power iteration
            Usually iteration = 1 will be enough
            """
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
