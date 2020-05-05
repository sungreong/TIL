
import numpy as np
import logging , os
from ray import tune
import ray
from ray.tune.trial import Trial
from ray.tune.logger import DEFAULT_LOGGERS, NoopLogger
import re


# [ i for i in dir(Trial) if re.search("^_", i) is None]
#
# print(Trial.get_trainable_cls())
#
# print([for i in dir(Trial) if re.search("^_", i) is None])
#
# print(Trial.TERMINATED)


import tensorflow as tf
data = np.random.normal(size = (100,2))
true = 5*np.random.normal(size = (100,1))
best_auc = 0.0
log = logging.getLogger('tune')
log.setLevel(logging.DEBUG)
ResultPath = os.getcwd()
fileHandler = logging.FileHandler(
    os.path.join(ResultPath,'tune_log.txt') , mode= "w")
log.addHandler(fileHandler)

class simplemodel(tune.logger.Logger) :
    def __init__(self):
        pass

    def run(self, config) :
        import tensorflow as tf
        tf.logging.set_verbosity(tf.logging.ERROR)
        Trial_ID = Trial.generate_id()
        print(f"ID : {Trial_ID}")
        w = config["node"]
        tf.reset_default_graph()
        X = tf.placeholder(tf.float32,shape=[None,2], name= "X")
        y = tf.placeholder(tf.float32,shape=[None,1], name= "y")
        W = tf.get_variable("w", shape=[2,w],dtype= tf.float32)
        W2 = tf.get_variable("ww", shape=[w,1], dtype = tf.float32)
        layer = tf.matmul(X , W)
        logit = tf.matmul(layer , W2)
        loss = tf.reduce_mean((y-logit)**2)
        optimizer = tf.train.AdamOptimizer(0.005)
        solver = optimizer.minimize(loss ,var_list = tf.trainable_variables() )
        config=tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config = config)
        sess.run(tf.global_variables_initializer())
        for i in range(100) :
            _ , loss2 = sess.run([solver , loss], feed_dict={X:data , y : true})
        tune.track.log(loss=loss2,  done=True)

def running(config) :
    model = simplemodel()
    model.run(config)


def trial_name_string(trial):
    """
    Args:
        trial (Trial): A generated trial object.

    Returns:
        trial_name (str): String representation of Trial.
    """
    #print(f"name : {trial}")
    return str(trial)

class TestLogger(tune.logger.Logger):
    def on_result(self, result):
        print("TestLogger", result)


def trial_str_creator(trial):
    name = "{}_{}_123".format(trial.trainable_name, trial.trial_id)
    print(f"name : {name}")
    return name

if __name__ == '__main__':
    from ray.tune.schedulers import AsyncHyperBandScheduler
    from ray.tune.suggest.hyperopt import HyperOptSearch
    from hyperopt import hp
    space = {
        "node" : hp.randint("node",5,10)
    }
    algo = HyperOptSearch(
            space,
            max_concurrent=4,
            metric="loss",
            mode="min",)
    # points_to_evaluate=current_best_params
    ray.init(local_mode=False)
    sched = AsyncHyperBandScheduler(
        metric="loss",
        mode="min",
        max_t=100,
        grace_period=2)
    analysis = tune.run(
        running,
        config = space ,
        name="tune",
        search_alg= algo,
        stop={
            "loss": 0.0
        },
        trial_name_creator=tune.function(trial_str_creator) ,
        num_samples=10,
        scheduler=sched,
#        loggers=[TestLogger],
        verbose =0
    )

df = analysis.dataframe()
print(df)
print(df)

df.sort_values("loss")
analysis.get_best_config(metric="loss")
#ray.shutdown()