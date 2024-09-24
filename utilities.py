import random as rnd
import numpy as np
import tensorflow as tf

def preset_rng_tf(seed: int = 123) -> None:
    """Preset tensorflow random number generator to given ``seed``."""
    rnd.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
