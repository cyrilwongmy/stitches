import numpy as np
import sys
import numdifftools as nd
sys.path.append("./src")
import mugrade
from simple_ml import *
try:
    from simple_ml_ext import *
except:
    pass

def test_parse_mnist():
    X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    assert X.dtype == np.float32
    assert y.dtype == np.uint8
    assert X.shape == (60000,784)
    assert y.shape == (60000,)
    np.testing.assert_allclose(np.linalg.norm(X[:10]), 27.892084)
    np.testing.assert_allclose(np.linalg.norm(X[:1000]), 293.0717,
        err_msg="""If you failed this test but not the previous one,
        you are probably normalizing incorrectly. You should normalize
        w.r.t. the whole dataset, _not_ individual images.""", rtol=1e-6)
    np.testing.assert_equal(y[:10], [5, 0, 4, 1, 9, 2, 1, 3, 1, 4])