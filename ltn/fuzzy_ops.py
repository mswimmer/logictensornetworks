"""Element-wise fuzzy logic operators for tensorflow."""
from warnings import warn
import tensorflow as tf

# Element-wise fuzzy logic operators for tensorflow.
# Supports traditional NumPy/Tensorflow broadcasting.

# To use in LTN formulas (broadcasting w.r.t. ltn variables appearing in a formula),
# wrap the operator with `ltn.WrapperConnective` or `ltn.WrapperQuantifier`.

EPS = 1e-4

def not_zeros(x):
    """Returns x values slightly shifted away from 0 to avoid numerical instability in some operations."""
    return (1-EPS)*x + EPS


def not_ones(x):
    """Returns x values slightly shifted away from 1 to avoid numerical instability in some operations."""
    return (1-EPS)*x


class Not_Std:
    """Standard negation: Not(x) = 1 - x"""
    def __call__(self,x):
        return 1.-x


class Not_Godel:
    """Godel negation: Not(x) = 1 if x==0, 0 otherwise"""
    def __call__(self,x):
        return tf.cast(tf.equal(x,0),x.dtype)


class And_Min:
    """Minimum t-norm"""
    def __call__(self,x,y):
        return tf.minimum(x,y)


class And_Prod:
    """Product t-norm"""
    def __init__(self,stable=True):
        self.stable = stable

    def __call__(self,x,y,stable=None):
        stable = self.stable if stable is None else stable
        if stable:
            x, y = not_zeros(x), not_zeros(y)
        return tf.multiply(x,y)


class And_Luk:
    """Lukasiewicz t-norm"""
    def __call__(self,x,y):
        return tf.maximum(x+y-1.,0.)


class Or_Max:
    """Maximum s-norm"""
    def __call__(self,x,y):
        return tf.maximum(x,y)


class Or_ProbSum:
    """Probabilistic sum s-norm"""
    def __init__(self,stable=True):
        self.stable = stable

    def __call__(self,x,y,stable=None):
        stable = self.stable if stable is None else stable
        if stable:
            x, y = not_ones(x), not_ones(y)
        return x + y - tf.multiply(x,y)


class Or_Luk:
    """Lukasiewicz s-norm"""
    def __call__(self,x,y):
        return tf.minimum(x+y,1.)


class Implies_KleeneDienes:
    """Kleene-Dienes implication: Implies(x,y) = max(1 - x, y)"""
    def __call__(self,x,y):
        return tf.maximum(1.-x,y)


class Implies_Godel:
    """Godel implication"""
    def __call__(self,x,y):
        return tf.where(tf.less_equal(x,y),tf.ones_like(x),y)


class Implies_Reichenbach:
    """Reichenbach implication"""
    def __init__(self,stable=True):
        self.stable = stable

    def __call__(self,x,y,stable=None):
        stable = self.stable if stable is None else stable
        if stable:
            x, y = not_zeros(x), not_ones(y)
        return 1.-x+tf.multiply(x,y)


class Implies_Goguen:
    """Goguen implication"""
    def __init__(self,stable=True):
        self.stable = stable

    def __call__(self,x,y,stable=None):
        stable = self.stable if stable is None else stable
        if stable:
            x = not_zeros(x)
        return tf.where(tf.less_equal(x,y),tf.ones_like(x),tf.divide(y,x))


class Implies_Luk:
    """Lukasiewicz implication"""
    def __call__(self,x,y):
        return tf.minimum(1.-x+y,1.)


class Equiv:
    """Returns an operator that computes: And(Implies(x,y),Implies(y,x))"""
    def __init__(self, and_op, implies_op):
        self.and_op = and_op
        self.implies_op = implies_op

    def __call__(self, x, y):
        return self.and_op(self.implies_op(x,y), self.implies_op(y,x))


class Aggreg_Min:
    """Minimum aggregation operator"""
    def __call__(self,xs,axis=None,keepdims=False):
        return tf.reduce_min(xs,axis=axis,keepdims=keepdims)


class Aggreg_Max:
    """Maximum aggregation operator"""
    def __call__(self,xs,axis=None,keepdims=False):
        return tf.reduce_max(xs,axis=axis,keepdims=keepdims)


class Aggreg_Mean:
    """Mean aggregation operator"""
    def __call__(self,xs,axis=None,keepdims=False):
        return tf.reduce_mean(xs,axis=axis,keepdims=keepdims)


class Aggreg_pMean:
    """p-mean aggregation operator
    """
    def __init__(self,p=2,stable=True):
        self.p = p
        self.stable = stable

    def __call__(self,xs,axis=None,keepdims=False,p=None,stable=None):
        p = self.p if p is None else p
        stable = self.stable if stable is None else stable
        if stable:
            xs = not_zeros(xs)
        return tf.pow(tf.reduce_mean(tf.pow(xs,p),axis=axis,keepdims=keepdims),1/p)


class Aggreg_pMeanError:
    """p-mean error aggregation operator: 1 - pMean(1 - x)"""
    def __init__(self,p=2,stable=True):
        self.p = p
        self.stable = stable

    def __call__(self,xs,axis=None,keepdims=False,p=None,stable=None):
        p = self.p if p is None else p
        stable = self.stable if stable is None else stable
        if stable:
            xs = not_ones(xs)
        return 1.-tf.pow(tf.reduce_mean(tf.pow(1.-xs,p),axis=axis,keepdims=keepdims),1/p)


class Aggreg_Prod:
    """Product aggregation operator"""
    def __call__(self,xs,axis=None,keepdims=False):
        return tf.reduce_prod(xs,axis=axis,keepdims=keepdims)


class Aggreg_LogProd:
    """Logarithmic Product aggregation operator: sum(log(x))
    Note: outputs values out of the truth value range [0,1].
    Its usage with other connectives could be compromised. Use it carefully.
    """
    def __init__(self,stable=True):
        warn("`Aggreg_LogProd` outputs values out of the truth value range [0,1]. "
             "Its usage with other connectives could be compromised."
             "Use it carefully.", UserWarning)
        self.stable = stable

    def __call__(self,xs,stable=None,axis=None, keepdims=False):
        stable = self.stable if stable is None else stable
        if stable:
            xs=not_zeros(xs)
        return tf.reduce_sum(tf.math.log(xs),axis=axis,keepdims=keepdims)


Aggreg_SumLog = Aggreg_LogProd
