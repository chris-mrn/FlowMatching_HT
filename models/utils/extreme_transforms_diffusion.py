import torch
import torch.nn as nn
from torch.nn.functional import softplus, relu, sigmoid
# from nflows.transforms.autoregressive import AutoregressiveTransform
# from nflows.transforms import made as made_module
# from nflows.transforms import Transform
# from tailnflows.models.utils import inv_sftplus, inv_sigmoid
from typing import TypedDict, Optional, Callable
from math import sqrt


def inv_sftplus(x):
    return x + torch.log(-torch.expm1(-x))


def inv_sigmoid(x):
    return torch.log(x) - torch.log(1 - x)


MAX_TAIL = 5.0
LOW_TAIL_INIT = 0.1
HIGH_TAIL_INIT = 0.9
SQRT_2 = sqrt(2.0)
SQRT_PI = sqrt(torch.pi)
MIN_ERFC_INV = 1e-6
PI = torch.pi


class NNKwargs(TypedDict, total=False):
    hidden_features: int
    num_blocks: int
    use_residual_blocks: bool
    random_mask: bool
    activation: Callable
    dropout_probability: float
    use_batch_norm: bool


class SpecifiedNNKwargs(TypedDict, total=True):
    hidden_features: int
    num_blocks: int
    use_residual_blocks: bool
    random_mask: bool
    activation: Callable
    dropout_probability: float
    use_batch_norm: bool


def configure_nn(nn_kwargs: NNKwargs) -> SpecifiedNNKwargs:
    return {
        "hidden_features": nn_kwargs.get("hidden_features", 5),
        "num_blocks": nn_kwargs.get("num_blocks", 2),
        "use_residual_blocks": nn_kwargs.get("use_residual_blocks", True),
        "random_mask": nn_kwargs.get("random_mask", False),
        "activation": nn_kwargs.get("activation", relu),
        "dropout_probability": nn_kwargs.get("dropout_probability", 0.0),
        "use_batch_norm": nn_kwargs.get("use_batch_norm", False),
    }






def _erfcinv(x):
    with torch.no_grad():
        x = torch.clamp(x, min=MIN_ERFC_INV)
    return -torch.special.ndtri(0.5 * x) / SQRT_2


def _small_erfcinv(log_g):
    """
    Use series expansion for erfcinv(x) as x->0
    """
    log_z_sq = 2 * log_g

    inner = torch.log(torch.tensor(2 / PI)) - log_z_sq
    inner -= (torch.log(torch.tensor(2 / PI)) - log_z_sq).log()

    z = inner.sqrt() / SQRT_2

    return z


def _stable_erfcinv(x, log_x):
    x = torch.clamp(x, min=MIN_ERFC_INV)  # Or any small positive value
    return -torch.special.ndtri(0.5 * x) / torch.sqrt(torch.tensor(2.0))





    # with torch.no_grad():
    #     standard_x = torch.clamp(x, min=MIN_ERFC_INV, max=None)
    #     small_log_x = torch.clamp(log_x, min=None, max=torch.tensor(MIN_ERFC_INV, device=log_x.device).log())

    # return torch.where(
    #     x > MIN_ERFC_INV,
    #     -torch.special.ndtri(0.5 * standard_x) / SQRT_2,
    #     _small_erfcinv(small_log_x),
    # )


def _shift_power_transform_and_lad(z, tail_param):
    transformed = (SQRT_2 / SQRT_PI) * (torch.pow(1 + z / tail_param, tail_param) - 1)
    lad = (tail_param - 1) * torch.log(1 + z / tail_param)
    lad += torch.log(torch.tensor(SQRT_2 / SQRT_PI))
    return transformed, lad


def _shift_power_inverse_and_lad(x, tail_param):
    transformed = (
        (SQRT_PI / SQRT_2) * tail_param * (torch.pow(1 + x, 1 / tail_param) - 1)
    )
    lad = ((1 / tail_param) - 1) * torch.log(1 + x)
    lad -= torch.log(torch.tensor(SQRT_2 / SQRT_PI))
    return transformed, lad


def _extreme_transform_and_lad(z, tail_param):
    g = torch.erfc(z / SQRT_2)+0.000001 #Aditya has added the extra 0.000..1 to handle zero power negative in the next line
    x = (torch.pow(g, -tail_param) - 1) / tail_param

    lad = torch.log(g) * (-tail_param - 1)
    lad -= 0.5 * torch.square(z)
    lad += torch.log(torch.tensor(SQRT_2 / SQRT_PI))

    return x, lad


def _extreme_inverse_and_lad(x, tail_param):
    inner = 1 + tail_param * x
    g = torch.pow(inner, -1 / tail_param)
    # print("g",g)
    log_g = -torch.log(inner) / tail_param
    erfcinv_val = _stable_erfcinv(g, log_g)
    # print(erfcinv_val,"erfcinv_val")

    z = SQRT_2 * erfcinv_val

    lad = (-1 - 1 / tail_param) * torch.log(inner)
    lad += torch.square(erfcinv_val)
    lad += torch.log(torch.tensor(SQRT_PI / SQRT_2))

    return z, lad


def neg_extreme_transform_and_lad(z, tail_param):
    def _small_erfcinv(log_z):
        inner = torch.log(torch.tensor(2 / torch.pi)) - 2 * log_z
        inner -= (torch.log(torch.tensor(2 / torch.pi)) - 2 * log_z).log()
        return inner.pow(0.5) / SQRT_2

    erfc_val = torch.erfc(z / SQRT_2)
    g = erfc_val.pow(-tail_param)

    stable_g = g > MIN_ERFC_INV

    erfcinv_val = torch.zeros_like(z)
    erfcinv_val[stable_g] = _erfcinv(g[stable_g])

    log_z = -torch.log(z[~stable_g])
    log_z += -z[~stable_g].square() / 2
    log_z += torch.log(torch.tensor(SQRT_2 / SQRT_PI))
    log_z *= -tail_param[~stable_g]

    erfcinv_val[~stable_g] = _small_erfcinv(log_z)

    x = -erfcinv_val * 2 / (SQRT_PI * tail_param)

    lad = torch.square(erfcinv_val) - 0.5 * torch.square(z)
    lad += torch.log(torch.tensor(SQRT_2 / SQRT_PI))
    lad += (-1 - tail_param) * torch.log(erfc_val)

    return x, lad


def _tail_switch_transform(z, pos_tail, neg_tail, shift, scale):
    sign = torch.sign(z)
    tail_param = torch.where(z > 0, pos_tail, neg_tail)
    heavy_tail = tail_param > 0
    heavy_x, heavy_lad = _extreme_transform_and_lad(
        torch.abs(z[heavy_tail]), tail_param[heavy_tail]
    )
    light_x, light_lad = neg_extreme_transform_and_lad(
        torch.abs(z[~heavy_tail]), tail_param[~heavy_tail]
    )

    lad = torch.zeros_like(z)
    x = torch.zeros_like(z)

    x[heavy_tail] = heavy_x
    x[~heavy_tail] = light_x

    lad[heavy_tail] = heavy_lad
    lad[~heavy_tail] = light_lad

    lad += torch.log(scale)
    return sign * x * scale + shift, lad


def _tail_affine_transform(z, pos_tail, neg_tail, shift, scale):
    sign = torch.sign(z)
    tail_param = torch.where(z > 0, pos_tail, neg_tail)
    x, lad = _extreme_transform_and_lad(torch.abs(z), tail_param)
    lad += torch.log(scale)
    return sign * x * scale + shift, lad

def dTTF_dz(z, pos_tail, neg_tail,shift,scale): #aditya wrote this
    sqrt_2 = SQRT_2
    sqrt_2_over_pi = SQRT_2/SQRT_PI

    s = torch.sign(z)
    lambda_s = torch.where(s > 0, pos_tail, neg_tail)

    exp_term = torch.exp(-z**2 / 2)
    erfc_term = torch.erfc(torch.abs(z) / sqrt_2)+0.000001 #Aditya has added the extra 0.000..1 to handle zero power negative in the next line
    power_term = erfc_term ** (-lambda_s - 1)

    return scale * sqrt_2_over_pi * exp_term * power_term

def dTTF_dz_eval(z, pos_tail, neg_tail,shift,scale): #aditya wrote this
    sqrt_2 = SQRT_2
    sqrt_2_over_pi = SQRT_2/SQRT_PI

    s = torch.sign(z)
    lambda_s = torch.where(s > 0, pos_tail, neg_tail)

    exp_term = torch.exp(-z**2 / 2)
    erfc_term = torch.erfc(torch.abs(z) / sqrt_2)+0.000001 #Aditya has added the extra 0.000..1 to handle zero power negative in the next line
    power_term = erfc_term ** (-lambda_s - 1)

    return scale * sqrt_2_over_pi * exp_term * power_term

def dTTFInverse_dz(x, pos_tail, neg_tail,shift, scale): #aditya wrote this
    s = torch.sign(x - shift)
    
    # Compute λₛ based on sign
    lambda_s = torch.where(s > 0, pos_tail, neg_tail)
    
    # Compute y = λₛ|(x - μ)/σ| + 1
    y = lambda_s * torch.abs((x - shift) / scale) + 1
    
    # Compute y^{-1/λₛ - 1}
    y_pow = torch.pow(y, -1.0/lambda_s - 1)
    
    # Compute erfc^{-1}(y^{-1/λₛ})
    # Note: PyTorch doesn't have direct erfc inverse, so we use inverse of erf and adjust
    # erfc(z) = 1 - erf(z) => erfc^{-1}(w) = erf^{-1}(1 - w)
    w = torch.pow(y, -1.0/lambda_s)
    erfcinv_w =_stable_erfcinv(w, torch.log(w)) #torch.erfinv(1 - w)
    
    # Compute exp(erfcinv_w^2)
    exp_term = torch.exp(torch.square(erfcinv_w))
    
    # Combine all terms
    grad = (1 / scale) * (SQRT_PI/SQRT_2) * y_pow * exp_term
    
    return grad    

def dTTFInverse_dz_eval(x, pos_tail, neg_tail,shift, scale): #aditya wrote this
    s = torch.sign(x - shift)
    
    # Compute λₛ based on sign
    lambda_s = torch.where(s > 0, pos_tail, neg_tail)
    
    # Compute y = λₛ|(x - μ)/σ| + 1
    y = lambda_s * torch.abs((x - shift) / scale) + 1
    
    # Compute y^{-1/λₛ - 1}
    y_pow = torch.pow(y, -1.0/lambda_s - 1)
    
    # Compute erfc^{-1}(y^{-1/λₛ})
    # Note: PyTorch doesn't have direct erfc inverse, so we use inverse of erf and adjust
    # erfc(z) = 1 - erf(z) => erfc^{-1}(w) = erf^{-1}(1 - w)
    w = torch.pow(y, -1.0/lambda_s)
    erfcinv_w =torch.erfinv(1 - w)
    
    # Compute exp(erfcinv_w^2)
    exp_term = torch.exp(torch.square(erfcinv_w))
    
    # Combine all terms
    grad = (1 / scale) * (SQRT_PI/SQRT_2) * y_pow * exp_term
    
    return grad   

def _tail_affine_inverse(x, pos_tail, neg_tail, shift, scale):
    # affine
    x = (x - shift) / scale

    # tail transform
    sign = torch.sign(x)
    tail_param = torch.where(x > 0, pos_tail, neg_tail)
    # print(x,"hello")
    
    z, lad = _extreme_inverse_and_lad(torch.abs(x), torch.abs(tail_param))
    # print(z)

    lad -= torch.log(scale)
    return sign * z, lad


def _tail_forward(z, pos_tail, neg_tail):
    sign = torch.sign(z)
    tail_param = torch.where(z > 0, pos_tail, neg_tail)
    x, lad = _extreme_transform_and_lad(torch.abs(z), tail_param)
    return sign * x, lad.sum(axis=1)


def _tail_inverse(x, pos_tail, neg_tail):
    sign = torch.sign(x)
    tail_param = torch.where(x > 0, pos_tail, neg_tail)
    z, lad = _extreme_inverse_and_lad(torch.abs(x), torch.abs(tail_param))
    return sign * z, lad.sum(axis=1)


def _copula_transform_and_lad(u, tail_param):
    inner = torch.pow(1 - u, -tail_param)
    x = (inner - 1) / tail_param
    lad = (-tail_param - 1) * torch.log(1 - u)
    return x, lad


def _copula_inverse_and_lad(x, tail_param):
    u = 1 - torch.pow(tail_param * x + 1, -1 / tail_param)
    lad = (-1 - 1 / tail_param) * torch.log(tail_param * x + 1)
    return u, lad


def _sinh_asinh_transform_and_lad(z, kurtosis_param):
    x = torch.sinh(torch.arcsinh(z) / kurtosis_param)

    lad = torch.log(torch.cosh(torch.arcsinh(z) / kurtosis_param))
    lad -= torch.log(kurtosis_param)
    lad -= 0.5 * torch.log(torch.square(z) + 1)
    return x, lad


def _sinh_asinh_inverse_and_lad(x, kurtosis_param):
    z = torch.sinh(kurtosis_param * torch.arcsinh(x))

    lad = torch.log(torch.cosh(kurtosis_param * torch.arcsinh(x))) + torch.log(
        kurtosis_param
    )
    lad -= 0.5 * torch.log(torch.square(x) + 1)
    return z, lad


def _asymmetric_scale_transform_and_lad(z, pos_scale, neg_scale):
    sq_plus_1 = (z.square() + 1.0).sqrt()
    a = pos_scale + neg_scale
    b = pos_scale - neg_scale

    pos_x = pos_scale * (sq_plus_1 + z)
    neg_x = neg_scale * (z - sq_plus_1)
    x = 0.5 * (pos_x + neg_x - b)

    lad = torch.log1p((b / a) * (z / sq_plus_1))
    lad -= torch.log(torch.tensor(2.0))
    return x, lad


def _asymmetric_scale_inverse_and_lad(x, pos_scale, neg_scale):
    a = pos_scale + neg_scale
    b = pos_scale - neg_scale
    disc = a**2 - b**2

    z_dash = a * b + 2 * a * x
    term_2 = (a**2 + 4 * b * x + 4 * x**2).sqrt()

    z = (z_dash - torch.sign(b) * term_2) / disc

    lad = torch.log(2 * a - torch.sign(b) * (2 * b + 4 * x) / term_2)
    lad -= torch.log(disc)
    return z, lad


def two_scale_affine_forward(z, shift, scale_neg, scale_pos, bound=torch.tensor(1.0)):
    # build batch x dim x knots arrays
    derivatives = torch.ones([*z.shape, 3])
    derivatives[:, :, 0] = scale_neg
    derivatives[:, :, -1] = scale_pos

    input_knots = torch.zeros([*z.shape, 3])
    input_knots[:, :, 0] = -bound
    input_knots[:, :, -1] = bound

    output_knots = torch.zeros([*z.shape, 3])
    output_knots[:, :, 0] = -bound
    output_knots[:, :, -1] = bound

    neg_region = z < -bound
    pos_region = z > bound
    body = ~torch.logical_or(neg_region, pos_region)
    neg_scale_ix = (neg_region * torch.arange(z.shape[-1]))[neg_region]
    pos_scale_ix = (pos_region * torch.arange(z.shape[-1]))[pos_region]

    x = torch.empty_like(z)
    lad = torch.empty_like(z)

    x[neg_region] = (z[neg_region] + bound) * scale_neg[neg_scale_ix] - bound
    x[pos_region] = (z[pos_region] - bound) * scale_pos[pos_scale_ix] + bound
    lad[neg_region] = -torch.log(scale_neg[neg_scale_ix])
    lad[pos_region] = -torch.log(scale_pos[pos_scale_ix])

    body_x, body_lad = forward_rqs(
        z[body], input_knots[body], output_knots[body], derivatives[body]
    )
    x[body] = body_x
    # this has already been inverted, so undo for subsequent inversion
    lad[body] = -body_lad

    x += shift

    return x, lad


def two_scale_affine_inverse(x, shift, scale_neg, scale_pos, bound=torch.tensor(1.0)):
    # build batch x dim x knots arrays
    derivatives = torch.ones([*x.shape, 3])
    derivatives[:, :, 0] = scale_neg
    derivatives[:, :, -1] = scale_pos

    input_knots = torch.zeros([*x.shape, 3])
    input_knots[:, :, 0] = -bound
    input_knots[:, :, -1] = bound

    output_knots = torch.zeros([*x.shape, 3])
    output_knots[:, :, 0] = -bound
    output_knots[:, :, -1] = bound

    # undo shift
    x -= shift

    # regions and place holders
    neg_region = x < -bound
    pos_region = x > bound
    body = ~torch.logical_or(neg_region, pos_region)
    neg_scale_ix = (neg_region * torch.arange(x.shape[-1]))[neg_region]
    pos_scale_ix = (pos_region * torch.arange(x.shape[-1]))[pos_region]

    z = torch.empty_like(x)
    lad = torch.empty_like(x)

    # scales
    z[neg_region] = (x[neg_region] + bound) / scale_neg[neg_scale_ix] - bound
    z[pos_region] = (x[pos_region] - bound) / scale_pos[pos_scale_ix] + bound
    lad[neg_region] = torch.log(scale_neg[neg_scale_ix])
    lad[pos_region] = torch.log(scale_pos[pos_scale_ix])

    # body
    body_z, body_lad = inverse_rqs(
        x[body], input_knots[body], output_knots[body], derivatives[body]
    )
    z[body] = body_z
    lad[body] = body_lad

    return z, lad

def grad_R(z,  lambda_p, lambda_n,mu, sigma):
    s = torch.sign(z)
    abs_z = torch.abs(z)
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=z.device))
    erfc_term = torch.erfc(abs_z / sqrt_2)+0.0000001 #Aditya has added the extra 0.000..1 to handle zero power negative in the next line
    log_erfc = torch.log(erfc_term)
    
    # Common terms
    lambda_s = torch.where(s == 1, lambda_p, lambda_n)
    erfc_pow = erfc_term ** (-lambda_s)
    term = (erfc_pow - 1)
    
    # Partial derivatives
    dR_dmu = torch.ones_like(z)
    dR_dsigma = (s / lambda_s) * term
    
    # For lambda_+ (only when s == 1)
    mask_p = (s == 1)
    dR_dlambda_p = torch.zeros_like(z)
    dR_dlambda_p[mask_p] = sigma[mask_p] * (
        - (erfc_pow[mask_p] - 1) / (lambda_p[mask_p] ** 2) -
        erfc_pow[mask_p] * log_erfc[mask_p] / lambda_p[mask_p]
    )
    
    # For lambda_- (only when s == -1)
    mask_n = (s == -1)
    dR_dlambda_n = torch.zeros_like(z)
    dR_dlambda_n[mask_n] = sigma[mask_n] * (
        (erfc_pow[mask_n] - 1) / (lambda_n[mask_n] ** 2) +
        erfc_pow[mask_n] * log_erfc[mask_n] / lambda_n[mask_n]
    )
    
    return dR_dlambda_p, dR_dlambda_n,dR_dmu, dR_dsigma

def flip(transform):
    """
    if it is an autoregressive transform change around the element wise transform,
    to preserve the direction of the autoregression. Otherwise, we can flip the full
    transformation.
    """

    _inverse = transform.inverse
    transform.inverse = transform.forward
    transform.forward = _inverse

    _inverse_grad=transform.inverse_derivative
    # print("hello",_inverse_grad)
    transform.inverse_derivative=transform.forward_derivative
    # print("hello2",transform.inverse_derivative)
    transform.forward_derivative=_inverse_grad
    # print("hello3",transform.forward_derviative)

    
    _inverse_grad_eval=transform.inverse_derivative_eval
    transform.inverse_derivative_eval=transform.forward_derivative_eval
    transform.forward_derivative_eval=_inverse_grad_eval



    return transform


class TailMarginalTransform(nn.Module):
    def __init__(
        self,
        features,
        pos_tail_init=None,
        neg_tail_init=None,
    ):
        self.features = features

        super(TailMarginalTransform, self).__init__()

        # init with heavy tail, otherwise heavy targets may fail to fit
        if pos_tail_init is None:
            pos_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        if neg_tail_init is None:
            neg_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        assert torch.Size([features]) == pos_tail_init.shape
        assert torch.Size([features]) == neg_tail_init.shape

        self._unc_pos_tail = torch.nn.parameter.Parameter(inv_sftplus(pos_tail_init))
        self._unc_neg_tail = torch.nn.parameter.Parameter(inv_sftplus(neg_tail_init))

    def forward(self, z, context=None):
        pos_tail_param = softplus(self._unc_pos_tail)
        neg_tail_param = softplus(self._unc_neg_tail)
        x, lad = _tail_forward(z, pos_tail_param, neg_tail_param)
        return x, lad

    def inverse(self, x, context=None):
        pos_tail_param = softplus(self._unc_pos_tail)
        neg_tail_param = softplus(self._unc_neg_tail)
        z, lad = _tail_inverse(x, pos_tail_param, neg_tail_param)
        return z, lad

    def fix_tails(self):
        # freeze only the parameters related to the tail
        self._unc_pos_tail.requires_grad = False
        self._unc_neg_tail.requires_grad = False

        

class AffineMarginalTransform(nn.Module):
    def __init__(
        self,
        features,
        shift_init=None,
        scale_init=None,
    ):
        self.features = features
        super(AffineMarginalTransform, self).__init__()

        # random inits if needed
        if shift_init is None:
            shift_init = torch.zeros([features])

        if scale_init is None:
            scale_init = torch.ones([features])

        assert torch.Size([features]) == shift_init.shape
        assert torch.Size([features]) == scale_init.shape

        # convert to unconstrained versions
        self._unc_shift = torch.nn.parameter.Parameter(shift_init)
        self._unc_scale = torch.nn.parameter.Parameter(inv_sftplus(scale_init))

    def forward(self, z, context=None):
        print('DONOT ENTER')
        shift = self._unc_shift
        # scale = softplus(self._unc_scale)
        scale = 1e-3 + softplus(self._unc_scale)

        x = z * scale + shift
        lad = torch.log(scale).sum()
        return x, lad

    def inverse(self, x, context=None):
        """heavy -> light"""
        print('DONOT ENTER')
        shift = self._unc_shift
        scale = 1e-3 + softplus(self._unc_scale)

        z = (x - shift) / scale
        lad = -torch.log(scale).sum()
        return z, lad





class TailAffineMarginalTransform(nn.Module):
    def __init__(
        self,
        features,
        pos_tail_init=None,
        neg_tail_init=None,
        shift_init=None,
        scale_init=None,
    ):
        self.features = features
        super(TailAffineMarginalTransform, self).__init__()

        # random inits if needed
        if pos_tail_init is None:
            pos_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        if neg_tail_init is None:
            neg_tail_init = torch.distributions.Uniform(
                LOW_TAIL_INIT, HIGH_TAIL_INIT
            ).sample([features])

        if shift_init is None:
            shift_init = torch.zeros([features])

        if scale_init is None:
            scale_init = torch.ones([features])

        assert torch.Size([features]) == pos_tail_init.shape
        assert torch.Size([features]) == neg_tail_init.shape
        assert torch.Size([features]) == shift_init.shape
        assert torch.Size([features]) == scale_init.shape

        # convert to unconstrained versions
        self._unc_pos_tail = torch.nn.parameter.Parameter(inv_sftplus(pos_tail_init))
        self._unc_neg_tail = torch.nn.parameter.Parameter(inv_sftplus(neg_tail_init))
        self.shift = torch.nn.parameter.Parameter(shift_init)
        self._unc_scale = torch.nn.parameter.Parameter(inv_sftplus(scale_init))

    @property
    def pos_tail(self):
        return softplus(self._unc_pos_tail)

    @property
    def neg_tail(self):
        return softplus(self._unc_neg_tail)

    @property
    def scale(self):
        return 1e-3 + softplus(self._unc_scale)

    def forward(self, z, context=None):
        """light -> heavy"""
        x, lad = _tail_affine_transform(
            z, self.pos_tail, self.neg_tail, self.shift, self.scale
        )
        return x, lad.sum(axis=-1)
    
    def forward_derivative(self,z):
        """light to heavy"""
        x=dTTF_dz(z, self.pos_tail, self.neg_tail, self.shift, self.scale)
        return(x)
    
    def forward_derivative_eval(self,z):
        """light to heavy"""
        x=dTTF_dz_eval(z, self.pos_tail, self.neg_tail, self.shift, self.scale)
        return(x)    
    
    def inverse_derivative(self,x):
        """heavy -> light"""
        z=dTTFInverse_dz(x, self.pos_tail, self.neg_tail, self.shift, self.scale)
        return(z)
    
    def inverse_derivative_eval(self,x):
        """heavy -> light"""
        z=dTTFInverse_dz_eval(x, self.pos_tail, self.neg_tail, self.shift, self.scale)
        return(z)    

    def inverse(self, x, context=None):
        """heavy -> light"""
        z, lad = _tail_affine_inverse(
            x, self.pos_tail, self.neg_tail, self.shift, self.scale
        )
        return z, lad.sum(axis=-1)

    def fix_tails(self):
        # freeze only the parameters related to the tail
        self._unc_pos_tail.requires_grad = False
        self._unc_neg_tail.requires_grad = False



class TailAffineMarginalTransform_SeparateNetParam(nn.Module):
    def __init__(
        self,device='cuda:0'
    ):
        # self.features = features
        super(TailAffineMarginalTransform_SeparateNetParam, self).__init__()
        self.dimz=2
        self.tail_param_perdim=4
        # rowsz=np.array(list(range(self.dimz*self.tail_param_perdim)))%self.dimz
        # colsz=list(range(self.dimz*self.tail_param_perdim))
        rowsz=torch.tensor(list(range(self.dimz*self.tail_param_perdim)))%self.dimz
        colsz=torch.tensor(list(range(self.dimz*self.tail_param_perdim)))


        self.flat_indicesz = self.dimz*self.tail_param_perdim * rowsz + colsz  # Flattened index in 3x12 matrix
        self.flat_indicesz=self.flat_indicesz.to(device)
        # self.MMM = torch.zeros((self.dimz*self.dimz*self.tail_param_perdim, self.dimz*self.tail_param_perdim), dtype=torch.float32)
        # self.MMM[flat_indicesz, torch.arange(self.dimz*self.tail_param_perdim)] = 1.0

        # random inits if needed
        # if pos_tail_init is None:
        #     pos_tail_init = torch.distributions.Uniform(
        #         LOW_TAIL_INIT, HIGH_TAIL_INIT
        #     ).sample([features])

        # if neg_tail_init is None:
        #     neg_tail_init = torch.distributions.Uniform(
        #         LOW_TAIL_INIT, HIGH_TAIL_INIT
        #     ).sample([features])

        # if shift_init is None:
        #     shift_init = torch.zeros([features])

        # if scale_init is None:
        #     scale_init = torch.ones([features])

        # assert torch.Size([features]) == pos_tail_init.shape
        # assert torch.Size([features]) == neg_tail_init.shape
        # assert torch.Size([features]) == shift_init.shape
        # assert torch.Size([features]) == scale_init.shape

        # convert to unconstrained versions
        # self._unc_pos_tail = torch.nn.parameter.Parameter(inv_sftplus(pos_tail_init))
        # self._unc_neg_tail = torch.nn.parameter.Parameter(inv_sftplus(neg_tail_init))
        # self.shift = torch.nn.parameter.Parameter(shift_init)
        # self._unc_scale = torch.nn.parameter.Parameter(inv_sftplus(scale_init))

    
    def pos_tail(self,x):
        return softplus(x)

    
    def neg_tail(self,x):
        return softplus(x)

    
    def scale(self,x):
        return 1e-3 + softplus(x)

    def forward(self, z,tail_param,fix_tail=False,fix_tail_param_pos=None,fix_tail_param_neg=None):
        # print("ZZZ",z.shape)
        # z=z.reshape(-1,2)
        dim=2
        _unc_pos_tail,_unc_neg_tail,shift,_unc_scale = tail_param[...,0:dim],tail_param[...,dim:2*dim],tail_param[...,2*dim:3*dim],tail_param[...,3*dim:4*dim]
        """light -> heavy"""
        if fix_tail==True:
            # print("hello11",z.shape,fix_tail_param_pos.shape,_unc_pos_tail.shape)
            fix_tail_param_pos=fix_tail_param_pos.reshape(_unc_pos_tail.shape)
            fix_tail_param_neg=fix_tail_param_neg.reshape(_unc_neg_tail.shape)

            x, lad = _tail_affine_transform(
                z, fix_tail_param_pos, fix_tail_param_neg, shift, self.scale(_unc_scale)
                )
        else:
            x, lad = _tail_affine_transform(
                z, self.pos_tail(_unc_pos_tail), self.neg_tail(_unc_neg_tail), shift, self.scale(_unc_scale)
            )
        # print("NAN CHECK",torch.sum(torch.isnan(x)))
        return x
    

    def inverse(self, x,tail_param, fix_tail=False,fix_tail_param_pos=None,fix_tail_param_neg=None):
        """heavy -> light"""
        dim=2
        # x=x.reshape(-1,2)
        _unc_pos_tail,_unc_neg_tail,shift,_unc_scale, = tail_param[...,0:dim],tail_param[...,dim:2*dim],tail_param[...,2*dim:3*dim],tail_param[...,3*dim:4*dim]
        # print("HHHH-",_unc_pos_tail.shape)
        if fix_tail==True:
            # print("hello",fix_tail_param_pos.shape,_unc_pos_tail.shape)
            fix_tail_param_pos=fix_tail_param_pos.reshape(_unc_pos_tail.shape)
            fix_tail_param_neg=fix_tail_param_neg.reshape(_unc_neg_tail.shape) 

            z, lad = _tail_affine_inverse(
                x, fix_tail_param_pos,fix_tail_param_neg , shift, self.scale(_unc_scale)
                )
        else:
            z, lad = _tail_affine_inverse(
                x, self.pos_tail(_unc_pos_tail), self.neg_tail(_unc_neg_tail), shift, self.scale(_unc_scale)
                )
        return z
    def dTTF_dtailparam(self,z,tail_param): #outputs 4 deruvative B X 2    4 times aditya
        dim=2
        _unc_pos_tail,_unc_neg_tail,shift,_unc_scale = tail_param[...,0:dim],tail_param[...,dim:2*dim],tail_param[...,2*dim:3*dim],tail_param[...,3*dim:4*dim] 

        return grad_R(z,  self.pos_tail(_unc_pos_tail), self.neg_tail(_unc_neg_tail), shift, self.scale(_unc_scale)) 

        # return(reshape_even_odd_columns(torch.cat(grad_R(z,  self.pos_tail(_unc_pos_tail), self.neg_tail(_unc_neg_tail), shift, self.scale(_unc_scale)),1)))
    def fwd_dTTF_dz(self,z,tail_param): #outputs a jacobian B X 2 X 2 aditya
        dim=2
        _unc_pos_tail,_unc_neg_tail,shift,_unc_scale = tail_param[...,0:dim],tail_param[...,dim:2*dim],tail_param[...,2*dim:3*dim],tail_param[...,3*dim:4*dim]        

        return torch.diag_embed((dTTF_dz(z,  self.pos_tail(_unc_pos_tail), self.neg_tail(_unc_neg_tail), shift, self.scale(_unc_scale))))

    # def fix_tails(self):
    #     # freeze only the parameters related to the tail
    #     self._unc_pos_tail.requires_grad = False
    #     self._unc_neg_tail.requires_grad = False
    def reshape_generalized(self,input_tensor):
        B, H = input_tensor.shape
        # Compute full Jacobian
        self.flat_indicesz=self.flat_indicesz

        scatter_indices = self.flat_indicesz.expand(B, H)  # (B, K)

        # Create output buffer
        J_flat = torch.zeros((B, self.dimz*self.dimz*self.tail_param_perdim), dtype=input_tensor.dtype).to(input_tensor.device)

        # Scatter each v_batch[b, :] into the correct flat index positions
        J_flat.scatter_(dim=1, index=scatter_indices, src=input_tensor)

        # Reshape to (B, m, n)
        J_batch = J_flat.view(B, self.dimz, self.dimz*self.tail_param_perdim)


        # MMM=(self.MMM).to(input_tensor.device)
    

        # # Batched matrix multiply: (B × 36) = (B × 12) @ (12 × 36)ᵀ
        # J_flat_batch = torch.matmul(input_tensor, MMM.T)  # shape (B, 36)

        # # Reshape into (B, 3, 12)
        # J_batch = J_flat_batch.view(-1, self.dimz, self.dimz*self.tail_param_perdim)
        # J_batch
        return(J_batch.permute(0,2,1)) #B X12 X3


def reshape_even_odd_columns(input_tensor):

    B, H = input_tensor.shape
    output = torch.zeros((B, H, 2), device=input_tensor.device)
    
    # Even indices (0, 2, ...) -> channel 0
    output[:, 0::2, 0] = input_tensor[:, 0::2]
    
    # Odd indices (1, 3, ...) -> channel 1
    output[:, 1::2, 1] = input_tensor[:, 1::2]
    
    return output


import numpy as np


class TailAffineMarginalTransform_SeparateNetParam2(nn.Module):
    def __init__(
        self,device='cuda:0',dimz=20
    ):
        # self.features = features
        super(TailAffineMarginalTransform_SeparateNetParam2, self).__init__()
        self.dimz=dimz
        self.tail_param_perdim=4
        self.og_tail_shape=(self.tail_param_perdim,self.dimz)





  

    
    def pos_tail2(self,x):
        return softplus(x)

    
    def neg_tail2(self,x):
        return softplus(x)

    
    def scale2(self,x):
        return 1e-3 + softplus(x)

    def forward(self, z,tail_param,fix_tail=False,fix_tail_param_pos=None,fix_tail_param_neg=None):
        # print("ZZZ",z.shape)
        # z=z.reshape(-1,2)
        # print("hello",tail_param.shape)
        dim=self.dimz
        dummy_tail_param=tail_param.reshape(tail_param.shape[0],self.og_tail_shape[0],self.og_tail_shape[1])

        # print("dummy_extreme",dummy_tail_param.shape,z.shape)

        _unc_pos_tail,_unc_neg_tail,shift,_unc_scale =dummy_tail_param[:,0,:],dummy_tail_param[:,1,:],dummy_tail_param[:,2,:],dummy_tail_param[:,3,:]
        """light -> heavy"""
        if fix_tail==True:
            # print("hello11",z.shape,fix_tail_param_pos.shape,_unc_pos_tail.shape)
            fix_tail_param_pos=fix_tail_param_pos.reshape(_unc_pos_tail.shape)
            fix_tail_param_neg=fix_tail_param_neg.reshape(_unc_neg_tail.shape)

            x, lad = _tail_affine_transform(
                z, fix_tail_param_pos, fix_tail_param_neg, shift, self.scale(_unc_scale)
                )
        else:
            x, lad = _tail_affine_transform(
                z, (_unc_pos_tail), (_unc_neg_tail), shift, (_unc_scale)
            )
        # print('fwd',z.max(),z.min(),torch.sinh(z).mean())
        # print('fwd',z.max())
        has_nan = torch.isnan(z).any()
        if has_nan:
            print('firwardZZZZ has nan')



        has_nan = torch.isnan(x).any()
        if has_nan:
            print('firward has nan')
        return torch.clamp(x,-100000,100000)#torch.log(z + torch.sqrt(z**2 + 1))#(z**3)#torch.clamp(torch.sinh(z),-3000,3000)#x
    

    def inverse(self, x,tail_param, fix_tail=False,fix_tail_param_pos=None,fix_tail_param_neg=None):
        """heavy -> light"""
        dim=self.dimz
        # x=x.reshape(-1,2)
        nan7=torch.isnan(tail_param).any()
        if nan7:
            print("tail has nan")
            print(gdgdg)
        dim=self.dimz
        dummy_tail_param=tail_param.reshape(tail_param.shape[0],self.og_tail_shape[0],self.og_tail_shape[1])

        _unc_pos_tail,_unc_neg_tail,shift,_unc_scale =dummy_tail_param[:,0,:],dummy_tail_param[:,1,:],dummy_tail_param[:,2,:],dummy_tail_param[:,3,:]        # print("HHHH-",_unc_pos_tail.shape)
        if fix_tail==True:
            # print("hello",fix_tail_param_pos.shape,_unc_pos_tail.shape)
            fix_tail_param_pos=fix_tail_param_pos.reshape(_unc_pos_tail.shape)
            fix_tail_param_neg=fix_tail_param_neg.reshape(_unc_neg_tail.shape) 

            z, lad = _tail_affine_inverse(
                x, fix_tail_param_pos,fix_tail_param_neg , shift, self.scale(_unc_scale)
                )
        else:
            z, lad = _tail_affine_inverse(
                x, (_unc_pos_tail),  (_unc_neg_tail), shift,  (_unc_scale)
                )
        # print('inverse',x.max(),x.min(),torch.arcsinh(x).mean())
        # print("inverse",x.max())
        has_nan = torch.isnan(z).any()
        if has_nan:
            print('inverse has nan')
        return z#torch.clamp((torch.exp(x) - torch.exp(-x)) / 2,-1000,1000)#torch.sign(x)*torch.pow(torch.sign(x)*x,1/3)#torch.arcsinh(x)#z
    def dTTF_dtailparam(self,z,tail_param): #outputs 4 derovatove each of B X 2 aditya
        dim=self.dimz
        dummy_tail_param=tail_param.reshape(tail_param.shape[0],self.og_tail_shape[0],self.og_tail_shape[1])

        _unc_pos_tail,_unc_neg_tail,shift,_unc_scale =dummy_tail_param[:,0,:],dummy_tail_param[:,1,:],dummy_tail_param[:,2,:],dummy_tail_param[:,3,:]      
        return grad_R(z,   (_unc_pos_tail), (_unc_neg_tail), shift,  (_unc_scale))    

    def fwd_dTTF_dz(self,z,tail_param): #outputs a jacobian B X 2 X 2 aditya
        dim=self.dimz
        dummy_tail_param=tail_param.reshape(tail_param.shape[0],self.og_tail_shape[0],self.og_tail_shape[1])

        _unc_pos_tail,_unc_neg_tail,shift,_unc_scale =dummy_tail_param[:,0,:],dummy_tail_param[:,1,:],dummy_tail_param[:,2,:],dummy_tail_param[:,3,:]            
        has_nan = torch.isnan(dTTF_dz(z,   (_unc_pos_tail), (_unc_neg_tail), shift,  (_unc_scale))).any()
        if has_nan:
            print('fwd_dTTF_dz has nan')
        
        # print("fwd_dTTF_dz",z.max(),(dTTF_dz(z,   (_unc_pos_tail), (_unc_neg_tail), shift,  (_unc_scale))).max())
        # print('fwd_dTTF_dz',z.max(),z.min(),torch.diag_embed(torch.cosh(z)).mean())
        # return  torch.diag_embed(1 / torch.sqrt(z**2 + 1))#torch.clamp(torch.diag_embed(torch.cosh(z)),-100000,1000000)  #torch.diag_embed((dTTF_dz(z,   (_unc_pos_tail), (_unc_neg_tail), shift,  (_unc_scale))))
        return torch.clamp(torch.diag_embed((dTTF_dz(z,   (_unc_pos_tail), (_unc_neg_tail), shift,  (_unc_scale)))),-10000,10000)
    def fwd_dTTFInverse_dz(self,z,tail_param):

        dim=self.dimz
        dummy_tail_param=tail_param.reshape(tail_param.shape[0],self.og_tail_shape[0],self.og_tail_shape[1])

        _unc_pos_tail,_unc_neg_tail,shift,_unc_scale =dummy_tail_param[:,0,:],dummy_tail_param[:,1,:],dummy_tail_param[:,2,:],dummy_tail_param[:,3,:]            
        # print('fwd_dTTFInverse_dz',z.max(),z.min(),(1/torch.cosh(z)).mean())
        # print("fwd_dTTFInverse_dz",z.max(),dTTFInverse_dz(z,  (_unc_pos_tail),  (_unc_neg_tail), shift,  (_unc_scale)).max())
        has_nan = torch.isnan(dTTFInverse_dz(z,  (_unc_pos_tail),  (_unc_neg_tail), shift,  (_unc_scale))).any()
        if has_nan:
            print('fwd_dTTFInverse_dz has nan')
            print(gorilla)

        return torch.clamp(dTTFInverse_dz(z,  (_unc_pos_tail),  (_unc_neg_tail), shift,  (_unc_scale)),-100000,100000)
        # return #torch.clamp((torch.exp(z) + torch.exp(-z)) / 2 ,-1000,1000)#(1/3) * (torch.abs(z)+0.00001) ** (-2/3)#torch.clamp(1/torch.cosh(z),-10000,10000)#dTTFInverse_dz(z,  (_unc_pos_tail),  (_unc_neg_tail), shift,  (_unc_scale))

