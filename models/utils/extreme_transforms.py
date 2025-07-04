import torch
import torch.nn as nn
from torch.nn.functional import softplus, relu, sigmoid
from typing import TypedDict, Optional, Callable
from math import sqrt
import numpy as np


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


def _stable_erfcinv(x, log_x):
    # print("CORRECT")
    x = torch.clamp(x, min=MIN_ERFC_INV)  # Or any small positive value
    return -torch.special.ndtri(0.5 * x) / torch.sqrt(torch.tensor(2.0))


def _extreme_transform_and_lad(z, tail_param):

    g = torch.erfc(z / SQRT_2) + 0.000001 #Aditya has added the extra 0.000..1 to handle zero power negative in the next line
    x = (torch.pow(g, -tail_param) - 1) / tail_param
    return x


def _extreme_inverse_and_lad(x, tail_param):
    inner = 1 + tail_param * x
    g = torch.pow(inner, -1 / tail_param)
    log_g = -torch.log(inner) / tail_param
    erfcinv_val = _stable_erfcinv(g, log_g)

    z = SQRT_2 * erfcinv_val

    lad = (-1 - 1 / tail_param) * torch.log(inner)
    lad += torch.square(erfcinv_val)
    lad += torch.log(torch.tensor(SQRT_PI / SQRT_2))

    return z, lad


def _tail_affine_transform(z, lambd_plus, lambd_neg, mu, sigma):

    lambd_plus = softplus(lambd_plus)
    lmabd_neg = softplus(lambd_neg)
    sigma = 1e-3 + softplus(sigma)

    sign = torch.sign(z)
    tail_param = torch.where(z > 0, lambd_plus, lmabd_neg)
    g = torch.erfc(torch.abs(z) / SQRT_2) + 0.000001 #Aditya has added the extra 0.000..1 to handle zero power negative in the next line
    x = (torch.pow(g, -tail_param) - 1) / tail_param
    return sign * x * sigma + mu


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

    z, lad = _extreme_inverse_and_lad(torch.abs(x), torch.abs(tail_param))

    lad -= torch.log(scale)
    return sign * z, lad

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



class TTF(nn.Module):
    def __init__(
        self, device='cuda:0', dim=20
    ):
        # self.features = features
        super(TTF, self).__init__()
        self.lambd_plus = torch.nn.Parameter(torch.randn(()))
        self.lambd_neg = torch.nn.Parameter(torch.randn(()))
        self.mu = torch.nn.Parameter(torch.randn(()))
        self.sigma = torch.nn.Parameter(torch.randn(()))

    def forward(self, z):
        sign = torch.sign(z)
        tail_param = torch.where(z > 0, self.lambd_plus, self.lmabd_neg)
        g = torch.erfc(torch.abs(z) / SQRT_2) + 1e-6 # handle zero power negative in next line
        x = (torch.pow(g, -tail_param) - 1) / tail_param
        x =  sign * x * self.sigma + self.mu

        return x
