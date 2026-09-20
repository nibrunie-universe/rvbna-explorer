# -*- coding: utf-8 -*-

from pysollya import SollyaObject, round_sol, floor_sol, log2, RN, halfprecisionformat, singleformat, doubleformat, bfloat16format
import random
import math
import statistics
import numpy as np

# Named format lookup for the web API
FORMAT_MAP = {
    "bf16": bfloat16format,
    "fp16": halfprecisionformat,
    "fp32": singleformat,
    "fp64": doubleformat,
}

def exactDotProd(a, b):
    """Exact dot product: no intermediate rounding, exact accumulation, no final rounding."""
    s = SollyaObject(0.0)
    for p in (ai * bi for (ai, bi) in zip(a, b)):
        s += p
    return s

def correctlyRoundedDotProd(a, b, resPrec=singleformat):
    """Correctly rounded dot product: no intermediate rounding,
    exact accumulation then a single final round."""
    prods = [ai * bi for (ai, bi) in zip(a, b)]
    s = SollyaObject(0.0)
    for p in prods:
        s += p
    return round_sol(s, resPrec, RN)

def approxMultDotProd(a, b, multPrec=halfprecisionformat, resPrec=singleformat):
    """Dot product with rounded products (FP MUL) then exact accumulation before final rounding."""
    # Each product is rounded independently
    prods = [round_sol(ai * bi, multPrec, RN) for (ai, bi) in zip(a, b)]
    # The sum has no intermediate rounding, the final result is rounded to the target resPrec format
    return round_sol(sum(prods, SollyaObject(0)), resPrec, RN)

def approxMultBinTreeAccDotProd(a, b, multPrec=halfprecisionformat, addPrec=halfprecisionformat, resPrec=singleformat):
    """Dot product with rounded products and a binary-tree of rounded additions."""
    prods = [round_sol(ai * bi, multPrec, RN) for (ai, bi) in zip(a, b)]
    def binAddTree(v):
        if len(v) == 1:
            return v
        evenOps = v[0::2]
        oddOps  = v[1::2]
        addRes = [round_sol(ai + bi, addPrec, RN) for (ai, bi) in zip(evenOps, oddOps)] + ([v[-1]] if len(v) % 2 == 1 else [])
        return binAddTree(addRes)
    result = binAddTree(prods)[0]
    return round_sol(result, resPrec, RN)

def approxMultLinearAccDotProd(a, b, multPrec=halfprecisionformat, addPrec=halfprecisionformat, resPrec=singleformat):
    """Dot product with rounded products and a linear comb of rounded additions."""
    prods = [round_sol(ai * bi, multPrec, RN) for (ai, bi) in zip(a, b)]
    result = SollyaObject(0)
    for p in prods:
        result = round_sol(result + p, addPrec, RN)
    return round_sol(result, resPrec, RN)

def fmaDotProd(a, b, prec=singleformat, resPrec=singleformat):
    """Dot product based on a sequence of FMA (fused multiply-add).
    Each step: res = round(res + ai * bi)  — one rounding per FMA."""
    res = SollyaObject(0.0)
    for ai, bi in zip(a, b):
        res = round_sol(res + ai * bi, prec, RN)
    return round_sol(res, resPrec, RN)

def roundToOddFixed(v, lsbIndex=0):
    """Rounding-to-odd (jamming inexact value) for fixed-point."""
    sign = v < 0
    v = abs(v)
    scalingFactor = SollyaObject(2) ** (-lsbIndex)
    try:
        scaled = int(floor_sol(v * scalingFactor))
    except ValueError as e:
        print(v)
        raise
    notExact = (scaled / scalingFactor) != v
    rounded = scaled | (1 if notExact else 0)
    # TODO: need to handle overflow 
    # TODO: need to handle underflow 
    return (-1 if sign else 1) * rounded / scalingFactor

def roundToOdd(v, prec: int, emin=None):
    """Rounding-to-odd (jamming inexact value) for floating-point."""
    sign = v < 0
    v = abs(v)
    if v == 0:
        return 0
    exp = floor_sol(log2(abs(v)))
    if emin:
        exp = max(exp, emin)
    preScalingFactor = SollyaObject(2) ** (-exp)
    preScaled = v * preScalingFactor
    return (-1 if sign else 1) * roundToOddFixed(preScaled, -prec) / preScalingFactor

class ExactFormat:
    def __init__(self):
        pass
    @staticmethod
    def exponent(value):
        if value == 0:
            return 0
        return floor_sol(log2(abs(value)))

def bulkNormDotProd(a, b, bulkNormPrec=25, finalPrec=24, prodFormats=(ExactFormat(), ExactFormat())):
    """Bulk normalization dot product: products are rounded to a fixed-point
    representation whose exponent is determined from the maximum product exponent,
    then accumulated and rounded."""
    prods = [ai * bi for (ai, bi) in zip(a, b)]
    lhsFormat, rhsFormat = prodFormats
    prodExps = [lhsFormat.exponent(ai) + rhsFormat.exponent(bi) for (ai, bi) in zip(a, b)]
    maxExp = max(prodExps)
    roundedProds = [roundToOddFixed(p, maxExp - bulkNormPrec) for p in prods]
    return roundToOdd(sum(roundedProds), finalPrec)

def generate_vectors(n, k, average, sigma, input_prec=halfprecisionformat,
                     a_average=None, a_sigma=None, b_average=None, b_sigma=None,
                     a_distribution="gaussian", b_distribution="gaussian", seed=None):
    """Generate n pairs of k-element random vectors.
    
    Each vector can have its own distribution parameters.
    Supported distributions: "gaussian", "uniform", "lognormal".
    """
    # Fall back to shared average/sigma if per-vector params not given
    a_avg = a_average if a_average is not None else average
    a_sig = a_sigma if a_sigma is not None else sigma
    b_avg = b_average if b_average is not None else average
    b_sig = b_sigma if b_sigma is not None else sigma

    if seed is not None:
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
    else:
        rng = random
        np_rng = np.random

    def _sample(distribution, avg, sig):
        if distribution == "uniform":
            # Uniform on [avg - sig, avg + sig]
            return rng.uniform(avg - sig, avg + sig)
        elif distribution == "lognormal":
            # Log-normal with underlying normal(avg, sig)
            return rng.lognormvariate(avg, sig)
        else:  # gaussian (default)
            if seed is not None:
                return float(np_rng.normal(avg, sig))
            else:
                return float(np.random.normal(avg, sig))

    def genVector(k, distribution, avg, sig):
        return [round_sol(_sample(distribution, avg, sig), input_prec, RN) for _ in range(k)]

    return [(genVector(k, a_distribution, a_avg, a_sig),
             genVector(k, b_distribution, b_avg, b_sig)) for _ in range(n)]

def evaluate_errors(vectors, func, kwargs, golden_values):
    """Compute relative errors of func vs golden, return sorted errors + stats."""
    abs_errors = []
    rel_errors = []
    signed_rel_errors = []
    signed_errors = []
    pos_a_count = 0
    neg_a_count = 0
    pos_b_count = 0
    neg_b_count = 0
    exact_pos_count = 0
    exact_neg_count = 0
    opposite_sign_count = 0
    err_pos_res_pos = []
    err_neg_res_pos = []
    err_pos_res_neg = []
    err_neg_res_neg = []
    for ((a, b), golden) in zip(vectors, golden_values):
        if golden > 0:
            exact_pos_count += 1
        elif golden < 0:
            exact_neg_count += 1
            
        for ai in a:
            if ai > 0: pos_a_count += 1
            elif ai < 0: neg_a_count += 1
        for bi in b:
            if bi > 0: pos_b_count += 1
            elif bi < 0: neg_b_count += 1
            
        res = func(a, b, **kwargs)
        
        if (res > 0 and golden < 0) or (res < 0 and golden > 0):
            opposite_sign_count += 1
            
        err = res - golden
        abs_error = abs(err)
        if golden == 0:
            rel_error = 0.0 if abs_error == 0 else 1e308
            signed_rel_error = 0.0 if abs_error == 0 else (1e308 if err > 0 else -1e308)
        else:
            rel_error = abs(abs_error / golden)
            # Here, we are interested in the sign of the error regardless of the sign of the golden value
            signed_rel_error = float(err / abs(golden))
            
        if golden > 0:
            if signed_rel_error > 0:
                err_pos_res_pos.append(float(signed_rel_error))
            elif signed_rel_error < 0:
                err_neg_res_pos.append(float(signed_rel_error))
        elif golden < 0:
            if signed_rel_error > 0:
                err_pos_res_neg.append(float(signed_rel_error))
            elif signed_rel_error < 0:
                err_neg_res_neg.append(float(signed_rel_error))
                
        # Check if the result is NaN (Not a Number).
        if abs_error != abs_error or rel_error != rel_error:
            print(f"NaN error detected, func={func.__name__}, a={a}, b={b}, golden={golden}, res={res}")
        abs_errors.append(float(abs_error))
        rel_errors.append(float(rel_error))
        signed_rel_errors.append(float(signed_rel_error))
        signed_errors.append(float(err))

    sorted_rel_errors = sorted(rel_errors)
    sorted_signed_rel_errors = sorted(signed_rel_errors)
    max_err = max(rel_errors)
    min_err = min(rel_errors)

    # Geometric mean excluding exact zeros and extreme sentinels
    non_zero = [e for e in rel_errors if e > 0 and e < 1e308]
    if non_zero:
        geo_mean = math.exp(sum(math.log(e) for e in non_zero) / len(non_zero))
    else:
        geo_mean = 0.0

    exact_count = len(rel_errors) - len(non_zero)
    
    mean_signed_error = sum(signed_errors) / len(signed_errors) if signed_errors else 0.0
    valid_signed_rel = [e for e in signed_rel_errors if abs(e) < 1e308]
    mean_signed_rel_error = sum(valid_signed_rel) / len(valid_signed_rel) if valid_signed_rel else 0.0

    sum_signed_error = sum(signed_errors)
    sum_signed_rel_error = sum(valid_signed_rel)
    
    pos_count = sum(1 for e in signed_errors if e > 0)
    neg_count = sum(1 for e in signed_errors if e < 0)

    return {
        "sorted_rel_errors": sorted_rel_errors,
        "sorted_signed_rel_errors": sorted_signed_rel_errors,
        "max": max_err,
        "min": min_err,
        "geometric_mean": geo_mean,
        "exact_count": exact_count,
        "mean_signed_error": mean_signed_error,
        "mean_signed_rel_error": mean_signed_rel_error,
        "sum_signed_error": sum_signed_error,
        "sum_signed_rel_error": sum_signed_rel_error,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "pos_a_count": pos_a_count,
        "neg_a_count": neg_a_count,
        "pos_b_count": pos_b_count,
        "neg_b_count": neg_b_count,
        "exact_pos_count": exact_pos_count,
        "exact_neg_count": exact_neg_count,
        "opposite_sign_count": opposite_sign_count,
        "sorted_err_pos_res_pos": sorted(err_pos_res_pos),
        "sorted_err_neg_res_pos": sorted(err_neg_res_pos),
        "sorted_err_pos_res_neg": sorted(err_pos_res_neg),
        "sorted_err_neg_res_neg": sorted(err_neg_res_neg),
    }
