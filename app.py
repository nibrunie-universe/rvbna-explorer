import datetime
from flask import Flask, request, jsonify, send_from_directory
from rvbna_web import (
    exactDotProd,
    correctlyRoundedDotProd,
    approxMultDotProd,
    approxMultBinTreeAccDotProd,
    approxMultLinearAccDotProd,
    fmaDotProd,
    bulkNormDotProd,
    generate_vectors,
    evaluate_errors,
    FORMAT_MAP,
    singleformat,
    halfprecisionformat,
)

import os

MAX_N = int(os.environ.get("MAX_N", 50000))
DEBUG = int(os.environ.get("DEBUG", 0))

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__, static_folder="static")

# Maximum number of requests per minutes that the server serves
MAX_REQUESTS_PER_MINUTE = 10
# Maximum number of schemes the user is allowed to request at once
MAX_SCHEME_NUM = 10
# Current version of the application advertised on the user interface
# Changelog(s):
# 0.1.2: adding short-handle for demo links (fprox's blog post)
# 0.1.3: adding zoom capabilities (start,end range on x-axis) for plots
# 0.1.4: restricting number of samples for 64-element experiments (limit compute time)
VERSION = "0.1.4"

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[f"{MAX_REQUESTS_PER_MINUTE} per minute"],
    storage_uri="memory://",
)


EXPERIMENTS = {
    "demo": {
        "n": 1000,
        "k": 128,
        "inputPrec": "bf16",
        "aDistribution": "gaussian",
        "aAverage": 0,
        "aSigma": 5,
        "bDistribution": "gaussian",
        "bAverage": 0,
        "bSigma": 5,
        "schemes": [
            {"id": "s1", "variant": "correctly_rounded", "resPrec": "fp32", "name": "Exact [FP32]"},
            {"id": "s2", "variant": "approx_mult", "multPrec": "bf16", "resPrec": "fp32", "name": "FP MUL [BF16] + FP Add [FP32]"},
            {"id": "s3", "variant": "approx_mult_acc", "multPrec": "bf16", "addPrec": "bf16", "resPrec": "fp32", "name": "FP MUL [BF16] + Add Tree [BF16]"}
        ]
    },
    "exp_4d": {
        "appVersion": "0.0.9",
        "dataSource": "random",
        "n": 10000,
        "k": 4,
        "average": 0,
        "sigma": 5,
        "inputPrec": "bf16",
        "aDistribution": "gaussian",
        "aAverage": 0,
        "aSigma": 5,
        "bDistribution": "gaussian",
        "bAverage": 0,
        "bSigma": 5,
        "seed": 1306,
        "schemes": [
            {
                "variant": "correctly_rounded",
                "resPrec": "fp32",
                "customName": "A. Correcty Rounded",
                "name": "A. Correcty Rounded"
            },
            {
                "variant": "fma",
                "fmaPrec": "fp32",
                "resPrec": "fp32",
                "customName": "B. Sequence of FMAs",
                "name": "B. Sequence of FMAs"
            },
            {
                "variant": "approx_mult",
                "multPrec": "bf16",
                "resPrec": "fp32",
                "customName": "C. FP Mul + Exact Acc",
                "name": "C. FP Mul + Exact Acc"
            },
            {
                "variant": "approx_mult_acc",
                "multPrec": "bf16",
                "addPrec": "bf16",
                "resPrec": "fp32",
                "customName": "D. FP Mul + FP Add tree (small precision)",
                "name": "D. FP Mul + FP Add tree (small precision)"
            },
            {
                "variant": "approx_mult_acc",
                "multPrec": "fp32",
                "addPrec": "fp32",
                "resPrec": "fp32",
                "customName": "E. FP Mul + FP Add tree (large precision):",
                "name": "E. FP Mul + FP Add tree (large precision):"
            },
            {
                "variant": "bulk_norm",
                "bulkNormPrec": "25",
                "finalPrec": "23",
                "customName": "F. Bulk Normalization",
                "name": "F. Bulk Normalization"
            }
        ]
    },
    "exp_4d_zoom": {
        "appVersion": "0.0.9",
        "dataSource": "random",
        "n": 10000,
        "k": 4,
        "average": 0,
        "sigma": 5,
        "inputPrec": "bf16",
        "aDistribution": "gaussian",
        "aAverage": 0,
        "aSigma": 5,
        "bDistribution": "gaussian",
        "bAverage": 0,
        "bSigma": 5,
        "seed": 1306,
        "start": 9700,
        "schemes": [
            {
                "variant": "correctly_rounded",
                "resPrec": "fp32",
                "customName": "A. Correcty Rounded",
                "name": "A. Correcty Rounded"
            },
            {
                "variant": "fma",
                "fmaPrec": "fp32",
                "resPrec": "fp32",
                "customName": "B. Sequence of FMAs",
                "name": "B. Sequence of FMAs"
            },
            {
                "variant": "approx_mult_acc",
                "multPrec": "fp32",
                "addPrec": "fp32",
                "resPrec": "fp32",
                "customName": "E. FP Mul + FP Add tree (large precision):",
                "name": "E. FP Mul + FP Add tree (large precision):"
            },
            {
                "variant": "bulk_norm",
                "bulkNormPrec": "25",
                "finalPrec": "23",
                "customName": "F. Bulk Normalization",
                "name": "F. Bulk Normalization"
            }
        ]
    },
    "exp_64d": {
        "appVersion": "0.0.9",
        "dataSource": "random",
        "n": 2500,
        "k": 64,
        "average": 0,
        "sigma": 5,
        "inputPrec": "bf16",
        "aDistribution": "gaussian",
        "aAverage": 0,
        "aSigma": 5,
        "bDistribution": "gaussian",
        "bAverage": 0,
        "bSigma": 5,
        "seed": 1306,
        "schemes": [
            {
                "variant": "correctly_rounded",
                "resPrec": "fp32",
                "customName": "A. Correcty Rounded",
                "name": "A. Correcty Rounded"
            },
            {
                "variant": "fma",
                "fmaPrec": "fp32",
                "resPrec": "fp32",
                "customName": "B. Sequence of FMAs",
                "name": "B. Sequence of FMAs"
            },
            {
                "variant": "approx_mult",
                "multPrec": "bf16",
                "resPrec": "fp32",
                "customName": "C. FP Mul + Exact Ac",
                "name": "C. FP Mul + Exact Ac"
            },
            {
                "variant": "approx_mult_acc",
                "multPrec": "bf16",
                "addPrec": "bf16",
                "resPrec": "fp32",
                "customName": "D. FP Mul + FP Add tree (small precision)",
                "name": "D. FP Mul + FP Add tree (small precision)"
            },
            {
                "variant": "approx_mult_acc",
                "multPrec": "fp32",
                "addPrec": "fp32",
                "resPrec": "fp32",
                "customName": "E. FP Mul + FP Add tree (large precision):",
                "name": "E. FP Mul + FP Add tree (large precision):"
            },
            {
                "variant": "bulk_norm",
                "bulkNormPrec": "25",
                "finalPrec": "23",
                "customName": "F. Bulk Normalization",
                "name": "F. Bulk Normalization"
            }
        ]
    },
    "exp_64d_zoom": {
        "appVersion": "0.0.9",
        "dataSource": "random",
        "n": 2500,
        "k": 64,
        "average": 5,
        "sigma": 5,
        "inputPrec": "bf16",
        "aDistribution": "gaussian",
        "aAverage": 5,
        "aSigma": 5,
        "bDistribution": "gaussian",
        "bAverage": 5,
        "bSigma": 5,
        "seed": 1306,
        "schemes": [
            {
                "variant": "correctly_rounded",
                "resPrec": "fp32",
                "customName": "A. Correcty Rounded",
                "name": "A. Correcty Rounded"
            },
            {
                "variant": "fma",
                "fmaPrec": "fp32",
                "resPrec": "fp32",
                "customName": "B. Sequence of FMAs",
                "name": "B. Sequence of FMAs"
            },
            {
                "variant": "approx_mult_acc",
                "multPrec": "fp32",
                "addPrec": "fp32",
                "resPrec": "fp32",
                "customName": "E. FP Mul + FP Add tree (large precision):",
                "name": "E. FP Mul + FP Add tree (large precision):"
            },
            {
                "variant": "bulk_norm",
                "bulkNormPrec": "25",
                "finalPrec": "23",
                "customName": "F. Bulk Normalization",
                "name": "F. Bulk Normalization"
            }
        ]
    },
}

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/experiment/<handle>")
def get_experiment(handle):
    print(f"Trying to localte handle {handle}")
    if handle in EXPERIMENTS:
        print("handle found!")
        return jsonify(EXPERIMENTS[handle])
    print("handle not found!")
    return jsonify({"error": "Experiment not found"}), 404

@app.route("/api/version")
def get_version():
    return jsonify({"version": VERSION})

@app.route("/api/config")
def get_config():
    return jsonify({"max_n": MAX_N})

@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    data = request.json

    schemes = data.get("schemes", [])
    if len(schemes) > MAX_SCHEME_NUM:
        return jsonify({"error": f"Too many schemes requested. Maximum allowed is {MAX_SCHEME_NUM}."}), 400

    # Distribution config
    n = int(data.get("n", 1000))
    k = int(data.get("k", 2))
    average = float(data.get("average", 5.0))
    sigma = float(data.get("sigma", 5.0))
    input_prec_name = data.get("inputPrec", "fp16")
    input_prec = FORMAT_MAP.get(input_prec_name, halfprecisionformat)

    # Clamp n to avoid extreme computation
    n = min(n, MAX_N)

    # Per-vector distribution parameters
    a_distribution = data.get("aDistribution", "gaussian")
    a_average = float(data.get("aAverage", average))
    a_sigma = float(data.get("aSigma", sigma))
    b_distribution = data.get("bDistribution", "gaussian")
    b_average = float(data.get("bAverage", average))
    b_sigma = float(data.get("bSigma", sigma))
    
    seed_val = data.get("seed", None)
    if seed_val is not None and str(seed_val).strip() != "":
        try:
            seed_val = int(seed_val)
        except ValueError:
            print(f"[ERROR] could not convert requested seed value {seed_val} to integer")
            seed_val = None
    else:
        # generate a random seed
        now = datetime.datetime.now()
        seed_val = int(now.timestamp() * 1000)

    # dumping detected configuration
    print(f"n={n}, k={k}, average={average}, sigma={sigma}, input_prec={input_prec_name}, a_distribution={a_distribution}, a_average={a_average}, a_sigma={a_sigma}, b_distribution={b_distribution}, b_average={b_average}, b_sigma={b_sigma}, seed={seed_val}")

    # Generate random vectors
    vectors = generate_vectors(n, k, average, sigma, input_prec=input_prec,
                               a_average=a_average, a_sigma=a_sigma,
                               b_average=b_average, b_sigma=b_sigma,
                               a_distribution=a_distribution,
                               b_distribution=b_distribution,
                               seed=seed_val)

    # Generate golden values (exact dot product)
    golden_values = [exactDotProd(a, b) for (a, b) in vectors]

    results = {}

    for scheme in schemes:
        name = scheme.get("name")
        variant = scheme.get("variant")

        if variant == "correctly_rounded":
            res_prec_name = scheme.get("resPrec", "fp32")
            res = evaluate_errors(vectors, correctlyRoundedDotProd, {
                "resPrec": FORMAT_MAP.get(res_prec_name, singleformat)
            }, golden_values)
            results[name] = res

        elif variant == "approx_mult":
            mult_prec_name = scheme.get("multPrec", "fp16")
            res_prec_name = scheme.get("resPrec", "fp32")
            res = evaluate_errors(vectors, approxMultDotProd, {
                "multPrec": FORMAT_MAP.get(mult_prec_name, halfprecisionformat),
                "resPrec": FORMAT_MAP.get(res_prec_name, singleformat),
            }, golden_values)
            results[name] = res

        elif variant == "approx_mult_acc":
            mult_prec_name = scheme.get("multPrec", "fp16")
            add_prec_name = scheme.get("addPrec", "fp16")
            res_prec_name = scheme.get("resPrec", "fp32")
            res = evaluate_errors(vectors, approxMultBinTreeAccDotProd, {
                "multPrec": FORMAT_MAP.get(mult_prec_name, halfprecisionformat),
                "addPrec": FORMAT_MAP.get(add_prec_name, halfprecisionformat),
                "resPrec": FORMAT_MAP.get(res_prec_name, singleformat),
            }, golden_values)
            results[name] = res

        elif variant == "approx_mult_linear_acc":
            mult_prec_name = scheme.get("multPrec", "fp16")
            add_prec_name = scheme.get("addPrec", "fp16")
            res_prec_name = scheme.get("resPrec", "fp32")
            res = evaluate_errors(vectors, approxMultLinearAccDotProd, {
                "multPrec": FORMAT_MAP.get(mult_prec_name, halfprecisionformat),
                "addPrec": FORMAT_MAP.get(add_prec_name, halfprecisionformat),
                "resPrec": FORMAT_MAP.get(res_prec_name, singleformat),
            }, golden_values)
            results[name] = res

        elif variant == "fma":
            prec_name = scheme.get("fmaPrec", "fp32")
            res_prec_name = scheme.get("resPrec", "fp32")
            res = evaluate_errors(vectors, fmaDotProd, {
                "prec": FORMAT_MAP.get(prec_name, singleformat),
                "resPrec": FORMAT_MAP.get(res_prec_name, singleformat),
            }, golden_values)
            results[name] = res

        elif variant == "bulk_norm":
            bulk_norm_prec = int(scheme.get("bulkNormPrec", 25))
            final_prec = int(scheme.get("finalPrec", 23))
            res = evaluate_errors(vectors, bulkNormDotProd, {
                "bulkNormPrec": bulk_norm_prec,
                "finalPrec": final_prec,
            }, golden_values)
            results[name] = res

    for k in results:
        results[k]["seed"] = seed_val

    return jsonify(results)


if __name__ == "__main__":
    print("Application configuration:")
    print(f"  MAX_N is {MAX_N}")
    print(f"  DEBUG is {DEBUG}")
    app.run(debug=DEBUG, port=5001)
