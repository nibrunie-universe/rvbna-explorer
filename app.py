from flask import Flask, request, jsonify, send_from_directory
from rvbna_web import (
    correctlyRoundedDotProd,
    approxMultDotProd,
    approxMultAccDotProd,
    fmaDotProd,
    bulkNormDotProd,
    generate_vectors,
    evaluate_errors,
    FORMAT_MAP,
    singleformat,
    halfprecisionformat,
)

app = Flask(__name__, static_folder="static")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    data = request.json

    # Distribution config
    data_source = data.get("dataSource", "random")
    n = int(data.get("n", 1000))
    k = int(data.get("k", 2))
    average = float(data.get("average", 5.0))
    sigma = float(data.get("sigma", 5.0))
    input_prec_name = data.get("inputPrec", "fp16")
    input_prec = FORMAT_MAP.get(input_prec_name, halfprecisionformat)

    # Clamp n to avoid extreme computation
    n = min(n, 50000)

    if data_source == "tiny_llama":
        import json
        import os
        from pysollya import round_sol, RN
        
        file_path = os.path.join(os.path.dirname(__file__), "tiny_llama_vectors.json")
        if not os.path.exists(file_path):
            return jsonify({"error": "tiny_llama_vectors.json not found. Please run extract_tinyllama_vectors.py first."}), 404
            
        with open(file_path, "r") as f:
            vector_pairs = json.load(f)
            
        # We need to round them to the input precision, similar to random vectors.
        vectors = []
        for pair in vector_pairs:
            a = [round_sol(val, input_prec, RN) for val in pair["a"]]
            b = [round_sol(val, input_prec, RN) for val in pair["b"]]
            vectors.append((a, b))
            
        # Limit to requested n
        vectors = vectors[:n]
    else:
        # Per-vector distribution parameters
        a_distribution = data.get("aDistribution", "gaussian")
        a_average = float(data.get("aAverage", average))
        a_sigma = float(data.get("aSigma", sigma))
        b_distribution = data.get("bDistribution", "gaussian")
        b_average = float(data.get("bAverage", average))
        b_sigma = float(data.get("bSigma", sigma))

        # Generate random vectors
        vectors = generate_vectors(n, k, average, sigma, input_prec=input_prec,
                                   a_average=a_average, a_sigma=a_sigma,
                                   b_average=b_average, b_sigma=b_sigma,
                                   a_distribution=a_distribution,
                                   b_distribution=b_distribution)

    # Generate golden values (exact dot product)
    golden_values = [correctlyRoundedDotProd(a, b) for (a, b) in vectors]

    results = {}
    schemes = data.get("schemes", [])

    for scheme in schemes:
        name = scheme.get("name")
        variant = scheme.get("variant")

        if variant == "exact":
            res = evaluate_errors(vectors, correctlyRoundedDotProd, {}, golden_values)
            # we do not expect any errors here, if any error is found, we should log it for debug
            print("exact_count", res["exact_count"])
            print("n", n)
            print("max", res["max"])
            if res["max"] > 0 or res["exact_count"] != n:
                # print the sorted errors which exceed 0
                for err in res["sorted_rel_errors"]:
                    if err > 0:
                        print("Exact dot product found with error:", err)
                app.logger.error(f"Exact dot product found with errors: {res}")
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
            res = evaluate_errors(vectors, approxMultAccDotProd, {
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

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
