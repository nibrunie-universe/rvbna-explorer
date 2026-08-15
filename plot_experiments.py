import matplotlib.pyplot as plt
import os
import argparse
import json
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
    bfloat16format
)

def main():
    parser = argparse.ArgumentParser(description="Generate plot of signed absolute errors")
    parser.add_argument("--variants", type=str, help="Path to JSON file containing variants configuration")
    parser.add_argument("-k", "--vectorsize", type=int, default=4, help="Vector size (K)")
    parser.add_argument("--n-start", type=int, default=1000, help="Start of N range")
    parser.add_argument("--n-end", type=int, default=20000, help="End of N range (inclusive)")
    parser.add_argument("--n-step", type=int, default=1000, help="Step of N range")
    parser.add_argument("--avg", type=float, default=0.0, help="Distribution average")
    parser.add_argument("--sigma", type=float, default=10.0, help="Distribution sigma")
    parser.add_argument("-o", "--output", type=str, default="signed_error_plot.png", help="Output plot filename")
    args = parser.parse_args()

    if args.variants:
        with open(args.variants, 'r') as f:
            schemes = json.load(f)
    else:
        schemes = [
            {"name": "Exact", "variant": "exact"},
            {"name": "Bulk Norm [Fixed 24, Final 23]", "variant": "bulk_norm", "bulkNormPrec": 24, "finalPrec": 23},
        ]

    ns = list(range(args.n_start, args.n_end + 1, args.n_step))
    k = args.vectorsize
    avg = args.avg
    sigma = args.sigma

    results_sum = {s["name"]: [] for s in schemes}
    results_avg = {s["name"]: [] for s in schemes}
    
    total_pos = {s["name"]: 0 for s in schemes}
    total_neg = {s["name"]: 0 for s in schemes}
    total_pos_a = {s["name"]: 0 for s in schemes}
    total_neg_a = {s["name"]: 0 for s in schemes}
    total_pos_b = {s["name"]: 0 for s in schemes}
    total_neg_b = {s["name"]: 0 for s in schemes}
    total_exact_pos = {s["name"]: 0 for s in schemes}
    total_exact_neg = {s["name"]: 0 for s in schemes}
    total_opposite_sign = {s["name"]: 0 for s in schemes}

    for n in ns:
        print(f"Evaluating n={n}...")
        vectors = generate_vectors(
            n, k, avg, sigma,
            input_prec=bfloat16format,
            a_average=avg, a_sigma=sigma,
            b_average=avg, b_sigma=sigma,
            a_distribution="gaussian", b_distribution="gaussian"
        )
        golden_values = [correctlyRoundedDotProd(a, b) for (a, b) in vectors]
        
        for scheme in schemes:
            name = scheme.get("name")
            variant = scheme.get("variant")
            
            if variant == "exact":
                res = evaluate_errors(vectors, correctlyRoundedDotProd, {}, golden_values)
            elif variant == "approx_mult":
                res = evaluate_errors(vectors, approxMultDotProd, {
                    "multPrec": FORMAT_MAP.get(scheme.get("multPrec"), bfloat16format),
                    "resPrec": FORMAT_MAP.get(scheme.get("resPrec"), singleformat),
                }, golden_values)
            elif variant == "approx_mult_acc":
                res = evaluate_errors(vectors, approxMultAccDotProd, {
                    "multPrec": FORMAT_MAP.get(scheme.get("multPrec"), bfloat16format),
                    "addPrec": FORMAT_MAP.get(scheme.get("addPrec"), bfloat16format),
                    "resPrec": FORMAT_MAP.get(scheme.get("resPrec"), singleformat),
                }, golden_values)
            elif variant == "fma":
                res = evaluate_errors(vectors, fmaDotProd, {
                    "prec": FORMAT_MAP.get(scheme.get("fmaPrec"), singleformat),
                    "resPrec": FORMAT_MAP.get(scheme.get("resPrec"), singleformat),
                }, golden_values)
            elif variant == "bulk_norm":
                res = evaluate_errors(vectors, bulkNormDotProd, {
                    "bulkNormPrec": scheme.get("bulkNormPrec"),
                    "finalPrec": scheme.get("finalPrec"),
                }, golden_values)
                
            results_sum[name].append(res["sum_signed_error"])
            results_avg[name].append(res["mean_signed_error"])
            
            total_pos[name] += res["pos_count"]
            total_neg[name] += res["neg_count"]
            total_pos_a[name] += res["pos_a_count"]
            total_neg_a[name] += res["neg_a_count"]
            total_pos_b[name] += res["pos_b_count"]
            total_neg_b[name] += res["neg_b_count"]
            total_exact_pos[name] += res["exact_pos_count"]
            total_exact_neg[name] += res["exact_neg_count"]
            total_opposite_sign[name] += res["opposite_sign_count"]

    print("\nError Direction Summary (aggregated over all N):")
    print("-" * 148)
    print(f"{'Scheme':<40} | {'Pos Errors':<10} | {'Neg Errors':<10} | {'Pos A':<8} | {'Neg A':<8} | {'Pos B':<8} | {'Neg B':<8} | {'Exact Pos':<9} | {'Exact Neg':<9} | {'Opp Sign':<8}")
    print("-" * 148)
    for scheme in schemes:
        name = scheme.get("name")
        print(f"{name:<40} | {total_pos[name]:<10} | {total_neg[name]:<10} | {total_pos_a[name]:<8} | {total_neg_a[name]:<8} | {total_pos_b[name]:<8} | {total_neg_b[name]:<8} | {total_exact_pos[name]:<9} | {total_exact_neg[name]:<9} | {total_opposite_sign[name]:<8}")
    print("-" * 148 + "\n")

    plt.figure(figsize=(15, 6))
    
    # Plot Sum of Signed Error
    plt.subplot(1, 2, 1)
    for name, data in results_sum.items():
        plt.plot(ns, data, marker='o', label=name)
    plt.xlabel("n (number of samples)")
    plt.ylabel("Sum of Signed Error")
    plt.title(f"Sum of Signed Error vs N\n(k={k}, avg={avg}, sigma={sigma}, input=bf16)")
    plt.legend()
    plt.grid(True)
    
    # Plot Avg Signed Error
    plt.subplot(1, 2, 2)
    for name, data in results_avg.items():
        plt.plot(ns, data, marker='o', label=name)
    plt.xlabel("n (number of samples)")
    plt.ylabel("Average Signed Error")
    plt.title(f"Average Signed Error vs N\n(k={k}, avg={avg}, sigma={sigma}, input=bf16)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    output_path = os.path.abspath(args.output)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
