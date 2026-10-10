import matplotlib.pyplot as plt
import os
import argparse
import json
from rvbna_web import (
    correctlyRoundedDotProd,
    approxMultDotProd,
    approxMultBinTreeAccDotProd,
    fmaDotProd,
    bulkNormDotProd,
    generate_vectors,
    evaluate_errors_vector,
    FORMAT_MAP,
    singleformat,
    halfprecisionformat,
    bfloat16format
)

def parse_format(s):
    if s == "fp32":
        return singleformat
    elif s == "bf16":
        return bfloat16format
    elif s == "fp16":
        return halfprecisionformat
    else:
        raise ValueError(f"Unknown format: {s}")

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
    parser.add_argument("--input-format", type=parse_format, default=bfloat16format, help="Input format")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
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
    
    states = {s["name"]: None for s in schemes}
    last_n = 0

    n_max = args.n_end

    print("Generating random input vectors ...")
    full_vectors = generate_vectors(
            n_max, k, avg, sigma,
            input_prec=args.input_format,
            a_average=avg, a_sigma=sigma,
            b_average=avg, b_sigma=sigma,
            a_distribution="gaussian", b_distribution="gaussian",
            seed=args.seed
        )
    print("Evaluating correctly rounded dot product...")
    full_golden_values = [correctlyRoundedDotProd(a, b) for (a, b) in full_vectors]

    full_res_vectors = {}
    print(f"{len(schemes)} scheme(s) found")
    print("Pre-computing full result vectors for all variants...")
    for scheme in schemes:
        name = scheme.get("name")
        variant = scheme.get("variant")
        
        if variant == "exact":
            full_res_vectors[name] = full_golden_values
        elif variant == "approx_mult":
            kwargs = {
                "multPrec": FORMAT_MAP.get(scheme.get("multPrec"), bfloat16format),
                "resPrec": FORMAT_MAP.get(scheme.get("resPrec"), singleformat),
            }
            full_res_vectors[name] = [approxMultDotProd(a, b, **kwargs) for (a, b) in full_vectors]
        elif variant == "approx_mult_acc":
            kwargs = {
                "multPrec": FORMAT_MAP.get(scheme.get("multPrec"), bfloat16format),
                "addPrec": FORMAT_MAP.get(scheme.get("addPrec"), bfloat16format),
                "resPrec": FORMAT_MAP.get(scheme.get("resPrec"), singleformat),
            }
            full_res_vectors[name] = [approxMultBinTreeAccDotProd(a, b, **kwargs) for (a, b) in full_vectors]
        elif variant == "fma":
            kwargs = {
                "prec": FORMAT_MAP.get(scheme.get("fmaPrec"), singleformat),
                "resPrec": FORMAT_MAP.get(scheme.get("resPrec"), singleformat),
            }
            full_res_vectors[name] = [fmaDotProd(a, b, **kwargs) for (a, b) in full_vectors]
        elif variant == "bulk_norm":
            kwargs = {
                "bulkNormPrec": scheme.get("bulkNormPrec"),
                "finalPrec": scheme.get("finalPrec"),
            }
            full_res_vectors[name] = [bulkNormDotProd(a, b, **kwargs) for (a, b) in full_vectors]

    for n in ns:
        print(f"Evaluating n={n}...")
        vectors = full_vectors[last_n:n]
        golden_values = full_golden_values[last_n:n]
        
        for scheme in schemes:
            name = scheme.get("name")
            res_vector = full_res_vectors[name][last_n:n]
            
            states[name] = evaluate_errors_vector(vectors, res_vector, golden_values, state=states[name])
                
            results_sum[name].append(states[name]["sum_signed_error"])
            results_avg[name].append(states[name]["mean_signed_error"])
            
        last_n = n

    print("\nError Direction Summary (aggregated over all N):")
    print("-" * 148)
    print(f"{'Scheme':<40} | {'Pos Errors':<10} | {'Neg Errors':<10} | {'Pos A':<8} | {'Neg A':<8} | {'Pos B':<8} | {'Neg B':<8} | {'Exact Pos':<9} | {'Exact Neg':<9} | {'Opp Sign':<8}")
    print("-" * 148)
    for scheme in schemes:
        name = scheme.get("name")
        st = states[name]
        print(f"{name:<40} | {st['pos_count']:<10} | {st['neg_count']:<10} | {st['pos_a_count']:<8} | {st['neg_a_count']:<8} | {st['pos_b_count']:<8} | {st['neg_b_count']:<8} | {st['exact_pos_count']:<9} | {st['exact_neg_count']:<9} | {st['opposite_sign_count']:<8}")
    print("-" * 148 + "\n")

    plt.figure(figsize=(15, 6))
    
    # Plot Sum of Signed Error
    plt.subplot(1, 2, 1)
    for name, data in results_sum.items():
        plt.plot(ns, data, marker='o', label=name)
    plt.xlabel("n (number of samples)")
    plt.ylabel("Sum of Signed Error")
    plt.title(f"Sum of Signed Error vs N\n(k={k}, avg={avg}, sigma={sigma}, input={args.input_format})")
    plt.legend()
    plt.grid(True)
    
    # Plot Avg Signed Error
    plt.subplot(1, 2, 2)
    for name, data in results_avg.items():
        plt.plot(ns, data, marker='o', label=name)
    plt.xlabel("n (number of samples)")
    plt.ylabel("Average Signed Error")
    plt.title(f"Average Signed Error vs N\n(k={k}, avg={avg}, sigma={sigma}, input={args.input_format})")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    output_path = os.path.abspath(args.output)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
