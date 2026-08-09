import argparse
import sys
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

def main():
    parser = argparse.ArgumentParser(description="RVBNA Explorer Command Line Tool")
    parser.add_argument("-n", "--samples", type=int, default=1000, help="Number of samples (N)")
    parser.add_argument("-k", "--vectorsize", type=int, default=2, help="Vector size (K)")
    parser.add_argument("--avg", type=float, default=5.0, help="Mean (μ) for both vectors")
    parser.add_argument("--sigma", type=float, default=5.0, help="StdDev (σ) for both vectors")
    parser.add_argument("--input-prec", type=str, default="fp16", choices=["bf16", "fp16", "fp32", "fp64"], help="Input precision")
    
    args = parser.parse_args()
    
    input_prec = FORMAT_MAP.get(args.input_prec, halfprecisionformat)
    
    print(f"Generating {args.samples} samples of size {args.vectorsize}...")
    vectors = generate_vectors(
        args.samples, args.vectorsize, args.avg, args.sigma,
        input_prec=input_prec,
        a_average=args.avg, a_sigma=args.sigma,
        b_average=args.avg, b_sigma=args.sigma,
        a_distribution="gaussian", b_distribution="gaussian"
    )
    
    print("Computing golden values (exact dot product)...")
    golden_values = [correctlyRoundedDotProd(a, b) for (a, b) in vectors]
    
    schemes = [
        {"name": "Exact", "variant": "exact"},
        {"name": "FP MUL [FP16] + Exact Acc", "variant": "approx_mult", "multPrec": "fp16", "resPrec": "fp32"},
        {"name": "FP MUL [FP16] + Add Tree [FP16]", "variant": "approx_mult_acc", "multPrec": "fp16", "addPrec": "fp16", "resPrec": "fp32"},
        {"name": "FMA [FP32]", "variant": "fma", "fmaPrec": "fp32", "resPrec": "fp32"},
        {"name": "Bulk Norm [Fixed 24, Final 23]", "variant": "bulk_norm", "bulkNormPrec": 24, "finalPrec": 23},
    ]

    results = []

    for scheme in schemes:
        name = scheme.get("name")
        variant = scheme.get("variant")
        
        if variant == "exact":
            res = evaluate_errors(vectors, correctlyRoundedDotProd, {}, golden_values)
        elif variant == "approx_mult":
            res = evaluate_errors(vectors, approxMultDotProd, {
                "multPrec": FORMAT_MAP.get(scheme.get("multPrec"), halfprecisionformat),
                "resPrec": FORMAT_MAP.get(scheme.get("resPrec"), singleformat),
            }, golden_values)
        elif variant == "approx_mult_acc":
            res = evaluate_errors(vectors, approxMultAccDotProd, {
                "multPrec": FORMAT_MAP.get(scheme.get("multPrec"), halfprecisionformat),
                "addPrec": FORMAT_MAP.get(scheme.get("addPrec"), halfprecisionformat),
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
            
        results.append((name, res))

    # Print results as a formatted table
    print("\nResults:")
    print("-" * 235)
    header = f"{'Scheme':<40} | {'Min Error':<12} | {'Max Error':<12} | {'Geo Mean':<12} | {'Exact Hits':<15} | {'Avg Signed Rel':<14} | {'Avg Signed Err':<14} | {'Sum Signed Rel':<14} | {'Sum Signed Err':<14} | {'Pos Errors':<10} | {'Neg Errors':<10} | {'Pos A':<8} | {'Neg A':<8} | {'Pos B':<8} | {'Neg B':<8}"
    print(header)
    print("-" * 235)
    
    # Sort results by geometric mean to match the web app
    results.sort(key=lambda x: x[1]["geometric_mean"])
    
    for name, r in results:
        exact_pct = (r['exact_count'] / args.samples) * 100
        exact_str = f"{exact_pct:.1f}% ({r['exact_count']})"
        row = f"{name:<40} | {r['min']:<12.3e} | {r['max']:<12.3e} | {r['geometric_mean']:<12.3e} | {exact_str:<15} | {r['mean_signed_rel_error']:<14.3e} | {r['mean_signed_error']:<14.3e} | {r['sum_signed_rel_error']:<14.3e} | {r['sum_signed_error']:<14.3e} | {r['pos_count']:<10} | {r['neg_count']:<10} | {r['pos_a_count']:<8} | {r['neg_a_count']:<8} | {r['pos_b_count']:<8} | {r['neg_b_count']:<8}"
        print(row)
        
    print("-" * 235)

if __name__ == "__main__":
    main()
