import argparse
import sys

from CASER import predict, predict_subtype
import utils.ML_models as ml


MODELS = {
    "tri_training": ml.tri_training,
    "co_training": ml.co_training,
    "semiboost": ml.semiboost,
    "lapsvm": ml.lapsvm,
    "assemble": ml.assemble,
    "tsvm": ml.tsvm,
    "ssgmm": ml.ssgmm,
    "svc": ml.svc,
    "rf": ml.rf,
    "lr": ml.LR,
    "xgboost": ml.Xgboost,
    "gdbt": ml.GDBT,
}


def parse_csv_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="CASER command-line runner (train + predict)."
    )
    parser.add_argument(
        "--mode",
        choices=["single", "subtype"],
        default="single",
        help="single: one labeled + one unlabeled file. subtype: multiple files.",
    )
    parser.add_argument(
        "--known",
        required=True,
        help="Path(s) to known labeled gene file(s). Use comma-separated list for subtype.",
    )
    parser.add_argument(
        "--unknown",
        required=True,
        help="Path(s) to unknown gene file(s). Use comma-separated list for subtype.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path(s). Use comma-separated list for subtype.",
    )
    parser.add_argument(
        "--model",
        default="tri_training",
        choices=sorted(MODELS.keys()),
        help="Model to use.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print available model names and exit.",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for name in sorted(MODELS.keys()):
            print(f"- {name}")
        return 0

    model_fn = MODELS[args.model]

    if args.mode == "single":
        if "," in args.known or "," in args.unknown or "," in args.out:
            print("Error: single mode does not accept comma-separated lists.", file=sys.stderr)
            return 2
        predict(args.known, args.unknown, args.out, model=model_fn)
        print("Done. Results saved to:", args.out)
        return 0

    known_list = parse_csv_list(args.known)
    unknown_list = parse_csv_list(args.unknown)
    out_list = parse_csv_list(args.out)

    if not (len(known_list) == len(unknown_list) == len(out_list)):
        print("Error: subtype mode requires equal counts for --known, --unknown, and --out.", file=sys.stderr)
        return 2

    predict_subtype(known_list, unknown_list, out_list, model=model_fn)
    print("Done. Results saved to:", ", ".join(out_list))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
