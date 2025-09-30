import pandas as pd
import numpy as np
import argparse
import torch
import pickle
import os
from pathlib import Path
import esm

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate input data for PLaNET-X using ESM2. "
            "For multi-chain proteins (comma-separated), embeddings are extracted per chain and concatenated."
        )
    )
    parser.add_argument(
        "--input", "-i", required=True, type=str,
        help="TSV file with columns: id, seq, [binding_site]."
    )
    parser.add_argument(
        "--output", "-o", required=True, type=str,
        help="Output pickle path. Saved as (IDs, seqs, [binding_sites], prots_feat_list)."
    )
    parser.add_argument(
    "--labels", "-l",
    dest="labels",            # 변수명 지정
    action="store_true",      # 옵션 붙이면 True
    help="Use when generating training data with binding site labels"
    )

    parser.add_argument(
        "--no-labels",
        dest="labels",
        action="store_false",     # 옵션 붙이면 False
        help="Use when generating test data without binding site labels"
    )

    parser.set_defaults(labels=True)  # 기본값을 True로 할지 False로 할지 결정
    args = parser.parse_args()

    input_abspath = os.path.abspath(args.input)
    if not os.path.isfile(input_abspath):
        raise IOError(f"Please check input file path; {input_abspath} does not exist")

    output_abspath = os.path.abspath(args.output)
    out_dir = os.path.abspath(os.path.dirname(output_abspath))
    if not os.path.isdir(out_dir):
        raise IOError(f"Please check output dir path; {out_dir} does not exist")

    print("1. Load data ...")
    prots_df = pd.read_csv(input_abspath, sep="\t")
    if args.labels:
        IDs = prots_df.iloc[:, 0].astype(str).values
        seqs = prots_df.iloc[:, 1].astype(str).values
        binding_sites = prots_df.iloc[:, 2].astype(str).values
    else:
        IDs = prots_df.iloc[:, 0].astype(str).values
        seqs = prots_df.iloc[:, 1].astype(str).values

    print("2. Load pretrained ESM2 model ...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    print("3. Compute embeddings (per chain, then concat) ...")
    prots_feat_list = []

    with torch.no_grad():
        for seq in seqs:
            # split to chain seqs (same rule as ProtT5 script)
            chain_seqs = [s.strip() for s in seq.split(",") if s.strip()]
            all_embeddings = []

            for chain in chain_seqs:
                # ESM2: simple tuple list -> batch_converter
                data = [("protein", chain)]
                labels, strs, tokens = batch_converter(data)  # labels/strs not used
                tokens = tokens.to(device)

                out = model(tokens, repr_layers=[33], return_contacts=False)
                token_repr = out["representations"][33]        # [1, L+2, 1280]
                emb = token_repr[0, 1:-1]                     # drop [CLS], [EOS]
                emb_np = emb.detach().cpu().numpy().astype(np.float32)  # [L, 1280]

                # append chain features in order
                all_embeddings.extend(emb_np)

            # concat chains along length -> (L_total, 1280)
            prots_feat_list.append(np.array(all_embeddings, dtype=np.float32))

    print("4. Save to pickle ...")
    with open(output_abspath, "wb") as f:
        if args.labels:
            pickle.dump((IDs, seqs, binding_sites, prots_feat_list), f)
        else:
            pickle.dump((IDs, seqs, prots_feat_list), f)

    print(f"Done. Saved: {output_abspath}")

if __name__ == "__main__":
    main()
