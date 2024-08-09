#!/usr/bin/env python3

import argparse
from Bio import SeqIO
import numpy as np
import pandas as pd
import os
import math


def load_csv_matrix(matrix_file):
    # Read the CSV file
    df = pd.read_csv(matrix_file, index_col=0)

    # Ensure the matrix is symmetric
    if not df.index.equals(df.columns):
        raise ValueError("Matrix is not symmetric: row and column labels do not match")

    # Convert to a dictionary for quick lookups
    matrix_dict = {(aa1, aa2): df.loc[aa1, aa2]
                   for aa1 in df.index
                   for aa2 in df.columns}

    return matrix_dict, list(df.index)


def calculate_similarity(score):
    # Transform log-odds score to a similarity score
    return 1 / (1 + math.exp(-score))


def calculate_pairwise_similarity(seq1, seq2, matrix):
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length")

    similarities = []
    for aa1, aa2 in zip(seq1, seq2):
        if aa1 == '-' and aa2 == '-':
            similarities.append(1.0)  # Gap-to-gap alignment
        elif aa1 == '-' or aa2 == '-':
            similarities.append(0.0)  # Gap-to-AA alignment
        elif aa1 == aa2:
            similarities.append(1.0)  # Identical amino acids
        else:
            score = matrix.get((aa1, aa2), matrix.get((aa2, aa1), 0))
            similarity = calculate_similarity(score)
            similarities.append(similarity)

    return similarities


def main():
    parser = argparse.ArgumentParser(description="Calculate pairwise amino acid similarity using CSV matrix")
    parser.add_argument("fasta_file", help="Path to the FASTA alignment file")
    parser.add_argument("--matrix", default="lg_matrix_full.csv", help="Name of the CSV matrix file")
    parser.add_argument("--matrix_dir", default="./matrix", help="Directory containing matrix files")
    parser.add_argument("--output", help="Path to output file (optional)")
    args = parser.parse_args()

    try:
        # Load the substitution matrix
        matrix_path = os.path.join(args.matrix_dir, args.matrix)
        if not os.path.exists(matrix_path):
            raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

        matrix, aa_order = load_csv_matrix(matrix_path)

        # Read the FASTA file
        with open(args.fasta_file, 'r') as f:
            sequences = list(SeqIO.parse(f, 'fasta'))

        if len(sequences) != 2:
            raise ValueError("FASTA file must contain exactly two sequences")

        seq1, seq2 = str(sequences[0].seq), str(sequences[1].seq)

        # Calculate pairwise similarities
        similarities = calculate_pairwise_similarity(seq1, seq2, matrix)

        # Prepare output
        output_lines = [
            f"Positional amino acid similarities (using {args.matrix} matrix):",
            "Position\tAmino Acid 1\tAmino Acid 2\tSimilarity Score"
        ]

        for i, (aa1, aa2, sim) in enumerate(zip(seq1, seq2, similarities), start=1):
            output_lines.append(f"{i}\t{aa1}\t{aa2}\t{sim:.4f}")

        # Calculate and add global similarity
        global_similarity = np.mean(similarities)
        output_lines.append(f"\nGlobal Similarity Score: {global_similarity:.4f}")

        # Join all lines with newline characters
        output_text = '\n'.join(output_lines)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_text)
            print(f"Results written to {args.output}")
        else:
            print(output_text)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()