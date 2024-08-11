#!/usr/bin/env python3

import argparse
from Bio.PDB import PDBParser
from Bio import SeqIO, AlignIO
import numpy as np
import pandas as pd
import os
import math
import sys
import csv

# Custom 3-to-1 amino acid mapping
aa_mapping = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'MSE': 'M', 'HSD': 'H', 'HSE': 'H', 'HSP': 'H',
}


def parse_pdb_sequences_and_coordinates(pdb_file):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_file)
    except Exception as e:
        print(f"Error parsing PDB file: {str(e)}")
        sys.exit(1)

    chains = {}
    for model in structure:
        for chain in model:
            sequence = ""
            coords = {}
            for residue in chain:
                if all(atom in residue for atom in ['N', 'CA', 'C']):
                    try:
                        one_letter = aa_mapping.get(residue.resname, 'X')
                        sequence += one_letter
                        coords[len(sequence)] = {
                            'backbone': np.array([residue['N'].get_coord(),
                                                  residue['CA'].get_coord(),
                                                  residue['C'].get_coord()]),
                            'ca': residue['CA'].get_coord(),
                            'residue': one_letter
                        }
                    except KeyError:
                        print(f"Warning: Unknown residue {residue.resname} in chain {chain.id}. Using 'X'.")
            chains[chain.id] = {'sequence': sequence, 'coords': coords}
    return chains


def find_matching_chains(chains, fasta_sequences):
    matching_chains = []
    for fasta_record in fasta_sequences:
        fasta_seq_str = str(fasta_record.seq).replace('-', '')
        fasta_chain_id = fasta_record.id.split('.')[0]
        for chain_id, chain_data in chains.items():
            pdb_seq_str = chain_data['sequence']
            if pdb_seq_str in fasta_seq_str or fasta_seq_str in pdb_seq_str:
                matching_chains.append((chain_id, fasta_chain_id))
                break
    return matching_chains


def parse_fasta_alignment(fasta_file):
    try:
        alignment = AlignIO.read(fasta_file, "fasta")
    except Exception as e:
        print(f"Error reading FASTA file: {str(e)}")
        sys.exit(1)

    seq1, seq2 = str(alignment[0].seq), str(alignment[1].seq)
    mapping = []
    pos1, pos2 = 0, 0
    for res1, res2 in zip(seq1, seq2):
        if res1 != '-' and res2 != '-':
            mapping.append((pos1 + 1, pos2 + 1))
        if res1 != '-':
            pos1 += 1
        if res2 != '-':
            pos2 += 1
    return mapping


def calculate_residue_rmsd(coords1, coords2):
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


def rmsd_to_similarity(rmsd, max_rmsd):
    return np.exp(-rmsd / max_rmsd)


def load_csv_matrix(matrix_file):
    df = pd.read_csv(matrix_file, index_col=0)
    if not df.index.equals(df.columns):
        raise ValueError("Matrix is not symmetric: row and column labels do not match")
    matrix_dict = {(aa1, aa2): df.loc[aa1, aa2]
                   for aa1 in df.index
                   for aa2 in df.columns}
    return matrix_dict, list(df.index)


def calculate_similarity(score):
    return 1 / (1 + math.exp(-score))


def calculate_pairwise_similarity(seq1, seq2, matrix):
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length")

    similarities = []
    for aa1, aa2 in zip(seq1, seq2):
        if aa1 == '-' and aa2 == '-':
            similarities.append(1.0)
        elif aa1 == '-' or aa2 == '-':
            similarities.append(0.0)
        elif aa1 == aa2:
            similarities.append(1.0)
        else:
            score = matrix.get((aa1, aa2), matrix.get((aa2, aa1), 0))
            similarity = calculate_similarity(score)
            similarities.append(similarity)
    return similarities


def calculate_residue_lddt(predicted_points, true_points, cutoff=15.0):
    dmat_true = np.sqrt(np.sum((true_points[:, None] - true_points[None, :]) ** 2, axis=-1))
    dmat_predicted = np.sqrt(np.sum((predicted_points[:, None] - predicted_points[None, :]) ** 2, axis=-1))

    dists_to_score = (dmat_true < cutoff).astype(np.float32) * (1 - np.eye(dmat_true.shape[0]))

    dist_l1 = np.abs(dmat_true - dmat_predicted)

    score = 0.25 * ((dist_l1 < 0.5).astype(np.float32) +
                    (dist_l1 < 1.0).astype(np.float32) +
                    (dist_l1 < 2.0).astype(np.float32) +
                    (dist_l1 < 4.0).astype(np.float32))

    norm = 1.0 / (1e-10 + np.sum(dists_to_score, axis=-1))
    lddt_scores = norm * (1e-10 + np.sum(dists_to_score * score, axis=-1))

    return lddt_scores


def evaluate_alignment_quality(pdb_file, fasta_file, matrix_file, max_rmsd=10.0, lddt_cutoff=15.0, weights=None):
    if weights is None:
        weights = {'rmsd': 1, 'seq': 1, 'lddt': 1}

    chains = parse_pdb_sequences_and_coordinates(pdb_file)
    fasta_sequences = list(SeqIO.parse(fasta_file, "fasta"))
    matching_chains = find_matching_chains(chains, fasta_sequences)

    if len(matching_chains) != 2:
        raise ValueError(f"Expected to find 2 matching chains, but found {len(matching_chains)}")

    chain1_id, chain2_id = matching_chains[0][0], matching_chains[1][0]
    alignment = parse_fasta_alignment(fasta_file)

    coords1 = chains[chain1_id]['coords']
    coords2 = chains[chain2_id]['coords']

    # RMSD and Similarity calculation
    rmsd_similarities = []
    for a in alignment:
        if a[0] in coords1 and a[1] in coords2:
            rmsd = calculate_residue_rmsd(coords1[a[0]]['backbone'], coords2[a[1]]['backbone'])
            similarity = rmsd_to_similarity(rmsd, max_rmsd)
            rmsd_similarities.append((a[0], a[1], rmsd, similarity))

    # Sequence-based similarity calculation
    matrix, aa_order = load_csv_matrix(matrix_file)
    seq1 = ''.join([chains[chain1_id]['sequence'][i - 1] for i, _ in alignment])
    seq2 = ''.join([chains[chain2_id]['sequence'][i - 1] for _, i in alignment])
    seq_similarities = calculate_pairwise_similarity(seq1, seq2, matrix)

    # LDDT calculation
    aligned_coords1 = np.array([coords1[a[0]]['ca'] for a in alignment if a[0] in coords1 and a[1] in coords2])
    aligned_coords2 = np.array([coords2[a[1]]['ca'] for a in alignment if a[0] in coords1 and a[1] in coords2])
    lddt_scores = calculate_residue_lddt(aligned_coords1, aligned_coords2, lddt_cutoff)

    # Combine all scores
    combined_scores = []
    for i, (res1, res2) in enumerate(alignment):
        rmsd_sim = next((s for r1, r2, _, s in rmsd_similarities if r1 == res1 and r2 == res2), None)
        seq_sim = seq_similarities[i] if i < len(seq_similarities) else None
        lddt = lddt_scores[i] if i < len(lddt_scores) else None

        scores = []
        weights_sum = 0
        if rmsd_sim is not None:
            scores.append(rmsd_sim * weights['rmsd'])
            weights_sum += weights['rmsd']
        if seq_sim is not None:
            scores.append(seq_sim * weights['seq'])
            weights_sum += weights['seq']
        if lddt is not None:
            scores.append(lddt * weights['lddt'])
            weights_sum += weights['lddt']

        avg_score = sum(scores) / weights_sum if weights_sum > 0 else None

        combined_scores.append({
            'residue_index_chain1': res1,
            'residue_chain1': coords1[res1]['residue'] if res1 in coords1 else '-',
            'residue_index_chain2': res2,
            'residue_chain2': coords2[res2]['residue'] if res2 in coords2 else '-',
            'rmsd_similarity': rmsd_sim,
            'seq_similarity': seq_sim,
            'lddt_score': lddt,
            'average_score': avg_score
        })

    # Calculate global scores
    global_rmsd_sim = np.mean([s['rmsd_similarity'] for s in combined_scores if s['rmsd_similarity'] is not None])
    global_seq_sim = np.mean(seq_similarities)
    global_lddt = np.mean(lddt_scores)
    global_average = (global_rmsd_sim * weights['rmsd'] + global_seq_sim * weights['seq'] + global_lddt * weights[
        'lddt']) / sum(weights.values())

    return combined_scores, chain1_id, chain2_id, global_rmsd_sim, global_seq_sim, global_lddt, global_average


def main(pdb_file, fasta_file, matrix="lg_matrix_full.csv", matrix_dir="./matrix", max_rmsd=10.0, lddt_cutoff=15.0,
         output=None, weights=None):
    try:
        matrix_path = os.path.join(matrix_dir, matrix)
        if not os.path.exists(matrix_path):
            raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

        results, chain1_id, chain2_id, global_rmsd_sim, global_seq_sim, global_lddt, global_average = evaluate_alignment_quality(
            pdb_file, fasta_file, matrix_path, max_rmsd, lddt_cutoff, weights
        )

        # Prepare data for CSV output
        csv_data = [
            ["Chain1_Residue_Index", "Chain1_Residue", "Chain2_Residue_Index", "Chain2_Residue", "RMSD_Similarity",
             "Seq_Similarity", "LDDT_Score", "Average_Score"]
        ]
        for result in results:
            csv_data.append([
                result['residue_index_chain1'],
                result['residue_chain1'],
                result['residue_index_chain2'],
                result['residue_chain2'],
                f"{result['rmsd_similarity']:.3f}" if result['rmsd_similarity'] is not None else "N/A",
                f"{result['seq_similarity']:.3f}" if result['seq_similarity'] is not None else "N/A",
                f"{result['lddt_score']:.3f}" if result['lddt_score'] is not None else "N/A",
                f"{result['average_score']:.3f}" if result['average_score'] is not None else "N/A"
            ])

        # Add global scores
        csv_data.extend([
            [],
            ["Global Scores"],
            ["Metric", "Score"],
            ["Global RMSD Similarity", f"{global_rmsd_sim:.3f}"],
            ["Global Sequence Similarity", f"{global_seq_sim:.3f}"],
            ["Global LDDT Score", f"{global_lddt:.3f}"],
            ["Overall Global Average", f"{global_average:.3f}"]
        ])

        # Write to CSV file
        if output:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(output), exist_ok=True)

            with open(output, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Identified chains: Chain 1 =", chain1_id, "Chain 2 =", chain2_id])
                writer.writerows(csv_data)

            print(f"Results written to {output}")

        return results, chain1_id, chain2_id, global_rmsd_sim, global_seq_sim, global_lddt, global_average

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive alignment quality evaluation")
    parser.add_argument("pdb_file", help="Path to the PDB file")
    parser.add_argument("fasta_file", help="Path to the FASTA alignment file")
    parser.add_argument("--matrix", default="LG.csv", help="Name of the CSV matrix file")
    parser.add_argument("--matrix_dir", default="./matrix", help="Directory containing matrix files")
    parser.add_argument("--max_rmsd", type=float, default=10.0, help="Maximum RMSD for similarity calculation")
    parser.add_argument("--lddt_cutoff", type=float, default=15.0, help="Cutoff distance for LDDT calculation")
    parser.add_argument("--output", help="Path to output CSV file (optional)")
    parser.add_argument("--rmsd_weight", type=float, default=1.0, help="Weight for RMSD similarity")
    parser.add_argument("--seq_weight", type=float, default=1.0, help="Weight for sequence similarity")
    parser.add_argument("--lddt_weight", type=float, default=1.0, help="Weight for LDDT score")
    args = parser.parse_args()

    weights = {
        'rmsd': args.rmsd_weight,
        'seq': args.seq_weight,
        'lddt': args.lddt_weight
    }

    main(args.pdb_file, args.fasta_file, args.matrix, args.matrix_dir, args.max_rmsd, args.lddt_cutoff, args.output, weights)