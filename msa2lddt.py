#!/usr/bin/env python3

import argparse
from Bio.PDB import PDBParser
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import AlignIO
import numpy as np
import sys

# Custom 3-to-1 amino acid mapping
aa_mapping = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    # Add any non-standard residues here
    'MSE': 'M',  # Selenomethionine
    'HSD': 'H',  # Histidine delta
    'HSE': 'H',  # Histidine epsilon
    'HSP': 'H',  # Histidine protonated
    # Add more mappings as needed
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
                if residue.has_id('CA'):
                    try:
                        one_letter = aa_mapping.get(residue.resname, 'X')
                        sequence += one_letter
                        coords[len(sequence)] = residue['CA'].get_coord()
                    except KeyError:
                        print(f"Warning: Unknown residue {residue.resname} in chain {chain.id}. Using 'X'.")
            chains[chain.id] = {'sequence': sequence, 'coords': coords}

    return chains


def find_matching_chains(chains, fasta_sequences):
    matching_chains = []
    for fasta_record in fasta_sequences:
        fasta_seq_str = str(fasta_record.seq).replace('-', '')
        fasta_chain_id = fasta_record.id.split('.')[0]  # Extracts 'btAID' from 'btAID.pdb'

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
            mapping.append((pos1 + 1, pos2 + 1))  # Use 1-based indexing
        if res1 != '-':
            pos1 += 1
        if res2 != '-':
            pos2 += 1
    return mapping


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


def evaluate_alignment_quality(pdb_file, fasta_file, cutoff=15.0):
    # Parse PDB file
    chains = parse_pdb_sequences_and_coordinates(pdb_file)

    # Parse FASTA file
    try:
        fasta_sequences = list(SeqIO.parse(fasta_file, "fasta"))
    except Exception as e:
        print(f"Error parsing FASTA file: {str(e)}")
        sys.exit(1)

    # Find matching chains
    matching_chains = find_matching_chains(chains, fasta_sequences)

    if len(matching_chains) != 2:
        print("Debugging information:")
        print("PDB Chains:")
        for chain_id, chain_data in chains.items():
            print(f"Chain {chain_id}: {chain_data['sequence'][:50]}...")
        print("\nFASTA Sequences:")
        for record in fasta_sequences:
            print(f"{record.id}: {str(record.seq)[:50]}...")
        raise ValueError(f"Expected to find 2 matching chains, but found {len(matching_chains)}")

    chain1_id, chain2_id = matching_chains[0][0], matching_chains[1][0]

    # Parse alignment
    alignment = parse_fasta_alignment(fasta_file)

    # Extract coordinates
    coords1 = chains[chain1_id]['coords']
    coords2 = chains[chain2_id]['coords']

    aligned_coords1 = np.array([coords1[a[0]] for a in alignment if a[0] in coords1 and a[1] in coords2])
    aligned_coords2 = np.array([coords2[a[1]] for a in alignment if a[0] in coords1 and a[1] in coords2])

    if len(aligned_coords1) == 0 or len(aligned_coords2) == 0:
        raise ValueError("No matching coordinates found between the PDB and FASTA alignment")

    lddt_scores = calculate_residue_lddt(aligned_coords1, aligned_coords2, cutoff)

    alignment_quality = [
        {
            'residue_index_chain1': alignment[i][0],
            'residue_index_chain2': alignment[i][1],
            'lddt_score': score
        }
        for i, score in enumerate(lddt_scores)
    ]

    return alignment_quality, chain1_id, chain2_id


def main():
    parser = argparse.ArgumentParser(description="Evaluate alignment quality using LDDT scores")
    parser.add_argument("pdb_file", help="Path to the PDB file")
    parser.add_argument("fasta_file", help="Path to the FASTA alignment file")
    parser.add_argument("--cutoff", type=float, default=15.0,
                        help="Cutoff distance for LDDT calculation (default: 15.0)")
    parser.add_argument("--output", help="Path to output file (optional)")
    args = parser.parse_args()

    try:
        alignment_quality, chain1_id, chain2_id = evaluate_alignment_quality(args.pdb_file, args.fasta_file,
                                                                             args.cutoff)

        output_lines = [
            f"Identified chains: Chain 1 = {chain1_id}, Chain 2 = {chain2_id}\n",
            "Residue-by-residue LDDT scores:\n"
        ]

        for result in alignment_quality:
            output_lines.append(f"Chain1 ({chain1_id}) Residue: {result['residue_index_chain1']}, "
                                f"Chain2 ({chain2_id}) Residue: {result['residue_index_chain2']}, "
                                f"LDDT Score: {result['lddt_score']:.4f}\n")

        overall_lddt = np.mean([result['lddt_score'] for result in alignment_quality])
        output_lines.append(f"\nOverall Alignment LDDT Score: {overall_lddt:.4f}\n")

        if args.output:
            with open(args.output, 'w') as f:
                f.writelines(output_lines)
            print(f"Results written to {args.output}")
        else:
            print("".join(output_lines))

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()