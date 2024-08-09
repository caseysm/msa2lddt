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
                if all(atom in residue for atom in ['N', 'CA', 'C']):
                    try:
                        one_letter = aa_mapping.get(residue.resname, 'X')
                        sequence += one_letter
                        coords[len(sequence)] = np.array([residue['N'].get_coord(),
                                                          residue['CA'].get_coord(),
                                                          residue['C'].get_coord()])
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


def calculate_residue_rmsd(coords1, coords2):
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


def evaluate_alignment_quality(pdb_file, fasta_file):
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

    alignment_quality = []

    for a in alignment:
        if a[0] in coords1 and a[1] in coords2:
            rmsd = calculate_residue_rmsd(coords1[a[0]], coords2[a[1]])
            alignment_quality.append({
                'residue_index_chain1': a[0],
                'residue_index_chain2': a[1],
                'rmsd': rmsd
            })

    if len(alignment_quality) == 0:
        raise ValueError("No matching coordinates found between the PDB and FASTA alignment")

    return alignment_quality, chain1_id, chain2_id


def main():
    parser = argparse.ArgumentParser(description="Evaluate alignment quality using backbone RMSD")
    parser.add_argument("pdb_file", help="Path to the PDB file")
    parser.add_argument("fasta_file", help="Path to the FASTA alignment file")
    parser.add_argument("--output", help="Path to output file (optional)")
    args = parser.parse_args()

    try:
        alignment_quality, chain1_id, chain2_id = evaluate_alignment_quality(args.pdb_file, args.fasta_file)

        output_lines = [
            f"Identified chains: Chain 1 = {chain1_id}, Chain 2 = {chain2_id}\n",
            "Residue-by-residue backbone RMSD:\n"
        ]

        for result in alignment_quality:
            output_lines.append(f"Chain1 ({chain1_id}) Residue: {result['residue_index_chain1']}, "
                                f"Chain2 ({chain2_id}) Residue: {result['residue_index_chain2']}, "
                                f"RMSD: {result['rmsd']:.4f} Å\n")

        overall_rmsd = np.sqrt(np.mean([result['rmsd'] ** 2 for result in alignment_quality]))
        output_lines.append(f"\nOverall Alignment RMSD: {overall_rmsd:.4f} Å\n")

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