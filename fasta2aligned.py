from Bio import PDB
from Bio import AlignIO
from Bio.PDB.Superimposer import Superimposer
import argparse
import numpy as np

def parse_fasta_alignment(fasta_file):
    alignment = AlignIO.read(fasta_file, "fasta")
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

def align_pdbs_with_biopython(pdb1, pdb2, fasta_file, output_pdb):
    # Parse PDB files
    parser = PDB.PDBParser(QUIET=True)
    structure1 = parser.get_structure("protein1", pdb1)
    structure2 = parser.get_structure("protein2", pdb2)

    # Parse FASTA alignment
    alignment = parse_fasta_alignment(fasta_file)

    # Extract aligned atoms
    atoms1 = []
    atoms2 = []
    for res1, res2 in alignment:
        try:
            atom1 = structure1[0]["A"][res1]["CA"]
            atom2 = structure2[0]["A"][res2]["CA"]
            atoms1.append(atom1)
            atoms2.append(atom2)
        except KeyError:
            # Skip if residue or atom is missing
            continue

    # Perform superposition
    super_imposer = Superimposer()
    super_imposer.set_atoms(atoms1, atoms2)
    super_imposer.apply(structure2.get_atoms())

    # Calculate RMSD
    rmsd = super_imposer.rms

    # Save aligned structures
    io = PDB.PDBIO()
    io.set_structure(structure1)
    io.save(f"{output_pdb}_aligned1.pdb")
    io.set_structure(structure2)
    io.save(f"{output_pdb}_aligned2.pdb")

    return rmsd

def main():
    parser = argparse.ArgumentParser(description="Align two PDB structures based on a FASTA alignment using BioPython")
    parser.add_argument("pdb1", help="Path to the first PDB file")
    parser.add_argument("pdb2", help="Path to the second PDB file")
    parser.add_argument("fasta_file", help="Path to the FASTA alignment file")
    parser.add_argument("output_pdb", help="Prefix for output PDB files")
    args = parser.parse_args()

    rmsd = align_pdbs_with_biopython(args.pdb1, args.pdb2, args.fasta_file, args.output_pdb)
    print(f"Alignment completed. RMSD: {rmsd:.2f} Å")
    print(f"Aligned structures saved to {args.output_pdb}_aligned1.pdb and {args.output_pdb}_aligned2.pdb")

if __name__ == "__main__":
    main()