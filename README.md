# GPUTSVMult
# Language: CUDA
# Input: TXT
# Output: TSV
# Tested with: PluMA 1.0, CUDA 10

Multiply two matrices on the GPU

Original authors: Adam Tahoun, Fernando Serrano, Dane Parchment, and Gabriel Perez 

The plugin accepts as input a TXT file of tab-delimited keyword-value pairs:
matrix1: First matrix (TSV file)
matrix2: Second matrix (TSV file)
M: Number of rows (first)
N: Number of columns (first), Number of rows (second)
P: Number of columns (second)

The Matrix product will be output as a TSV file
