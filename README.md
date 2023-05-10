# protein-metal-CNNs
Code for UCL MEng Mathematical Computation Final Year Project (COMP0138)
Student Number: 19014550

## Dependencies
This code makes use of PyTorch, Pandas and Scikit-Learn, as well as the "sequence-models" package mentioned specified [here](https://github.com/microsoft/protein-sequence-models).

## Usage
The pretrained CARP embeddings can be downloaded by following the instructions [here](https://github.com/microsoft/protein-sequence-models). To run the model provided in the repo, which is the best-performing model from the project report, the pretrained CARP embeddings should be stored in the root directory with filename "carp_76M.pt" (as required by the run_model.py script in this repo).

Binding site prediction can be performed with the run_model.py script for a specified protein sequence. The sequence parameter is a standard string of characters representing an amino acid sequence.
```
python run_model.py <sequence>
```
This will print the residue location(s) for which there is a metal binding prediction, and the metal/ligand which binds to it. Predictions for all labels are made under a threshold of 0.5.




