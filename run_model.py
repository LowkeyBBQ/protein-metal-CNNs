import torch
from cnn_models import CARP_CNN
from sequence_models.pretrained import load_model_and_alphabet
import sys

CARP_PATH = 'carp_76M.pt'
STATE_PATH = 'carp_cnn_weights.pt'

def predict(seq):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    carp_model, carp_collater = load_model_and_alphabet(CARP_PATH)
    carp_model.to(device)

    x = carp_collater([[seq]])[0]
    x = x.to(device)
    rep = carp_model(x)
    embedded_seq = rep["representations"][32]
    embedded_seq = torch.transpose(embedded_seq, 1, 2) # (1, num. metals, seq. length)

    model = CARP_CNN(29)
    model.load_state_dict(torch.load(state_path))
    model.eval()
    model.to(device)

    preds = torch.sigmoid(torch.squeeze(model(embedded_seq)))
    class_preds = preds > 0.5

    for pred in torch.nonzero(class_preds):
        print(f"metal {pred[0]} binding at residue {pred[1]}")

 
input_seq = sys.argv[1]
predict(input_seq)
