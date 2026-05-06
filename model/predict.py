from model import CharLSTM

VOCAB_SIZE = 
EMB_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 3

def predict(inp_sentence):
    m = CharLSTM()
    CharLSTM.forward(inp_sentence)