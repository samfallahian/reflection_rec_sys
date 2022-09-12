from sentence_transformers import SentenceTransformer, util


class Embedding:
    def __init__(self, sentences_past):
        self.sentences_past = sentences_past
        self.model = SentenceTransformer('stsb-roberta-large')
        self.embeded_past = self.model.encode(self.sentences_past, convert_to_tensor=True)

    def do_embedding(self, input_text):
        return self.model.encode(input_text, convert_to_tensor=True)

    def get_cosine_scores(self, challenge):
        embedded_challenge = self.do_embedding(challenge)
        return util.pytorch_cos_sim(self.embeded_past , embedded_challenge).numpy()

