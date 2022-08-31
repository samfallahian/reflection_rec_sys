from sentence_transformers import SentenceTransformer, util


class Embedding:
    def __init__(self, sentences_past):
        self.sentences_past = sentences_past
        self.model = SentenceTransformer('stsb-roberta-large')

    def do_embedding(self, input_text):
        return self.model.encode(input_text, convert_to_tensor=True)

    def get_cosine_scores(self, challenge):
        return util.pytorch_cos_sim(self.do_embedding(self.sentences_past), self.do_embedding(challenge)).numpy()

