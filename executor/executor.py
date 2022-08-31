from model.embedding import Embedding
import pandas as pd


class Executor:
    def __init__(self, cfg, data):
        self.cfg = cfg
        self.data = data
        self.sentences_past = data[cfg.challenge_prompt].values
        self.embedding = Embedding(sentences_past=self.sentences_past)

    def similarity(self, challenge):
        cosine_scores = self.embedding.get_cosine_scores(challenge=challenge)
        return cosine_scores

    def get_top_k(self, challenge, name, k=5):
        cosine_scores = self.similarity(challenge)
        new_df = pd.DataFrame()
        for i in range(len(self.sentences_past)):
            new_df = new_df.append({'cos_score': cosine_scores[i][0],
                                    'challenge': self.sentences_past[i],
                                    'solution': self.data.iloc[i][self.cfg.solution_prompt]}
                                   , ignore_index=True)

        written_by_self = self.data.index[self.data['name'] == name].tolist()

        if len(written_by_self) > 0:
            new_df.drop(index=written_by_self, inplace=True)

        return new_df.nlargest(k, ['cos_score'])

    def get_results(self, challenge, name):
        res = self.get_top_k(challenge, name)
        return res
