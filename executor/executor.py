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
                                    'solution': self.data.iloc[i][self.cfg.solution_prompt],
                                    'name' : self.data.iloc[i][self.cfg.feature_name]}
                                   , ignore_index=True)

        # Filter out any responses that this student has written themselves
        # Why: We do not want to recommend solutions back to a student that they have written (that's useless)
        new_df = new_df[new_df['name'] != name]

        return new_df.nlargest(k, ['cos_score'])

    def get_results(self, challenge, name):
        res = self.get_top_k(challenge, name)
        return res
