from model.embedding import Embedding
import pandas as pd


class Executor:
    def __init__(self, cfg, data):
        self.cfg = cfg
        self.data = data
        self.sentences_past = data[cfg.train.challenge_prompt].values
        self.embedding = Embedding(sentences_past=self.sentences_past)

    def similarity(self, challenge):
        cosine_scores = self.embedding.get_cosine_scores(challenge=challenge)
        return cosine_scores

    def get_top_k(self, challenge, name, k=5):
        cosine_scores = self.similarity(challenge)
        new_df = pd.DataFrame()
        for i in range(len(self.sentences_past)):
            new_df = new_df.append({f"cos_score": cosine_scores[i][0],
                                    f"challenge": self.sentences_past[i],
                                    f"solution": self.data.iloc[i][self.cfg.train.solution_prompt],
                                    f"name" : self.data.iloc[i][self.cfg.train.feature_name]}
                                   , ignore_index=True)

        # Filter out any responses that this student has written themselves
        # Why: We do not want to recommend solutions back to a student that they have written (that's useless)
        new_df = new_df[new_df['name'] != name]

        return new_df.nlargest(k, ['cos_score'])

    def get_results(self, input_file_name):
        inputs = pd.read_csv(self.cfg.data.path+f"/inputs/{input_file_name}.csv", encoding="cp1252", engine='python')
        results=[]
        for index, row in inputs.iterrows():
            top_k = self.get_top_k(challenge=row["challenge"], name=row["name"])
            results.append(top_k)

        df = pd.concat(results, ignore_index=True)
        df.to_csv(self.cfg.data.path+f"/results/result_for_{input_file_name}.csv", index=False)
        pass
