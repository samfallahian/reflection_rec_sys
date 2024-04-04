from model.embedding import Embedding
from model.llm import LLM
import pandas as pd
from datetime import datetime

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
            temp = {"cos_score": [cosine_scores[i][0]],
                        "challenge": [self.sentences_past[i]],
                        "solution": [self.data.iloc[i][self.cfg.train.solution_prompt]],
                        "name" : [self.data.iloc[i][self.cfg.train.feature_name]]}

            new_df = pd.concat([new_df, pd.DataFrame.from_dict(temp)], ignore_index=True)
            new_df.reset_index()

        # Filter out any responses that this student has written themselves
        # Why: We do not want to recommend solutions back to a student that they have written (that's useless)
        new_df = new_df[new_df['name'] != name]

        return new_df.nlargest(k, ['cos_score'])

    def get_results(self, input_file_name, k=5):
        # Read in current students' data
        inputs = pd.read_csv(self.cfg.data.path+f"/inputs/{input_file_name}.csv", encoding="cp1252", engine='python')

        results = pd.DataFrame()
        #results = pd.DataFrame(columns=['email', 'name', 'reflection', '0_cos_score', '0_challenge', '0_solution', '0_name', '1_cos_score', '1_challenge', '1_solution', '1_name', '2_cos_score', '2_challenge', '2_solution', '2_name', '3_cos_score', '3_challenge', '3_solution', '3_name', '4_cos_score', '4_challenge', '4_solution', '4_name'])
        #results=[]

        for index, row in inputs.iterrows():
            print(f"NEXT STUDENT: {row['Full Name']}")

            student_info = pd.DataFrame.from_dict({'email': [row['Email Address']], 'name': row["Full Name"], 'reflection' : row["student's reflection"]})

            # Get top K similar student results
            top_k = self.get_top_k(challenge=row["student's reflection"], name=row["Full Name"], k=k).reset_index()

            temp = top_k.stack().to_frame().T
            #temp.columns = ['{}_{}'.format(*c) for c in temp.columns]
            temp.columns = [f"{col[1]}_{col[0]}" for col in temp.columns]

            results = pd.concat([results, pd.concat([student_info, temp], axis=1)])

        #df = pd.concat(results, ignore_index=True)
        results.to_csv(self.cfg.data.path+f"/results/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{input_file_name}.csv", index=False)
        
        if self.cfg['model']['LLM'] == True:
            # Initiate LLM object with the parameters needed
            self.llm = LLM(model=self.cfg['model']['version'])

            print("WE MADE IT HERE..")
            # TODO. We have progress in a notebook file. next step would be to add it here

        
