from model.embedding import Embedding
from model.llm import LLM
import pandas as pd
from datetime import datetime

class Executor:
    def __init__(self, cfg, data):
        self.cfg = cfg
        self.data = data
        self.sentences_past = data[cfg['train']['challenge_prompt']].values
        self.embedding = Embedding(sentences_past=self.sentences_past, cfg=self.cfg)

    def similarity(self, challenge):
        cosine_scores = self.embedding.get_cosine_scores(challenge=challenge)
        return cosine_scores

    def get_top_k(self, challenge, name, k=5):
        cosine_scores = self.similarity(challenge)
        new_df = pd.DataFrame()

        for i in range(len(self.sentences_past)):
            temp = {"cos_score": [cosine_scores[i][0]],
                        "challenge": [self.sentences_past[i]],
                        "solution": [self.data.iloc[i][self.cfg['train']['solution_prompt']]],
                        "name" : [self.data.iloc[i][self.cfg['train']['feature_name']]]}

            new_df = pd.concat([new_df, pd.DataFrame.from_dict(temp)], ignore_index=True)
            new_df.reset_index()

        # Filter out any responses that this student has written themselves
        # Why: We do not want to recommend solutions back to a student that they have written (that's useless)
        new_df = new_df[new_df['name'] != name]

        return new_df.nlargest(k, ['cos_score'])

    def get_results(self, input_file_name, k=5):
        # Read in current students' data
        inputs = pd.read_csv(self.cfg['data']['path']+f"/inputs/{input_file_name}.csv", encoding="cp1252", engine='python')

        results = pd.DataFrame()
        #results = pd.DataFrame(columns=['email', 'name', 'reflection', '0_cos_score', '0_challenge', '0_solution', '0_name', '1_cos_score', '1_challenge', '1_solution', '1_name', '2_cos_score', '2_challenge', '2_solution', '2_name', '3_cos_score', '3_challenge', '3_solution', '3_name', '4_cos_score', '4_challenge', '4_solution', '4_name'])
        #results=[]

        # -------------------------------------------------------------
        # Get top-k similar students (challenges and solutions)
        # -------------------------------------------------------------
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
        results.to_csv(self.cfg['data']['path']+f"/results/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{input_file_name}.csv", index=False)
        
        # -------------------------------------------------------------
        # LLM Block - Use ChatGPT API with prompting to provide additional suggestions
        # and put together email communication to student.
        # -------------------------------------------------------------
        # If LLM is true, move on to the next block for generating text using chatgpt api based on student challenges and solutions.
        # Could possibly just add this into the iteration above but for now will just keep it thos way. Need to think about what is more efficient/preferable.
        if self.cfg['model']['LLM'] == True:
            # Initiate LLM object with the parameters needed
            self.llm = LLM(cfg = self.cfg, model=self.cfg['model']['version'])

            # Reindex the dataframe before proceeding
            results.reset_index(drop=True, inplace=True)

            prompting_params = self.cfg['prompting']
            
            # Iterate through each student in our results
            for index, student in results.iterrows():
                print(f"Current student: {student}, current row: {index}")
                current_challenge = student['reflection']
                
                # Initialize an empty list to hold the dictionaries
                student_challenges = [] # note this is for previous students. top-k similar students
                
                # Iterate through previous K student challenges and solutions to build dictionary
                for i in range(k):
                
                    # Assuming there's only one row in your DataFrame as per the structure you provided
                    # For multiple rows, you'd need to adjust this to iterate over rows as well
                    student_info = {
                        "Challenge": student[f"challenge_{i}"],
                        "Solution": student[f"solution_{i}"]
                    }
                    
                    student_challenges.append(student_info)
                
                # Formatting the list into the required string format based on the challenge-solution dictionary
                similar_challenges = "\n\n".join([
                    f"Student {index + 1}:\nChallenge: {entry['Challenge']}\nSolution: {entry['Solution']}"
                    for index, entry in enumerate(student_challenges)
                ])
                
                # Adding triple quotes to mimic the exact output format as asked
                similar_challenges = f"'''\n{similar_challenges}\n'''"

                # Building the prompt based on the challenge-solution pairs.
                prompt = f''' Your role is: {prompting_params['role']}
                
                Your context is: {prompting_params['context']}

                Your task is: {prompting_params['task']}

                An example response I'd like to see is: {prompting_params['example']}

                Here is a list of students' previous challenges and solutions:
                {similar_challenges}
                '''

                # Generate LLM output
                llm_output = self.llm.generate_prompt(prompt, current_challenge)

                print(f"INDEX IS: {index}")
                print(f"LLM OUTPUT IS: {llm_output}")
                # Update the DataFrame with the LLM output for the current row
                results.at[index, 'llm_output'] = llm_output

            # Save results again
            results.to_csv(self.cfg['data']['path']+f"/results/{self.cfg['model']['version']}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{input_file_name}.csv", index=False)
            
        
