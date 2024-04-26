from sentence_transformers import SentenceTransformer, util
import os
import torch

class Embedding:
    def __init__(self, sentences_past, cfg):
        self.sentences_past = sentences_past
        self.model = SentenceTransformer('stsb-roberta-large')
        print("finished loading model")

        self.cfg = cfg

        # Specify and make folder for storing embeddings if doesn't already exist
        self.embeddings_folder = f"{self.cfg['data']['path']}/data/embeddings"
        os.makedirs(self.embeddings_folder, exist_ok=True)
        print(f"Embeddings folder is: {self.embeddings_folder}")
        # Specify file name for saved embeddings
        filename = f"{self.cfg['data']['course_input']}.pt"
        print(f"File name is now: {filename}")
        # Full path for the embeddings file
        self.embeddings_file = os.path.join(self.embeddings_folder, filename)

        # TODO. Differentiate for when config.data.filter_class is True or False
        # Load embeddings if they exist, otherwise encode and save
        if os.path.exists(self.embeddings_file):
            print("Loading saved embeddings")
            self.embeded_past = torch.load(self.embeddings_file)
        else:
            print("Encoding sentences and saving embeddings")
            self.embeded_past = self.model.encode(self.sentences_past, convert_to_tensor=True)
            torch.save(self.embeded_past, self.embeddings_file)
            
        print("Finished setup")

    def do_embedding(self, input_text):
        return self.model.encode(str(input_text), convert_to_tensor=True)

    def get_cosine_scores(self, challenge):
        embedded_challenge = self.do_embedding(challenge)
        return util.pytorch_cos_sim(self.embeded_past , embedded_challenge).numpy()

