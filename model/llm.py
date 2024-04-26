import openai
import json
import time

class LLM:
    def __init__(self, cfg, model="gpt-3.5-turbo", retries = 3, temperature = 0.7, max_tokens = 2000):

        self.cfg = cfg
        # model can be: "gpt-3.5-turbo", "gpt-4-0125-preview"
        self.potential_models = ["gpt-3.5-turbo", "gpt-4-0125-preview"]
        self.model = model

        if self.model == self.potential_models[0]:
            self.request_timeout = 10
        elif self.model == self.potential_models[1]:
            self.request_timeout = 30
        else:
            self.request_timeout = 10

        self.retries = retries
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Load key file
        with open(f"{self.cfg['data']['path']}/configs/{self.cfg['model']['api_key_file']}") as key_file:
            key = json.load(key_file)
        openai.api_key = key['api_key']

    def generate_prompt(self, prompt, data):

        for _ in range(self.retries + 1):  # Include the initial attempt plus the specified retries
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        # Input the prompt
                        {"role": "system", "content": prompt},
                        # Input reflection response
                        {"role": "user", "content": data},
                    ],
                    temperature=self.temperature, # the lower the temperature, the less creative the responses are
                    max_tokens=self.max_tokens, # max length of output
                    request_timeout=self.request_timeout,
                    
                )
                # If the request was successful, break out of the loop
                break
            except openai.error.OpenAIError as e:
                # Handle the error (e.g., print the error message)
                print("Error:", e)
            except TimeoutError:
                # Handle a timeout (e.g., print a message or take other actions)
                print("Request timed out")
            
            # Add a delay before retrying (to avoid immediate retries)
            time.sleep(1)

        if response and len(response.choices) > 0:
            prompt_response =  str(response['choices'][0]['message']['content'])
            #print(prompt_response )
            return (prompt_response)