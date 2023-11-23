import time
import torch
from transformers import pipeline
from transformers.generation.streamers import TextStreamer

from ctransformers import AutoModelForCausalLM, AutoTokenizer

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
print('loading model...')
llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-OpenOrca-GGUF", model_file="mistral-7b-openorca.Q3_K_M.gguf", model_type="mistral", gpu_layers=0, hf=True)
tokenizer = AutoTokenizer.from_pretrained(llm)

print('loading model...')

                    

print('loading model done...')

instruction = """<|im_start|>system
You are a helpful assistant, please answer questions based on the input text.
<|im_end|>

<|im_start|>user
Input:
And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation's top legal minds, who will continue Justice Breyer's legacy of excellence. 

A former top litigator in private practice. A former federal public defender. And from a family of public school educators and police officers. A consensus builder. Since she's been nominated, she's received a broad range of support—from the Fraternal Order of Police to former judges appointed by Democrats and Republicans. 

And if we are to advance liberty and justice, we need to secure the Border and fix the immigration system. 

We can do both. At our border, we've installed new technology like cutting-edge scanners to better detect drug smuggling.  

We've set up joint patrols with Mexico and Guatemala to catch more human traffickers.  

We're putting in place dedicated immigration judges so families fleeing persecution and violence can have their cases heard faster. 

Question: What is the president's plan
<|im_end|>

<|im_start|>assistant
"""

def main():

    while True:
        print("")

        user_prompt = input("You: ")
        start = time.time()
        #streamer = TextStreamer(tokenizer, skip_prompt=True)
        generate_text = pipeline("text-generation",
                        model=llm, 
                        tokenizer = tokenizer,
                        max_new_tokens=200,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True)#,
                        #streamer = streamer)   
        res = generate_text(instruction, do_sample=False, num_beams=1, max_new_tokens=200)
        print(res[0]["generated_text"])

        end = time.time()

        print("Time used: %.3f" % (end-start))
        
        print("")
    

if __name__ == "__main__":
    main()