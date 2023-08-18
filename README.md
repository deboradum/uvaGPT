# uvaGPT
Query the different UVA Computer Science courses using Faiss. This tool allows you to chat with different CS textbooks using Langchain and the OpenAI api. A chat history memory is kept, so asking follow up questions is supported.
When the answer to the asked question is not to be found in the document, the model responds with "I don't know".

# Usage
To use uvaGPT, first download the requirements by running:
```
python3 -m pip install -r requirements.txt
```
Next, you need an [OpenAI api key](https://platform.openai.com/overview). Add this key to your .env file, and you can start the client using:
```
python3 uvaGPT.py -m <MODELNAME>
```
or
```
python3 uvaGPT.py --model <MODELNAME>
```
A list of available model names can be found by running
```
python3 uvaGPT.py -h
```
Other optional parameters are:
```
-h, --help            show this help message and exit
-m --model            The model you want to query.
--local               Run a local embedder and chat model in stead of the OpenAI api.
--save                Save chat history, including the source of the answer, to a local json file.
```
If you want to add a custom model, simply place the pdf- or txt file in its respective directory and run the program. It will automatically create a Faiss database that is allows lookup of questions.


# Local inference
Besides using OpenAI's Davinci-003 model, local inference is also supported. For local inference to work, you'll need [llama.cpp](https://github.com/ggerganov/llama.cpp) installed and you will need valid GGML model weights placed in the `models/` directory. These models can be downloaded [here (7B)](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main), [here (13B)](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/tree/main) or [here (70B)](https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGML/tree/main). You can also [request access](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and convert the weights to GGML yourself. Keep in mind that the llama chat model weights are needed, not the regular weights.

# Similar projects
I made some other projects based on this project. These other projects do mostly the same thing, but the documents are static and focussed on a specific niche.
- [pdfGPT](https://github.com/deboradum/pdfGPT), chat with any pdf to your choosing.
- [pastorGPT](https://github.com/deboradum/PastorGPT), chat with the Bible.
- [ImamGPT](https://github.com/deboradum/ImamGPT), chat with the Quran.

# Demo
.
