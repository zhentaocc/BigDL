# ChatGLM3

In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on ChatGLM3 models. For illustration purposes, we utilize the [THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b) as a reference ChatGLM3 model.

## 0. Requirements
To run these examples with BigDL-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example 1: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a ChatGLM3 model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm

pip install --pre --upgrade bigdl-llm[all] # install bigdl-llm with 'all' option
```

### 2. Run
```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the ChatGLM3 model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'THUDM/chatglm3-6b'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'AI是什么？'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

> **Note**: When loading the model in 4-bit, BigDL-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the ChatGLM3 model based on the capabilities of your machine.

#### 2.1 Client
On client Windows machine, it is recommended to run directly with full utilization of all cores:
```powershell
python ./generate.py 
```

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set BigDL-LLM env variables
source bigdl-llm-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./generate.py
```

#### 2.3 Sample Output
#### [THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
<|user|>
AI是什么？
<|assistant|>
-------------------- Output --------------------
[gMASK]sop <|user|>
AI是什么？
<|assistant|> AI是人工智能（Artificial Intelligence）的缩写，指的是通过计算机程序和算法模拟人类智能的技术。AI可以帮助我们解决各种问题，例如语音
```

```log
Inference time: xxxx s
-------------------- Prompt --------------------
<|user|>
What is AI?
<|assistant|>
-------------------- Output --------------------
[gMASK]sop <|user|>
What is AI?
<|assistant|>
AI stands for Artificial Intelligence. It refers to the development of computer systems that can perform tasks that would normally require human intelligence, such as recognizing speech or making
```

## Example 2: Stream Chat using `stream_chat()` API
In the example [streamchat.py](./streamchat.py), we show a basic use case for a ChatGLM3 model to stream chat, with BigDL-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm

pip install --pre --upgrade bigdl-llm[all] # install bigdl-llm with 'all' option
```

### 2. Run
**Stream Chat using `stream_chat()` API**:
```
python ./streamchat.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --question QUESTION
```

**Chat using `chat()` API**:
```
python ./streamchat.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --question QUESTION --disable-stream
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the ChatGLM3 model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'THUDM/chatglm3-6b'`.
- `--question QUESTION`: argument defining the question to ask. It is default to be `"晚上睡不着应该怎么办"`.
- `--disable-stream`: argument defining whether to stream chat. If include `--disable-stream` when running the script, the stream chat is disabled and `chat()` API is used.

> **Note**: When loading the model in 4-bit, BigDL-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the ChatGLM3 model based on the capabilities of your machine.

#### 2.1 Client
On client Windows machine, it is recommended to run directly with full utilization of all cores:
```powershell
$env:PYTHONUNBUFFERED=1  # ensure stdout and stderr streams are sent straight to terminal without being first buffered
python ./streamchat.py
```

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set BigDL-LLM env variables
source bigdl-llm-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
export PYTHONUNBUFFERED=1  # ensure stdout and stderr streams are sent straight to terminal without being first buffered
numactl -C 0-47 -m 0 python ./streamchat.py
```
