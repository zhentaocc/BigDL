from datasets import load_dataset, concatenate_datasets
import datasets
import transformers
import torch
import intel_extension_for_pytorch as ipex
import os
from lm_eval.base import MultipleChoiceTask, rf
from lm_eval.tasks.cmmlu import CmmluSubject
import random
import numpy as np
from lm_eval.models import get_model


def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            if context == "":
                # end of text as context
                context_enc, continuation_enc = [self.eot_token_id], self.tok_encode(
                    continuation
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)
            print(f'continutation {continuation}, continuation_enc {continuation_enc}') #XXX
            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)


def _model_call(self, inps):
    """
    inps: a torch tensor of shape [batch, sequence]
    the size of sequence may vary from call to call

    returns: a torch tensor of shape [batch, sequence, vocab] with the
    logits returned from the model
    """
    with torch.inference_mode():
        inps = inps.to(self.device)
        print(f'Input_ids: {inps}') #XXX
        res = self.model(inps)[0]
        print(f'Before Softmax:\n\tA: {res[:, -1:, 28741]};\n\tB: {res[:, -1:, 28760]};\n\tC: {res[:, -1:, 28743]};\n\tD: {res[:, -1:, 28757]}') #XXX
        import torch.nn.functional as F #XXX
        res_softmax = F.log_softmax(res, dim=-1) #XXX
        print(f'After Softmax:\n\tA: {res_softmax[:, -1:, 28741]};\n\tB: {res_softmax[:, -1:, 28760]};\n\tC: {res_softmax[:, -1:, 28743]};\n\tD: {res_softmax[:, -1:, 28757]}') #XXX
        return res


# construct_requests refer to: https://github.com/EleutherAI/lm-evaluation-harness/blob/008fc2a23245c40384f2312718433eeb1e0f87a9/lm_eval/tasks/cmmlu.py
def construct_requests(doc, ctx):
    lls = [
        rf.loglikelihood(ctx, "{}".format(choice))[0] for choice in doc['choices']
    ]
    return lls

# process_results refer to: https://github.com/EleutherAI/lm-evaluation-harness/blob/6843c5d7a0f9cd19a450713338b7662af17f40ce/lm_eval/base.py#L757
def process_results( doc, results, doc_id):
    gold = doc["gold"]
    acc = 1.0 if np.argmax(results) == gold else 0.0
    completion_len = np.array([float(len(i)) for i in doc["choices"]])
    # print(f'[doc_id: {doc_id}] predict_result:', results / completion_len)
    acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0
    
    return {
        "doc_id": doc_id,
        "gold": gold,
        "selected": int(np.argmax(results)),
        "logits": (results / completion_len).tolist(),
        "acc": acc,
        "acc_norm": acc_norm,
    }

# setting args
model = 'bigdl-llm'
model_args = "pretrained=/mnt/disk1/models/Mistral-7B-v0.1,load_in_low_bit=sym_int8"
batch_size = 1
max_batch_size = None
device = 'xpu:0'
task_name = 'mmlu' # 
#doc_id = 0 # randomly set for we only test few data

# load the model
from bigdl_llm import BigDLLM

## replace original model function to dump detail info
BigDLLM.loglikelihood = loglikelihood
BigDLLM._model_call = _model_call

from lm_eval import models
models.MODEL_REGISTRY['bigdl-llm'] = BigDLLM    # patch bigdl-llm to harness
lm = get_model(model).create_from_arg_string(
    model_args,
    {
        "batch_size": batch_size,
        "max_batch_size": max_batch_size,
        "device": device,
    },
)

# Note that train data needs to be pre-process to 'doc' and 'ctx' format. The transition has too many dependencies and is quite complicate, so we just dump the results after pre-process.
import json
with open("docs.json",'r') as f:
    docs = json.load(f)

with open("ctxes.json",'r') as f:
    ctxes = json.load(f)

outputs = []
for doc_id in range(10, 11): #XXX
    doc= docs[doc_id]
    ctx = ctxes[doc_id]

    import collections
    requests = collections.defaultdict(list)
    requests_origin = collections.defaultdict(list)
    process_res_queue = collections.defaultdict(list)

    reqs = construct_requests(doc=doc, ctx=ctx)

    # refer to evaluator.py
    for i,req in enumerate(reqs):
        requests[req.request_type].append(req)
        requests_origin[req.request_type].append((i, task_name, doc, doc_id))

    for reqtype, reqs in requests.items():
        print("Running", reqtype, "requests")
        resps = getattr(lm, reqtype)([req.args for req in reqs]) # this line produce predict results for each choice
        resps = [
            x if req.index is None else x[req.index] for x, req in zip(resps, reqs)
        ]
        for resp, (i, task_name, doc, doc_id) in zip(resps, requests_origin[reqtype]):
            process_res_queue[(task_name, doc_id)].append((i, resp))

    for (task_name, doc_id), requests in process_res_queue.items():
            requests.sort(key=lambda x: x[0])
            requests = [x[1] for x in requests]
            metrics = process_results(doc, requests, doc_id)
            print(metrics)
            outputs.append(metrics)

with open('output.json', 'w') as json_file:
    json.dump(outputs, json_file, indent=2)