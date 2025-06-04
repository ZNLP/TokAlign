import json
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import argparse

def read_tsv(file_path):
    res = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            res.append([d.split(" ") for d in line.strip().split("\t")])
    return res

# BLEU-1
def eval_trans_matrix(
    trans_dict_path="./log/pythia2gemma/glove-MX1M-iter15-d300.json", 
    eval_file_path="./data/pretrain-dataset/pythia-2-gemma-MX1K-eval",
    bleu_weights=(1, 0, 0, 0),
):
    with open(trans_dict_path, "r") as f:
        trans = json.load(f)

    eval_data = read_tsv(eval_file_path)

    tgt_len = len(list(trans.keys()))

    td = trans

    tgt_len = len(list(trans.keys()))

    total_b = 0
    # for s in tqdm(eval_data):
    for s in eval_data:
        # src: source token id, e.g., pythia ids, tgt: target token id, e.g., gemma ids
        src, tgt = s[0], s[1]

        # using td dict by maping target ids to source ids
        pred = [str(td[tid]) for tid in tgt]
        total_b += sentence_bleu([src], pred, bleu_weights)

        # using td dict by maping source ids to target ids
        # pred = [str(td[sid]) for sid in src]
        # total_b += sentence_bleu([tgt], pred, bleu_weights)

    print(f"Average bleu: {total_b/len(eval_data)}")

    return total_b/len(eval_data)

# BERT-Score
def eval_bert_score(
    trans_dict_path="./log/pythia2gemma/glove-MX1M-iter15-d300.json",
    eval_file_path="./data/pretrain-dataset/pythia-2-gemma-MX1K-eval",
    tok_path="./data/pythia-1b",
    model_path="all-mpnet-base-v2",
):
    tok = AutoTokenizer.from_pretrained(tok_path)
    model = SentenceTransformer(model_path)

    with open(trans_dict_path, "r") as f:
        trans = json.load(f)

    eval_data = read_tsv(eval_file_path)

    tgt_len = len(list(trans.keys()))

    td = trans

    tgt_len = len(list(trans.keys()))

    total_b = 0
    
    all_src = []
    all_tgt = []
    # for s in tqdm(eval_data):
    for s in eval_data:
        src, tgt = s[0], s[1]

        # td map source token id to target token id
        # pred = [int(td[sid]) for sid in src]
        # tgt = [int(tid) for tid in tgt]

        # td map target token id to source token id
        pred = [int(td[tid]) for tid in tgt]
        tgt = [int(sid) for sid in src]

        res = tok.batch_decode([pred, tgt])
        all_src.append(res[0])
        all_tgt.append(res[1])
    
    embed0 = model.encode(all_src)
    embed1 = model.encode(all_tgt)

    sims = model.similarity(embed0, embed1)
    sim_d = []
    for i in range(len(eval_data)):
        sim_d.append(sims[i][i].item())

    return sum(sim_d)/len(sim_d)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--evaluate-method", type=str, default="bleu")
    parser.add_argument("-m", "--one2one-matrix-path", type=str, default="./data/pythia2gemma/glove.json")
    parser.add_argument("-f", "--eval-file-path", type=str, default="./data/pretrain-dataset/pythia-2-gemma-MX1K-eval")
    parser.add_argument("-t", "--tokenizer-path", type=str, default="EleutherAI/pythia-1b")
    parser.add_argument("-b", "--bert-score-model-path", type=str, default="all-mpnet-base-v2")
    parser.add_argument("-w", "--bleu-weights", type=str, default="1,0,0,0")

    args = parser.parse_args()

    if args.evaluate_method.lower() == "bleu":
        weights = tuple([float(i) for i in args.bleu_weights.split(",")])
        assert len(weights) == 4, "There are only 4 BLEU weights (BLEU-1 to 4)"
        eval_trans_matrix(
            trans_dict_path = args.one2one_matrix_path,
            eval_file_path = args.eval_file_path,
            bleu_weights = weights
        )
    elif args.evaluate_method.lower() == "bert-score" or args.evaluate_method.lower() == "bertscore":
        eval_bert_score(
            trans_dict_path = args.one2one_matrix_path,
            eval_file_path = args.eval_file_path,
            tok_path = args.tokenizer_path,
            model_path = args.bert_score_model_path,
        )
    else:
        raise Exception(f"{args.evaluate_method} is not implemented.")
