import numpy as np
import json
from tqdm import tqdm
import random
import argparse

def load_glove_model(File):
    print("Loading Glove Model")
    glove_model = {}
    with open(File,'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model

def normalize(x):
    return x / np.linalg.norm(x)

def convert2matrix(glove_model):
    ids = list(glove_model.keys())
    arrs = []
    for gid in ids:
        arrs.append(normalize(glove_model[gid]))
    return ids, np.stack(arrs)

def top_k_indices(arr, k):
    """Returns the indices of the k largest elements in an array.(Note that the sequence of element is not strictly constrained)"""
    indices = np.argpartition(arr, -k)[-k:]  # Get indices of k largest
    return indices

def get_pivot_matrix(pivot_ids, glove_model):
    arrs = []
    for pid in pivot_ids:
        arrs.append(normalize(glove_model[pid]))
    return np.stack(arrs)

def convert2rel_rep(
    glove_vec,
    pivot_ids,
):
    ids, rep = convert2matrix(glove_vec)
    p_vec = get_pivot_matrix(
        pivot_ids = pivot_ids,
        glove_model = glove_vec
    )
    return np.matmul(rep, p_vec.T)

def convert2relative_rep(
    embed1,
    embed2,
    gold, # target token ID (key) --> source token ID (value)
    num_pivot=300,
    seed=0,
):
    tgt_keys = list(embed1.keys())
    src_keys = list(embed2.keys())

    random.seed(seed)
    
    ids = [k for k in gold.keys()]

    random.shuffle(ids)

    curr_i = 0
    tgt_ids = []
    for ci in ids:
        if (ci in tgt_keys) and (str(gold[ci]) in src_keys):
            tgt_ids.append(ci)
            curr_i += 1
        
        if curr_i >= num_pivot:
            break

    src_ids = [str(gold[i]) for i in tgt_ids]

    tgt_rep = convert2rel_rep(
        glove_vec = embed1,
        pivot_ids = tgt_ids
    )

    src_rep = convert2rel_rep(
        glove_vec = embed2,
        pivot_ids = src_ids
    )
    
    return tgt_keys, tgt_rep, src_keys, src_rep

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source-glove-vector-path", type=str, default="./data/glove_vec.pythia.txt")
    parser.add_argument("-s1", "--source-vocab-size", type=int, default=50304)
    parser.add_argument("-t", "--target-glove-vector-path", type=str, default="./data/glove_vec.gemma.txt")
    parser.add_argument("-s2", "--target-vocab-size", type=int, default=256000)
    parser.add_argument("-g", "--gold-target-to-source-path", type=str, default="./data/Vocab_count/gemma2pythia.json")
    parser.add_argument("-r", "--relative-representation", action='store_true')
    parser.add_argument("-v", "--vanilla-representation", action='store_true')
    parser.add_argument("-n", "--pivotal-token-number", type=int, default=300)
    parser.add_argument("-o", "--output-path", type=str, default="./data/pythia2gemma/glove.json")

    args = parser.parse_args()

    # new tokenizer glove path
    g_p1 = args.target_glove_vector_path
    g_vocab_len1 = args.target_vocab_size

    # old tokenizer glove path
    g_p2 = args.source_glove_vector_path
    g_vocab_len2 = args.source_vocab_size

    tgt_path = args.output_path

    # gold src to tgt id transition json dict
    t2l_supl_path=args.gold_target_to_source_path

    with open(t2l_supl_path, "r") as f:
        t2l_supl = json.load(f)

    embed1 = load_glove_model(g_p1)
    embed2 = load_glove_model(g_p2)

    if args.relative_representation:
        ids1, rep1, ids2, rep2 = convert2relative_rep(
            embed1=embed1,
            embed2=embed2,
            gold=t2l_supl,
            num_pivot=args.pivotal_token_number
        )
    elif args.vanilla_representation:
        # Calculate the transition matrix
        ids1, rep1 = convert2matrix(embed1)
        ids2, rep2 = convert2matrix(embed2)
    else:
        raise Exception(f"Only relative and varnilla representation are implemented.")

    sim = np.matmul(rep1, rep2.T)

    td = {}
    tids = [str(tid) for tid in range(g_vocab_len1)]
    supl_id = 0
    for tid in tqdm(tids, desc="Get the max prob target idx"):
        # gold label
        if tid in t2l_supl:
            td[tid] = t2l_supl[tid]
            supl_id += 1
            continue

        # missing token id: random pick
        if tid not in ids1:
            td[tid] = random.randint(0, g_vocab_len2-1)
            supl_id += 1
            continue

        id1_idx = ids1.index(tid)
        lix = np.argmax(sim[id1_idx])
        lid = ids2[lix]

        # assert(lid != 'unk'), tid
        # back to the second top id
        if lid == 'unk' or lid == '<unk>':
            lix = set(top_k_indices(sim[id1_idx], 2)) - set(top_k_indices(sim[id1_idx], 1))
            lid = ids2[lix.pop()]

        td[tid] = int(lid)

    print(f"{supl_id} ids are suppled with gold transition dictionary.")

    with open(tgt_path, "w") as f:
        json.dump(td, f, indent="\t")
