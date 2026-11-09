import json
import torch

def crop_ctc_probs(
    ctc_probs: torch.Tensor,
    N: int = 50,
    K: int = 5,
    blank_id: int = 0,
    mode: str = "entropy",
):
    """
    Args:
        ctc_probs: Tensor (T, V), softmax 后的概率
        N: 保留的时间步数
        K: 每个时间步保留的 token 数
        blank_id: CTC blank token id
        mode: 时间裁剪策略 ["entropy", "margin", "maxprob"]

    Returns:
        cropped: Tensor (N, K)
        token_ids: Tensor (N, K)
        time_ids: Tensor (N,)
    """

    T, V = ctc_probs.shape

    # ===== 1. 计算每一帧的重要性 =====
    if mode == "entropy":
        # 信息熵：越大越不确定，越值得给 LLM
        eps = 1e-8
        score = -(ctc_probs * (ctc_probs + eps).log()).sum(dim=-1)

    elif mode == "margin":
        # top1 - top2，越小越犹豫
        top2 = torch.topk(ctc_probs, k=2, dim=-1).values
        score = -(top2[:, 0] - top2[:, 1])

    elif mode == "maxprob":
        # 低置信度帧更重要
        score = -ctc_probs.max(dim=-1).values

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ===== 2. 选 N 个时间步 =====
    N = min(N, T)
    time_ids = torch.topk(score, k=N).indices.sort().values
    selected = ctc_probs[time_ids]  # (N, V)

    # ===== 3. 每个时间步取 top-K token =====
    topk = torch.topk(selected, k=K, dim=-1)
    probs = topk.values        # (N, K)
    token_ids = topk.indices   # (N, K)

    mask = token_ids[:, 0] != 0  # shape: (T,)
    probs_filt = probs[mask]
    tokens_filt = token_ids[mask]
    times_filt = time_ids[mask]

    return probs_filt, tokens_filt, times_filt




def save_NBbest_to_txt(text_transform, file_path, save_mid_src_paths, ctc_matrix, n_best_list, actuals):
    n_bests = []
    for nbl in n_best_list:  # B, N, L   batch
        nb = []
        for nbb in nbl.beams:  # 20
            # pred = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
            nbb_new = (nbb[0], text_transform.post_process(nbb[1]).replace("<eos>", ""))
            nb.append(nbb_new)
        n_bests.append(nb)

    ctc_matrix = list(ctc_matrix.permute(1, 0, 2))
    ctc_cms = []
    for i in range(len(ctc_matrix)):
        ctc_m = ctc_matrix[i]
        N = ctc_m.shape[0]  # 75  #len(n_bests[i][0][1])
        K = len(n_bests[0])
        probs, tokens_list, times = crop_ctc_probs(ctc_m, N, K, blank_id=-1, mode="entropy")

        tokens=[]
        for j in range(tokens_list.shape[0]):
            ctoken = text_transform.post_process(tokens_list[j]).replace("<eos>", "").replace("<blank>", "*")
            tokens.append(ctoken)
        evidence = [
            {
                "t": int(t),
                "candidates": [
                    {"token": tok.replace("*", "<blank>"), "prob": float(p)}
                    for tok, p in zip(tokens[i], probs[i])
                ]
            }
            for i, t in enumerate(times)
        ]
        ctc_cms.append(evidence)

    for i in range(len(actuals)):
        # file_path, save_mid_src_paths,  'cmlr_video_seg24s/s5/20151019/section_2_082.62_085.66.mp4'
        file_name = save_mid_src_paths[i].replace("cmlr_video_seg24s/", "").replace("/", "_").replace(".mp4", ".json")
        write_path = file_path + "/" + file_name

        data = {
            "actual": actuals[i],
            "n_best": [
                [float(score), text] for score, text in n_bests[i]
            ],
            "ctc_cm": ctc_cms[i]
        }
        with open(write_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)



