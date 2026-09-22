import ast
import os
import csv
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn import metrics
from sklearn.metrics import roc_curve
from collections import defaultdict
from pypinyin import pinyin, lazy_pinyin, Style


def compute_WordorChar_level_distance(seq1, seq2, datatype):
    if datatype == 'en':
        seq1 = seq1.replace("<space>", " ")
        seq2 = seq2.replace("<space>", " ")
        return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())
    elif datatype == 'zh':
        seq1 = seq1.replace(" ", "")
        seq2 = seq2.replace(" ", "")
        return torchaudio.functional.edit_distance(list(seq1.replace('<unk>', '*')), list(seq2.replace('<unk>', '*')))


def compute_cer_ser_255075(total_samples, CERs, dises, actuals,
                           CER_25=True, CER_50=True, CER_75=True, CER_S=True, CER_M=True, CER_L=True,
                           SER_25=True, SER_50=True, SER_75=True, SER_100=True, SER_S=True, SER_M=True, SER_L=True):
    dis_ = [0, 0, 0, 0, 0, 0]  # dis_25, dis_50, dis_75, dis_S, dis_M, dis_L
    len_ = [0, 0, 0, 0, 0, 0]  # len_25, len_50, len_75, len_S, len_M, len_L

    count_ = [0, 0, 0, 0, 0, 0, 0]  # count_25, count_50, count_75, count_100, count_S, count_M, count_L
    for i in range(total_samples):
        ## CER
        if CER_25 and CERs[i] < 0.25:
            dis_[0] = dis_[0] + dises[i]
            len_[0] = len_[0] + len(actuals[i])
        if CER_50 and CERs[i] < 0.5:
            dis_[1] = dis_[1] + dises[i]
            len_[1] = len_[1] + len(actuals[i])
        if CER_75 and CERs[i] < 0.75:
            dis_[2] = dis_[2] + dises[i]
            len_[2] = len_[2] + len(actuals[i])

        if CER_S and len(actuals[i]) <= 10:
            dis_[3] = dis_[3] + dises[i]
            len_[3] = len_[3] + len(actuals[i])
        if CER_M and 10<len(actuals[i]) <= 20:
            dis_[4] = dis_[4] + dises[i]
            len_[4] = len_[4] + len(actuals[i])
        if CER_L and len(actuals[i]) >20:
            dis_[5] = dis_[5] + dises[i]
            len_[5] = len_[5] + len(actuals[i])

        ## SER
        if SER_25 and CERs[i] < 0.25:
            count_[0] = count_[0] + 1
        if SER_50 and CERs[i] < 0.5:
            count_[1] = count_[1] + 1
        if SER_75 and CERs[i] < 0.75:
            count_[2] = count_[2] + 1
        if SER_100 and CERs[i] < 1.00:
            count_[3] = count_[3] + 1

        if SER_S and len(actuals[i]) <= 10:
            count_[4] = count_[4] + 1
        if SER_M and 10<len(actuals[i]) <= 20:
            count_[5] = count_[5] + 1
        if SER_L and len(actuals[i]) >20:
            count_[6] = count_[6] + 1

    CER_25 = dis_[0] / len_[0] if len_[0] != 0 else 0
    CER_50 = dis_[1] / len_[1] if len_[1] != 0 else 0
    CER_75 = dis_[2] / len_[2] if len_[2] != 0 else 0
    CER_S = dis_[3] / len_[3] if len_[3] != 0 else 0
    CER_M = dis_[4] / len_[4] if len_[4] != 0 else 0
    CER_L = dis_[5] / len_[5] if len_[5] != 0 else 0
    CER_SML = [CER_25, CER_50, CER_75, CER_S, CER_M, CER_L]

    SER_25 = 1-count_[0] / total_samples
    SER_50 = 1-count_[1] / total_samples
    SER_75 = 1-count_[2] / total_samples
    SER_100 = 1-count_[3] / total_samples
    SER_S = 1-count_[4] / total_samples
    SER_M = 1-count_[5] / total_samples
    SER_L = 1-count_[6] / total_samples
    SER_SML = [SER_25, SER_50, SER_75, SER_100, SER_S, SER_M, SER_L]

    return CER_SML, SER_SML, count_


def count_cers(cers, interval=0):
    if isinstance(interval, (int, float)):
        count = 0
        for cer in cers:
            if cer == interval:
                count += 1
    elif isinstance(interval, tuple):
        count = 0
        for cer in cers:
            if interval[0]<cer<interval[1]:
                count += 1
    return count


def resAna_cer(actuals, predicts, datatype='zh'):
    cers = []
    dises = []
    dis_total = 0
    len_total = 0
    for i in range(len(actuals)):
        dis_per = compute_WordorChar_level_distance(actuals[i], predicts[i], datatype)
        len_per = len(actuals[i])
        cer_per = dis_per/len_per
        dises.append(dis_per)
        cers.append(cer_per)
        dis_total += dis_per
        len_total += len_per
    cer_total = dis_total/len_total
    return cer_total, cers, dises


def res_cerDivide(actuals, predicts, datatype='zh'):

    actual_1 = []
    predict_1 = []

    actual_2 = []
    predict_2 = []

    actual_3 = []
    predict_3 = []

    actual_4 = []
    predict_4 = []

    actual_5 = []
    predict_5 = []

    for i in range(len(actuals)):
        dis_per = compute_WordorChar_level_distance(actuals[i], predicts[i], datatype)
        len_per = len(actuals[i])
        cer_per = dis_per/len_per
        if cer_per==0:
            actual_1.append(actuals[i])
            predict_1.append(predicts[i])
        elif 0 < cer_per <= 0.3:
            actual_2.append(actuals[i])
            predict_2.append(predicts[i])
        elif 0.3 < cer_per <= 0.6:
            actual_3.append(actuals[i])
            predict_3.append(predicts[i])
        elif 0.6 < cer_per <= 1.00:
            actual_4.append(actuals[i])
            predict_4.append(predicts[i])
        else:
            actual_5.append(actuals[i])
            predict_5.append(predicts[i])

    return [actual_1, actual_2, actual_3, actual_4, actual_5], [predict_1, predict_2, predict_3, predict_4, predict_5]


def resAna_csv_update_index_contray(index_now, dp):
    index_last = index_now  # (0,0)
    i = index_last[0]
    j = index_last[1]

    if i > 0 and j > 0:
        list_3 = [dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j]]  # 左上，左，上
        if list_3.index(min(list_3)) == 0:
            index_now = (i, j - 1)
        elif list_3.index(min(list_3)) == 1:
            index_now = (i - 1, j - 1)
        elif list_3.index(min(list_3)) == 2:
            index_now = (i - 1, j)
        else:
            raise ValueError("eval_matrix error !!!!!")
    elif i > 0 == j:
        index_now = (i - 1, j)
    elif i == 0 < j:  # i=0,j>0
        index_now = (i, j - 1)
    else:
        index_now = index_now

    return index_last, index_now


# def resAna_csv_edit_distance(s1, s2):
#     S = 0
#     D = 0
#     I = 0
#
#     # 创建一个大小为 (len(s1)+1) x (len(s2)+1) 的矩阵
#     dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
#     # 初始化第一行和第一列
#     for i in range(len(s1) + 1):
#         dp[i][0] = i
#     for j in range(len(s2) + 1):
#         dp[0][j] = j
#
#     # 计算编辑距离的动态规划填表
#     for i in range(1, len(s1) + 1):
#         for j in range(1, len(s2) + 1):
#             cost = 0 if s1[i - 1] == s2[j - 1] else 1  # s1[i - 1]与s2[j - 1]替换的成本
#             dp[i][j] = min(dp[i - 1][j] + 1,  # 删除   左下表示删除错误D
#                            dp[i][j - 1] + 1,  # 插入   右上表示插入错误I
#                            dp[i - 1][j - 1] + cost)  # 替换  左上表示替换错误S
#
#     step_now = (len(s1), len(s2))
#     errors = []  # [('例', '您', 0), ('如', '的', 1),
#     # is_run = 1
#     while step_now != (0, 0):
#         # if step_now[0] == 0 and step_now[1] == 0:
#         #     is_run = 0
#         step_last, step_now = resAna_csv_update_index_contray(step_now, dp)
#         if dp[step_now[0]][step_now[1]] != dp[step_last[0]][step_last[1]]:
#             if step_now[0] == step_last[0] - 1 and step_now[1] == step_last[1] - 1:
#                 S = S + 1
#                 error = (s2[step_now[1]], s1[step_now[0]], step_now[1]+1)
#                 errors.append(error)
#             elif step_now[0] == step_last[0] and step_now[1] == step_last[1] - 1:
#                 D = D + 1
#                 error = (s2[step_now[1]], '', step_now[1]+1)
#                 errors.append(error)
#             elif step_now[0] == step_last[0] - 1 and step_now[1] == step_last[1]:
#                 I = I + 1
#                 error = ('', s1[step_now[0]], step_now[1]+1)
#                 errors.append(error)
#             else:
#                 raise ValueError("eval_matrix error !!!!!")
#
#     errors = list(reversed(errors))
#     # 输出结果
#     # print(f"编辑距离: {dp[len(s1)][len(s2)]}")
#     # print(f"替换次数: {S}")
#     # print(f"删除次数: {D}")
#     # print(f"插入次数: {I}")
#
#     return errors, dp[len(s1)][len(s2)]  # 返回矩阵右下角的值


def resAna_csv_edit_distance(actual, predict):
    """
        计算两个字符串的编辑操作（替换、删除、插入）及其位置信息。
        返回：
            - errors: 操作列表，每个元素为 (s1_char, s2_char, position)
            - total_operations: 操作总数
        """
    s1 = predict
    s2 = actual
    m, n = len(s1), len(s2)
    # 初始化动态规划表，dp[i][j]表示s1前i个字符和s2前j个字符的编辑距离
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # 初始化操作记录表，存储每一步的操作
    operations = [[None] * (n + 1) for _ in range(m + 1)]

    # 初始化边界条件
    for i in range(1, m + 1):
        dp[i][0] = i
        operations[i][0] = ('delete', s1[i - 1], i - 1)
    for j in range(1, n + 1):
        dp[0][j] = j
        operations[0][j] = ('insert', s2[j - 1], 0)

    # 填充动态规划表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                operations[i][j] = ('match', None, i - 1)
            else:
                # 计算替换、删除、插入的代价
                replace_cost = dp[i - 1][j - 1] + 1
                delete_cost = dp[i - 1][j] + 1
                insert_cost = dp[i][j - 1] + 1

                # 选择最小代价的操作
                if replace_cost <= delete_cost and replace_cost <= insert_cost:
                    dp[i][j] = replace_cost
                    operations[i][j] = ('replace', (s1[i - 1], s2[j - 1]), i - 1)
                elif delete_cost <= insert_cost:
                    dp[i][j] = delete_cost
                    operations[i][j] = ('delete', s1[i - 1], i - 1)
                else:
                    dp[i][j] = insert_cost
                    operations[i][j] = ('insert', s2[j - 1], i)

    # 回溯操作路径，记录所有操作
    errors = []
    i, j = m, n
    while i > 0 or j > 0:
        op = operations[i][j]
        if op is None:
            break
        op_type, op_chars, pos = op

        if op_type == 'match':
            i -= 1
            j -= 1
        elif op_type == 'replace':
            s1_char, s2_char = op_chars
            errors.append((s1_char, s2_char, pos))
            i -= 1
            j -= 1
        elif op_type == 'delete':
            s1_char = op_chars
            errors.append((s1_char, '', pos))
            i -= 1
        elif op_type == 'insert':
            s2_char = op_chars
            errors.append(('', s2_char, pos))
            j -= 1

    # 反转操作顺序（因为回溯是逆序的）
    errors = errors[::-1]
    total_operations = len(errors)

    return errors, total_operations


def resAna_creErrorCSV(path, actuals, predicts):
    analy_data = []
    for i in range(len(actuals)):
        actual = actuals[i]
        predict = predicts[i]

        errors, dis = resAna_csv_edit_distance(actual, predict)
        analy_dict = {'actual': actual, 'predict': predict, 'errors': errors}
        analy_data.append(analy_dict)

    fields = ["actual", "predict", "errors"]
    csv_writer = csv.DictWriter(open(path, "w", newline="", encoding='utf-8'), fieldnames=fields)
    csv_writer.writeheader()
    csv_writer.writerows(analy_data)


def discrimeErrorType(error):
    id = 999
    if error[0] == "''":  # 纠错需要加字，所以属于3少字
        id = 3
    elif error[1] == "''":  # 纠错需要删字，所以属于2多字
        id = 2
    elif error[0] != "''" and error[1] != "''":  # 纠错需要替换，可能是近音字或未识别
        before = error[0].replace("'", "")  # lipreading predict, but CSC input
        after = error[1].replace("'", "")  # CSC output

        style = Style.NORMAL  # Pinyin_Tone
        py_before = lazy_pinyin(before, style=style)
        py_after = lazy_pinyin(after, style=style)
        if py_after == py_before:  # 声调Pinyin相等，则属于0谐音错误
            id = 0
        else:
            style = Style.FINALS  # Pinyin_FINALS
            py_before = lazy_pinyin(before, style=style)
            py_after = lazy_pinyin(after, style=style)
            if py_after == py_before:  # 韵母相等，则属于1近音错误
                id = 1
            else:  # 属于未识别错误
                id = 4
    return id


def resAna_error(path):
    with open(path, 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        error_type_count = [0, 0, 0, 0, 0]  # 谐音、韵母近音、多字、少字、未识别。加起来是错误总数。
        last_error_type = [0, 0, 0, 0, 0]
        error_sample_count = [0, 0, 0, 0, 0]  # 谐音、韵母近音、多字、少字、未识别。如果一个样本中的错误包含这类error，那就+1。
        err_list_single = []  # 存储单个样本的错误数量
        # error_samples = 0  # 错误样本总数
        samples_number = [0, 0, 0]  # 样本总数， 错误样本总数， 错误总数
        is_flag = 0
        for row in reader:
            is_flag = is_flag + 1
            if is_flag == 1:
                continue  # 跳过首行表头
            samples_number[0] += 1
            actual = row[0]
            predict = row[1]
            cer_now = torchaudio.functional.edit_distance(list(actual.replace('<unk>', '*')),
                                                          list(predict.replace('<unk>', '*')))
            if cer_now:
                # error_samples = error_samples + 1
                samples_number[1] += 1
            errors = row[2]

            if errors == '[]':
                err_list_single.append(0)
            else:
                single_count = 0  # 计算存在错误的（单）样本中错误数量
                if '), (' in errors:
                    errors = errors.split('), (')
                    for error in errors:
                        single_count = single_count + 1
                        error = error.replace('[(', '').replace(')]', '')
                        error = error.split(', ')
                        error_id = discrimeErrorType(error)
                        error_type_count[error_id] = error_type_count[error_id] + 1
                        # print('single:  ', error, error[0])
                    # print("____________________")
                else:
                    errors = errors.replace('[(', '').replace(')]', '')
                    error = errors.split(', ')
                    error_id = discrimeErrorType(error)
                    error_type_count[error_id] = error_type_count[error_id] + 1
                    single_count = single_count + 1
                err_list_single.append(single_count)
                samples_number[2] += single_count
            for i in range(len(last_error_type)):
                if last_error_type[i]!=error_type_count[i]:
                    error_sample_count[i] = error_sample_count[i] + 1
                    last_error_type[i] = error_type_count[i]
        f.close()
    # print('每个样本的错误数量：', err_list_single)
    return samples_number, error_type_count, error_sample_count


def get_HSR(samples_number, error_type_count, error_sample_count):
    total_samples, total_error_samples, total_errors = samples_number  # 样本总数， 错误样本总数， 错误总数
    # [谐音, 韵母近音, 多字, 少字, 未识别]
    T_HSRs = []
    HSRs = []
    HERs = []
    for i in range(len(error_sample_count)):
        error_samples = error_sample_count[i]
        errors = error_type_count[i]
        T_HSR = error_samples/total_samples
        HSR = error_samples/total_error_samples
        HER = errors/total_errors
        T_HSRs.append(T_HSR)
        HSRs.append(HSR)
        HERs.append(HER)

    return T_HSRs, HSRs, HERs


def get_word_num(word_num_list, char_list, sentence, data_type='zh'):
    '''
    Args:
        word_num_list: 存储字频，长度与char_list一致
        char_list: 字典，char_list[0]为unk
        sentence: 需要统计的句子
    Returns:
    '''
    if data_type!='zh':
        sentence = sentence.split(' ')
    for char in sentence:
        id = char_list.index(char) if char in char_list else 0
        word_num_list[id] += 1
    return word_num_list


def resAna_wordACC(data_path, char_list):
    word_num_list = np.zeros(len(char_list))
    word_num_list_error = np.zeros(len(char_list))

    with open(data_path, 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        error_samples = 0
        is_flag = 0
        for row in reader:
            is_flag = is_flag + 1
            if is_flag == 1:
                continue  # 跳过首行表头

            actual, predict, errors = row
            actual = actual.replace('\n', '')
            # predict = predict.replace('\n', '')
            if errors == '[]':
                word_num_list = get_word_num(word_num_list, char_list, actual)
            else:  # "[('起', '清', 3), ('处', '楚', 5)]"
                errors = ast.literal_eval(errors)
                wrong = ''
                ids = [x[2] for x in errors if x[1]!='']
                for i in range(len(actual)):
                    if i in ids:
                        wrong = wrong+actual[i]
                word_num_list = get_word_num(word_num_list, char_list, actual)
                word_num_list_error = get_word_num(word_num_list, char_list, wrong)
    return word_num_list, word_num_list_error


def get_cls_threshold(m, n):
    y_true = np.concatenate([np.ones_like(m), np.zeros_like(n)])
    y_scores = np.concatenate([m, n])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    youden_j = tpr - fpr

    # 忽略第一个阈值（inf）
    youden_j_no_inf = youden_j[1:]
    thresholds_no_inf = thresholds[1:]

    optimal_idx = np.argmax(youden_j_no_inf)
    optimal_threshold = thresholds_no_inf[optimal_idx]

    info = {
        "threshold": optimal_threshold,
        "sensitivity": tpr[optimal_idx+1],
        "specificity": 1 - fpr[optimal_idx+1],
        "youden_j": youden_j_no_inf[optimal_idx]
    }
    return optimal_threshold, info


def plot_threshold_hist(score_dict, key, save_dir, use_log_scale=False, log_base=10):
    positive_scores = score_dict['positive']
    negative_scores = score_dict['negative']

    plt.figure(figsize=(10, 6))
    bins = [i / 40 for i in range(41)]  # 0.0 ~ 1.0

    plt.hist(
        positive_scores, bins=bins, alpha=0.6, label='Positive Samples (TP + FN)',
        color='green', edgecolor='white'
    )
    plt.hist(
        negative_scores, bins=bins, alpha=0.6, label='Negative Samples (TN + FP)',
        color='red', edgecolor='white'
    )

    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Similarity Scores (Feature {key})')
    plt.legend()
    plt.grid(True)
    if use_log_scale:
        plt.yscale('log', base=log_base)
        # 设置刻度显示为对应的log_base的幂
        def log_formatter(x, pos):
            if x == 0:
                return '0'
            exponent = np.log(x) / np.log(log_base)
            if abs(exponent - round(exponent)) < 1e-10:
                return f'{log_base}^{int(round(exponent))}'
            else:
                return ''
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(log_formatter))

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"threshold_hist_key_{key}.png")
    plt.savefig(save_path)
    plt.close()


def plot_lipauth_threshold_hist(all_sim_scores, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 准备每个key下的正负样本
    scores_by_key = defaultdict(lambda: {'positive': [], 'negative': []})

    for item in all_sim_scores:
        # for key, (score, label) in item.items():
        for key, score_list in item.items():
            if len(score_list)==3:
                _, label, score = score_list
            else:
                score, label = score_list
            if label in ['TP', 'FN']:
                scores_by_key[key]['positive'].append(score)
            elif label in ['TN', 'FP']:
                scores_by_key[key]['negative'].append(score)

    # 逐个特征画图
    for key, score_dict in scores_by_key.items():
        plot_threshold_hist(score_dict, key, save_dir, use_log_scale=True, log_base=10)


def plot_lipauth_err(all_sim_scores, save_dir="far_frr_plots"):
    # Step 1: 预处理 all_sim_scores 为每个 key 对应的 [(score, 'Pos'/'Neg')]
    data_by_key = {}
    for sample in all_sim_scores:
        for key, score_list in sample.items():  # (score, label)
            if len(score_list) == 3:
                score, label, cer = score_list
            else:
                score, label = score_list
            label_pos_neg = 'Pos' if label in ['TP', 'FN'] else 'Neg'
            data_by_key.setdefault(key, []).append((score, label_pos_neg))

    # Step 2: 对每个 key 绘图
    keys_err = {}
    for key, samples in data_by_key.items():
        scores = np.array([s for s, _ in samples])
        labels = np.array([1 if l == 'Pos' else 0 for _, l in samples])

        thresholds = np.sort(np.unique(scores))
        fars, frrs = [], []

        for thresh in thresholds:
            preds = (scores >= thresh).astype(int)
            fa = np.sum((preds == 1) & (labels == 0))
            fr = np.sum((preds == 0) & (labels == 1))
            total_neg = np.sum(labels == 0)
            total_pos = np.sum(labels == 1)
            far = fa / total_neg if total_neg else 0
            frr = fr / total_pos if total_pos else 0
            fars.append(far)
            frrs.append(frr)

        fars, frrs = np.array(fars), np.array(frrs)
        abs_diff = np.abs(fars - frrs)
        err_idx = np.argmin(abs_diff)
        err_threshold = thresholds[err_idx]
        err = (fars[err_idx] + frrs[err_idx]) / 2
        keys_err[key] = err

        # Plot
        plt.figure()
        plt.plot(thresholds, fars, label="FAR", color="red")
        plt.plot(thresholds, frrs, label="FRR", color="blue")
        plt.plot(err_threshold, fars[err_idx], 'ko', label=f"ERR={err:.3f}")
        plt.axvline(err_threshold, linestyle="--", color="gray", alpha=0.5)

        plt.xlabel("Threshold")
        plt.ylabel("Error Rate")
        plt.title(f"FAR/FRR Curve for Key {key}\nERR @ threshold={err_threshold:.3f}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        save_path = os.path.join(save_dir, f"far_frr_key_{key}.png")
        plt.savefig(save_path)
        plt.close()
    return keys_err


def cal_lipauth_matrics(all_sim_scores, key_thresholds):
    data_by_key = {}
    # 按照特征类型组织得分和混淆结果
    for item in all_sim_scores:
        for key, score_list in item.items():  # (score, label)
            if len(score_list) == 3:
                score, result, cer = score_list
            else:
                score, result = score_list
            if key not in data_by_key:
                data_by_key[key] = {"scores": [], "labels": []}
            label = 1 if result in ["TP", "FN"] else 0  # 正样本为1，负样本为0
            data_by_key[key]["scores"].append(score)
            data_by_key[key]["labels"].append(label)

    opt_thresholds = {}
    confusion_metrics = {}
    key_metrics = {}

    # 逐特征处理
    for key in data_by_key:
        scores = np.array(data_by_key[key]["scores"])
        labels = np.array(data_by_key[key]["labels"])

        # 1. 计算EER阈值
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer_threshold = thresholds[eer_idx]
        eer = fpr[eer_idx]
        opt_thresholds[key] = float(eer_threshold)

        # 2. 基于 key_thresholds[key] 计算混淆矩阵
        threshold = key_thresholds[key]
        preds = (scores >= threshold).astype(int)
        TP = int(np.sum((preds == 1) & (labels == 1)))
        FP = int(np.sum((preds == 1) & (labels == 0)))
        TN = int(np.sum((preds == 0) & (labels == 0)))
        FN = int(np.sum((preds == 0) & (labels == 1)))
        confusion_metrics[key] = {"TP": TP, "FP": FP, "TN": TN, "FN": FN}

        # 3. 计算指标
        ACC = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
        FAR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        FRR = FN / (TP + FN) if (TP + FN) > 0 else 0.0
        AUC = metrics.roc_auc_score(labels, scores)

        # 4. 计算 TPR @ FAR=0.001, 0.01
        def compute_tpr_at_far(target_far):
            for fpr_val, tpr_val in zip(fpr, tpr):
                if fpr_val >= target_far:
                    return float(tpr_val)
            return float(tpr[-1])  # fallback

        tpr_0_1 = compute_tpr_at_far(0.001)
        tpr_1 = compute_tpr_at_far(0.01)

        key_metrics[key] = {
            "ACC": float(ACC),
            "AUC": float(AUC),
            "FAR": float(FAR),
            "FRR": float(FRR),
            "EER": float(eer),
            "TPR@0.1%": tpr_0_1,
            "TPR@1%": tpr_1
        }

    # ==== 系统级评估 ====
    total_labels = []
    total_preds = []
    for item in all_sim_scores:
        types = list(item.keys())
        is_positive = any(item[t][1] in ["TP", "FN"] for t in types)  # 只要任一类型是正样本就认为是正样本
        pred_true = all(item[t][0] >= key_thresholds[t] for t in types)

        total_labels.append(1 if is_positive else 0)
        total_preds.append(1 if pred_true else 0)

    total_labels = np.array(total_labels)
    total_preds = np.array(total_preds)

    TP = int(np.sum((total_preds == 1) & (total_labels == 1)))
    FP = int(np.sum((total_preds == 1) & (total_labels == 0)))
    TN = int(np.sum((total_preds == 0) & (total_labels == 0)))
    FN = int(np.sum((total_preds == 0) & (total_labels == 1)))

    Total_confusion_metrics = {"TP": TP, "FP": FP, "TN": TN, "FN": FN}

    ACC = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    FAR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    FRR = FN / (TP + FN) if (TP + FN) > 0 else 0.0

    Total_metrics = {
        'ACC': float(ACC),
        'FAR': float(FAR),
        'FRR': float(FRR)
        # 注意：不含TPR@X% 和 AUC，因系统级判断非线性门控结构
    }

    return opt_thresholds, confusion_metrics, key_metrics, Total_confusion_metrics, Total_metrics

