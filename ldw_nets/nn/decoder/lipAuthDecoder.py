import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


__all__ = (
    "LipAuthOutput",
)


class LipAuthOutput(nn.Module):
    def __init__(self, root_dir: str, support_path: str, id_list: List[str], save_layerFeatures: List[float], auth_type="eval"):
        super().__init__()
        """
        初始化 LipAuthOutput 类
        :param support_path: 注册用户特征存储目录
        :param id_list: 注册用户ID列表
        :param threshold: 认证相似度阈值
        """
        self.root_dir = root_dir
        # 'data/lipauth_support/VSR_M1/5_grid_multitask_test'

        if root_dir not in support_path:
            self.support_path = root_dir + "/" +support_path
        else:
            self.support_path = support_path
        self.id_list = id_list  # [1,2,20,22]
        # [['3', 'ctc', '0.75'], ['5', 'cos', '0.75']]
        self.save_layerFeatures = save_layerFeatures

        # 确保目录存在
        if auth_type == "demo":
            self.registered_features = self.load_registered_VSA()

    def load_registered_VSA(self, ) -> None:
        """
        加载数据库中所有注册用户的人脸特征
        假设每个用户特征存储在 {support_path}/{user_id}.pt 文件中
        """
        feature_db = {}
        for filename in os.listdir(self.support_path):
            if filename.endswith('.pt'):
                # 解析文件名 (id-num.pt)
                id_part = filename.split('_')[0]
                if not int(id_part) in self.id_list:
                    continue
                utt_part = filename.split('_')[1].replace('.pt', '')
                if not self.utt_list == "all":
                    if not int(utt_part) in self.utt_list:
                        continue
                # num_part = filename.split('_')[1].split('.')[0]

                # 加载特征向量
                feature = torch.load(os.path.join(self.support_path, filename))

                # 确保特征向量是1×512
                # assert feature.shape == (1, 512), f"特征向量 {filename} 的维度不正确"

                # 添加到数据库
                if id_part not in feature_db:
                    feature_db[id_part] = []
                feature_db[id_part].append(feature)

        return feature_db

    def save_registered_features(self, actuals, features, path: str) -> None:
        """
        保存单个用户的特征到数据库
        :param user_id: 用户ID
        """
        if isinstance(actuals, list):
            actuals, spoofer = actuals
        else:
            spoofer = None
        sentence_id = path.split('/')[-1].replace('.mp4', '')
        if actuals in self.id_list:
            model_name = self.support_path.split('/')[-2]
            for key, value in features.items():
                rep = self.support_path.split('/')[-1].split('_')[0]
                self.support_path = self.support_path.replace(model_name+'/'+rep, model_name+'/'+str(key))

                if spoofer:
                    feature_path = self.support_path + "/" + str(actuals) + "_" + str(spoofer) + "_" + str(sentence_id) + ".pt"
                else:
                    feature_path = self.support_path + "/" + str(actuals) + "_" + str(sentence_id) + ".pt"

                torch.save(value, feature_path)
                print(f"[OK] 用户 {actuals} 的特征已保存至 {feature_path}")

    def authentication_VSA(self, query: torch.Tensor) -> Tuple[str, float]:
        """
        批量计算相似度的优化版本
        """
        if not self.registered_features:
            print("未加载注册用户特征，请先调用 load_registered_features()")
            #raise ValueError("未加载注册用户特征，请先调用 load_registered_features()")
            return None, None

        # assert query.shape == (1, 512), "查询向量的维度不正确"

        max_similarity = -1
        best_id = None

        for id_str, features in self.registered_features.items():
            # 将当前ID的所有特征堆叠成(n,512)张量
            features_stack = torch.cat(features, dim=0)  # (n,512)

            # 扩展查询向量以匹配维度(n,512)
            query_expanded = query.expand_as(features_stack)
            # 计算余弦相似度 (结果形状为(n,))
            similarities = F.cosine_similarity(query_expanded, features_stack, dim=1)

            # cos_sim = F.cosine_similarity(
            #     query.unsqueeze(1),  # [1, 1, 512]
            #     features_stack.unsqueeze(0),  # [1, 9, 512]
            #     dim=2
            # ).squeeze()
            # euclidean_dist = torch.cdist(query, features_stack, p=2).squeeze()
            # cos_norm = (cos_sim + 1) / 2
            # euc_norm = 1 / (1 + euclidean_dist)
            # alpha = 0.7  # 余弦相似度权重
            # similarities = alpha * cos_norm + (1 - alpha) * euc_norm

            # 获取当前ID的最高相似度
            current_max, _ = torch.max(similarities, dim=0)
            current_max = current_max.item()

            if current_max > max_similarity:
                max_similarity = current_max
                best_id = id_str

        # 判断是否超过阈值
        if max_similarity >= self.threshold:
            return best_id, max_similarity
        else:
            return None, None

    def authentication(self, label, support, query, cer=None):
        """
        批量计算相似度的优化版本
        """
        # 'Neg s22_bgir3n.mp4 s2_lwik8s.mp4'
        # [['3', 'ctc', '0.75'], ['5', 'cos', '0.75']]
        model_name = self.support_path.split("/")[-2]

        sim_dict = {}
        support_features = []
        for feature in self.save_layerFeatures:
            rep = self.support_path.split("/")[-1].split("_")[0]
            fea_dir = self.support_path.replace(model_name+"/"+rep, model_name+"/"+feature[0])

            query_path = fea_dir + "/" + query.replace(".mp4", ".pt").replace("\n", "")
            query_feature = torch.load(query_path)
            if isinstance(query_feature, dict):
                query_feature = query_feature["inputs"]

            if isinstance(support, list):
                for support_ in support:
                    support_path_ = fea_dir + "/" + support_.replace(".mp4", ".pt").replace("\n", "")
                    support_feature_ = torch.load(support_path_)
                    if isinstance(support_feature_, dict):
                        support_feature_ = support_feature_["inputs"]
                    support_features.append(support_feature_)
                support_feature = torch.cat(support_features, dim=0)
                query_feature = query_feature.expand(len(support_features), -1, -1)
            else:
                support_path = fea_dir + "/" + support.replace(".mp4", ".pt").replace("\n", "")
                support_feature = torch.load(support_path)
                if isinstance(support_feature, dict):
                    support_feature = support_feature["inputs"]

            if feature[1] == "cos":
                support_feature = support_feature.view(support_feature.shape[0], -1)
                query_feature = query_feature.view(query_feature.shape[0], -1)
                if query_feature.shape[0]==1:
                    cos_sim = F.cosine_similarity(support_feature, query_feature, dim=1).item()
                else:
                    cos_sim = F.cosine_similarity(support_feature, query_feature, dim=1).mean().item()
                cos_sim = (cos_sim + 1) / 2

                if label == 'Pos':
                    if cos_sim >= float(feature[2]):
                        sim_dict[feature[0]] = [cos_sim, 'TP']
                    else:
                        sim_dict[feature[0]] = [cos_sim, 'FN']
                else:
                    if cos_sim >= float(feature[2]):
                        sim_dict[feature[0]] = [cos_sim, 'FP']
                    else:
                        sim_dict[feature[0]] = [cos_sim, 'TN']

            elif feature[1] in ["ctc", "ce", "att"]:  # 注意：用损失来验证时，要采用CER＞阈值来构建label
                cos_sim = -query_feature.item()
                if cer <= 0.2:  # 'Pos':
                    if cos_sim < float(feature[2]):
                        sim_dict[feature[0]] = [cos_sim, 'TP', cer]  # 预测为正，实际为正
                    else:
                        sim_dict[feature[0]] = [cos_sim, 'FN', cer]  # 预测为负，实际为正
                else:
                    if cos_sim < float(feature[2]):
                        sim_dict[feature[0]] = [cos_sim, 'FP', cer]  # 预测为正，实际为负
                    else:
                        sim_dict[feature[0]] = [cos_sim, 'TN', cer]
            elif feature[1] in ["cer"]:
                cos_sim, label = cer
                if label <= 0.2:  # 'Pos':
                    if cos_sim < 0.2:
                        sim_dict[feature[0]] = [-cos_sim, 'TP', label]  # 预测为正，实际为正
                    else:
                        sim_dict[feature[0]] = [-cos_sim, 'FN', label]  # 预测为负，实际为正
                else:
                    if cos_sim < 0.2:
                        sim_dict[feature[0]] = [-cos_sim, 'FP', label]  # 预测为正，实际为负
                    else:
                        sim_dict[feature[0]] = [-cos_sim, 'TN', label]
            else:
                raise ValueError("特征匹配方式未记录！！！")

        return sim_dict
