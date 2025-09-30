import numpy as np
import torch

def convert_binding_site_to_labels(binding_site, max_length):
    
    '''
    Convert binding site (str) to binary label
    e.g., "3, 5, 6, 10" -> 0010110001
    '''
    
    binding_site_labels = list()
    
    for idx, bs in enumerate(binding_site):
        targets = np.array(sorted(list(set(list(map(int, bs.split(",")))))))
        
        #단위행렬
        one_hot_targets = np.eye(max_length)[targets]
        one_hot_targets = np.sum(one_hot_targets, axis = 0) 
        binding_site_labels.append(one_hot_targets)
    
    return binding_site_labels
    
def prepare_prots_input(config, datasets, training = True):
    
    # Get features
    prot_ids = [data[0] for data in datasets] 
    prot_feats = [data[1] for data in datasets] #embedding한 protein
    prot_seqs = [data[2] for data in datasets]

    # Collate batch data
    aa_feat, protein_feat, prots_mask, position_ids, chain_idx = collate_prots_feats(config, prot_feats, prot_seqs)
    
    # Cast to tensor
    aa_feat = torch.tensor(aa_feat, dtype = torch.float32).cuda()
    protein_feat = torch.tensor(protein_feat, dtype = torch.float32).cuda()
    prots_mask = torch.tensor(prots_mask, dtype = torch.long).cuda()
    position_ids = torch.tensor(position_ids, dtype = torch.long).cuda()
    chain_idx = torch.tensor(chain_idx, dtype = torch.long).cuda()
    
    # 학습 모드라면, binding site를 binary label로 변형
    if training:
        prot_binding_sites = convert_binding_site_to_labels([data[3] for data in datasets],
                                                    config["prots"]["max_lengths"])
        pbs_np = np.asarray(prot_binding_sites, dtype=np.float32)  # 리스트 -> 단일 ndarray
        prot_binding_sites = torch.from_numpy(pbs_np).to(dtype=torch.float32, device="cuda", non_blocking=True)
        return aa_feat, protein_feat, prots_mask, prot_binding_sites, position_ids, chain_idx
        
    return aa_feat, protein_feat, prots_mask, position_ids, chain_idx


def collate_prots_feats(config, feats, seqs):
#feat는 (B,L,H) 단백질 임베딩 리스트
#seqs는 각 단백질 아미노산 서열을 , 로 구분한 문자열
#즉, feats에는 chain에 관련된 정보가 없으나 seqs를 통해 chain 정보를 줌
    
    #protein별 길이 계산 후, mask제작 (padding으로 채우기 전)
    lengths = [len(i) for i in feats]
    max_length = config["prots"]["max_lengths"]
    hidden_dim = config["architectures"]["prots_input_dim"]
    input_mask = [[1] * length for length in lengths]
    
    position_id, chain_idx = list(), list()

    #0으로만 채운 (B,L,H)
    aa_feat = np.zeros((len(feats), max_length, hidden_dim))
    protein_feat = np.zeros((len(feats), max_length, hidden_dim))

    for idx, line in enumerate(feats):

        #feat 복사
        seq_length = line.shape[0]
        aa_feat[idx,:seq_length,:] = line 

        #input mask에 padding추가
        #즉, input mask는 (B,L)
        pad_mask = [0] * (max_length - seq_length)
        input_mask[idx].extend(pad_mask)
        
        # 각 위치 인덱스 [0, 1, 2, ..., max_length-1]
        position_id.append([i for i in range(max_length)])

        #seq_list에는 단백질이 chain별로 나누어져있음
        seq_list = seqs[idx].split(',')
        start_seq, end_seq = 0, 0
        tmp_chain_idx = list()
        
        #체인마다 해당 구간 [start_seq:end_seq]의 임베딩들을 합해 그 값을 해당 구간 전체 위치에 복사
        for jdx, chain_seq in enumerate(seq_list):
            end_seq += len(chain_seq)
            protein_feat[idx, start_seq:end_seq, :] = np.mean(line[start_seq:end_seq, :], axis=0)
            start_seq = end_seq
            
            # chain index, 0은 패딩용이라서 index로 사용 불가 즉, +1부터 시작
            for i in range(len(chain_seq)):
                tmp_chain_idx.append(jdx + 1)
        
        for i in range(seq_length, max_length):
            tmp_chain_idx.append(0)
        
        chain_idx.append(tmp_chain_idx)

    return aa_feat, protein_feat, input_mask, position_id, chain_idx

def convert_bs(pred_binding_sites):
    
    prediction_row, prediction_cols = np.where((pred_binding_sites>0.5) == True)
    
    final_results, check_row, check_col = list(), set(), list()
    
    for pred_r, pred_c in zip(prediction_row, prediction_cols):
        if len(check_row) == 0 and pred_r not in check_row:
            check_row.add(pred_r)
            check_col.append(pred_c)
            
        elif pred_r in check_row:
            check_col.append(pred_c)
            
        elif pred_r not in check_row:    
            final_results.append(",".join(list(map(str, check_col))))
            check_col = list()
            check_row.add(pred_r)
            check_col.append(pred_c)
            
    final_results.append(",".join(list(map(str, check_col))))
    
    return final_results

def get_results(binding_sites, pred_binding_sites, sequences):
    
    T_TP, T_TN, T_FP, T_FN = 0, 0, 0, 0
    
    for bs, bps, seq in zip(binding_sites, pred_binding_sites, sequences):
        seq_len = len([i for i in seq if i != ","])
        index = [str(i) for i in range(seq_len)]
        
        positive_label = set(get_bs(bs))
        negative_label = set(index) - set(positive_label)
        positive_pred = set(bps.split(","))
        negative_pred = set(index) - set(positive_pred)
        
        TP = len(positive_pred & positive_label)
        TN = len(negative_pred & negative_label)
        FP = len(positive_pred & negative_label)
        FN = len(negative_pred & positive_label) 
        
        T_TP += TP
        T_TN += TN
        T_FP += FP
        T_FN += FN
        
    precision = T_TP / (T_TP + T_FP)
    recall = T_TP / (T_TP + T_FN)
    specificity = T_TN / (T_TN + T_FP) 
    G_mean = np.sqrt(specificity * recall)
    ACC = (T_TP + T_TN) / (T_TP + T_TN + T_FP + T_FN)
    F1_score = (2 * precision * recall) / (precision + recall)
    F2_score = (5 * precision * recall) / (4 * precision + recall)
    
    return np.round(precision,2), np.round(recall,2), np.round(specificity, 2), np.round(ACC, 2), np.round(G_mean, 2), np.round(F1_score, 2), np.round(F2_score, 2)

def get_bs(binding_sites):
    results = list()
    for bs in binding_sites.split("|"):
        results.extend(bs.split(","))
    return results