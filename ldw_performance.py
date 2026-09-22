import numpy as np
import yaml
import os
import ast
import csv

from ldw_utils.torch_utils import get_filetype
from ldw_utils.eval_matrix import (
    resAna_cer,
    res_cerDivide,
    resAna_creErrorCSV,
    resAna_error,
    resAna_wordACC,
    count_cers,
    compute_cer_ser_255075,
    get_HSR
)
from ldw_utils.plot import (
    word_num_visMatrix,
    sentenceLength_num_visColumnchart,
    frames_visHistchart,
    cers_visHistchart,
    draw_error_samples_rate_pie
)


def get_result(path):
    actuals = []
    predicts = []

    with open(path, 'r', encoding=get_filetype(path)) as f:
        lines = f.readlines()
        for line in lines:
            if "actual" in line:
                actual = line.split('   ')[1]
                predict = line.split('   ')[2]
                # actual, predict, _, _ = line.split('   ')
                actual = actual.replace('actual: ', '').replace('<unk>', '*')
                if actual[len(actual) - 1] == '*':
                    actual = actual.replace('*', '')
                predict = predict.replace('predict: ', '').replace('<unk>', '*')
                actuals.append(actual)
                predicts.append(predict)
    return actuals, predicts


# -------------------------------------------------------


def dataset_analysis(root_dir, data, subset='test'):
    path = root_dir+'/ldw_cfg/dataset/'+data
    data_root = root_dir + '/data/'
    with open(path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    char_path = config['char_path']
    char_list = open(root_dir+'/'+char_path, 'r').readlines()[1:-1]
    for i in range(len(char_list)):
        char_list[i] = char_list[i].replace('\n', '')
    if subset=='train':
        data_csv = config['train']
    elif subset=='test' or subset=='eval':
        data_csv = config['test']
    elif subset=='val':
        data_csv = config['val']
    else:
        print("subset is wrong!!!!!!!!!!!!!!!")

    word_num_list = np.zeros(len(char_list))  # 字/词频统计
    sentenceLength_num_dict = {'6':0}  # 语料长度统计
    frames_list = []  # 时长统计
    with open(root_dir+'/'+data_csv, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data_name, video_path, frames, token_ids = row
            frames_list.append(int(frames))
            token_ids = token_ids.split(' ')
            ids = [int(x) for x in token_ids if x]
            video_path = data_root + data_name + '/' + video_path
            txt_path = video_path.replace('cmlr_video_seg24s', 'cmlr_text_seg24s')[:-4]+'.txt'
            content = open(txt_path, 'r').readlines()[0].replace(' ', '')
            for c in content:
                id = char_list.index(c) if c in char_list else 0
                word_num_list[id] += 1
            if str(len(content)) in sentenceLength_num_dict:
                sentenceLength_num_dict[str(len(content))] += 1
            else:
                sentenceLength_num_dict[str(len(content))] = 1
    # 绘制字频统计图  #
    save_path = root_dir +'/output/eval_dataset/'+data.replace('.yaml', '')+'_word_num_visMatrix.png'
    word_num_visMatrix(save_path, word_num_list, char_list)

    # 绘制语料长度统计图  #
    save_path = root_dir + '/output/eval_dataset/' + data.replace('.yaml', '') + '_sentenceLength_num_visColumnchart.png'
    sentenceLength_num_visColumnchart(save_path, sentenceLength_num_dict)

    # 绘制样本时长直方图
    save_path = root_dir + '/output/eval_dataset/' + data.replace('.yaml', '') + '_frames_visHistchart.png'
    frames_visHistchart(save_path, frames_list)

    with open(root_dir + '/output/eval_dataset/' + data.replace('.yaml', '') + '.txt', 'w') as f:
        # 写入统计信息
        f.write("total samples: "+str(len(frames_list))+"\n")
        f.write("total hours: "+str(sum(frames_list)/25/60/60)+"   min seconds: "+str(min(frames_list))+"   max seconds: "+str(max(frames_list)))

        f.write("\n\n")
        f.write("---------sentence length count--------\n")
        for i,(key, value) in enumerate(sentenceLength_num_dict.items()):
            f.write(str(key)+" "+str(value)+"\n")

        f.write("\n\n")
        f.write("---------word/char frequencey count--------\n")
        for i in range(len(char_list)):
            f.write(char_list[i] + ": " + str(word_num_list[i]) + "\n")

        f.write("\n over.")
    f.close()


def model_analysis(root_dir, data, subset='test'):
    # parameters, FLOPs, speed
    return print("model_analysis is yet to be developed")


def results_analysis(root_dir, data, result_files):
    path = root_dir + '/ldw_cfg/dataset/' + data

    # get char_list
    with open(path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    char_path = config['char_path']
    encoding = get_filetype(root_dir + '/' + char_path)
    char_list = open(root_dir + '/' + char_path, 'r', encoding=encoding).readlines()[1:-1]
    for i in range(len(char_list)):
        char_list[i] = char_list[i].replace('\n', '')

    if not isinstance(result_files, list):
        result_files = [result_files]
    for result_file in result_files:
        # model_name = result_file.split('/')[-1].replace('.txt', '')
        # if os.path.exists(result_file):
        #     path = root_dir + '/output/' + result_file
        # else:
        #     path = result_file
        path = result_file  # root_dir + '/output/' + result_file
        actuals, predicts = get_result(path)

        with open(path.replace('.txt', '_performace.txt'), 'w') as f:
            f.write('\n\n')
            f.write(path.replace('.txt', '_performace.txt'))
            f.write('\n\n')

            CER, CERs, dises = resAna_cer(actuals, predicts)  # [cer_total, cers]
            # Total Accuracy： 20418 samples，CER = 0.098932，SER = 0.546953.
            total_samples = len(actuals)
            SER = 1 - sum(1 for x in CERs if x == 0)/len(CERs)
            # Character Error Rate (CER) @[ CER=      all | length=    all] = 0.098932
            CER_SML, SER_SML, count_ = compute_cer_ser_255075(total_samples, CERs, dises, actuals)
            CER_25, CER_50, CER_75, CER_S, CER_M, CER_L = CER_SML
            print('Total Accuracy： '+str(total_samples)+' samples，CER = '+str(CER)[:8]+'，SER = '+str(SER)[:8]+'.\n')
            f.write('Total Accuracy： ' + str(total_samples) + ' samples，CER = ' + str(CER)[:8] + '，SER = ' + str(SER)[:8] + '.\n')
            print('Character Error Rate (CER) @[ CER=      all | length=    all] = '+str(CER)[:8])
            print('Character Error Rate (CER) @[ CER=    :0.25 | length=    all] = ' + str(CER_25)[:8])
            print('Character Error Rate (CER) @[ CER=    :0.50 | length=    all] = ' + str(CER_50)[:8])
            print('Character Error Rate (CER) @[ CER=    :0.75 | length=    all] = ' + str(CER_75)[:8])
            print('Character Error Rate (CER) @[ CER=      all | length=  0: 10] = ' + str(CER_S)[:8])
            print('Character Error Rate (CER) @[ CER=      all | length= 10: 20] = ' + str(CER_M)[:8])
            print('Character Error Rate (CER) @[ CER=      all | length= 20:  X] = ' + str(CER_L)[:8])
            print('')
            f.write('Character Error Rate (CER) @[ CER=      all | length=    all] = '+str(CER)[:8]+'\n')
            f.write('Character Error Rate (CER) @[ CER=    :0.25 | length=    all] = ' + str(CER_25)[:8]+'\n')
            f.write('Character Error Rate (CER) @[ CER=    :0.50 | length=    all] = ' + str(CER_50)[:8]+'\n')
            f.write('Character Error Rate (CER) @[ CER=    :0.75 | length=    all] = ' + str(CER_75)[:8]+'\n')
            f.write('Character Error Rate (CER) @[ CER=      all | length=  0: 10] = ' + str(CER_S)[:8]+'\n')
            f.write('Character Error Rate (CER) @[ CER=      all | length= 10: 20] = ' + str(CER_M)[:8]+'\n')
            f.write('Character Error Rate (CER) @[ CER=      all | length= 20:  X] = ' + str(CER_L)[:8]+'\n')
            f.write('\n')
            # Sentence Error Rate (SER) @[ CER=      all | length=    all] = 0.546953
            SER_25, SER_50, SER_75, SER_100, SER_S, SER_M, SER_L = SER_SML
            print('Sentence Error Rate (SER) @[ CER=      all | length=    all] = ' + str(SER)[:8])
            print('Sentence Error Rate (SER) @[ CER=    :0.25 | length=    all] = ' + str(SER_25)[:8])
            print('Sentence Error Rate (SER) @[ CER=    :0.50 | length=    all] = ' + str(SER_50)[:8])
            print('Sentence Error Rate (SER) @[ CER=    :0.75 | length=    all] = ' + str(SER_75)[:8])
            print('Sentence Error Rate (SER) @[ CER=    :1.00 | length=    all] = ' + str(SER_100)[:8])
            print('Sentence Error Rate (SER) @[ CER=      all | length=  0: 10] = ' + str(SER_S)[:8])
            print('Sentence Error Rate (SER) @[ CER=      all | length= 10: 20] = ' + str(SER_M)[:8])
            print('Sentence Error Rate (SER) @[ CER=      all | length= 20:  X] = ' + str(SER_L)[:8])
            f.write('Sentence Error Rate (SER) @[ CER=      all | length=    all] = ' + str(SER)[:8] + '\n')
            f.write('Sentence Error Rate (SER) @[ CER=    :0.25 | length=    all] = ' + str(SER_25)[:8] + '\n')
            f.write('Sentence Error Rate (SER) @[ CER=    :0.50 | length=    all] = ' + str(SER_50)[:8] + '\n')
            f.write('Sentence Error Rate (SER) @[ CER=    :0.75 | length=    all] = ' + str(SER_75)[:8] + '\n')
            f.write('Sentence Error Rate (SER) @[ CER=    :1.00 | length=    all] = ' + str(SER_100)[:8] + '\n')
            f.write('Sentence Error Rate (SER) @[ CER=      all | length=  0: 10] = ' + str(SER_S)[:8] + '\n')
            f.write('Sentence Error Rate (SER) @[ CER=      all | length= 10: 20] = ' + str(SER_M)[:8] + '\n')
            f.write('Sentence Error Rate (SER) @[ CER=      all | length= 20:  X] = ' + str(SER_L)[:8] + '\n')
            f.write('\n\n')
            print('\n')
            # Errors
            # count error: 5 type，谐音、韵母近音、多字、少字、未识别。
            csv_path = path.replace('.txt', '.csv')
            resAna_creErrorCSV(csv_path, actuals, predicts)
            samples_number, error_type_count, error_sample_count = resAna_error(csv_path)
            #   # samples_number=[样本总数， 错误样本总数， 错误总数]
            #   # error_type_count=[谐音, 韵母近音, 多字, 少字, 未识别]  统计错误数量
            #   # error_sample_count=[谐音, 韵母近音, 多字, 少字, 未识别] 统计样本数量
            print('Total Errors： '+str(total_samples)+' samples，'+str(samples_number[1])+' error samples, '+str(samples_number[2])+' errors.\n')
            print('    T-HSR: Total Homophone-error Samples Rate, (Homophone Error samples)/(Total samples)')
            print('    HSR: Homophone-error Samples Rate, (Homophone Error samples)/(Error samples)')
            print('    HER: Homophone Error Rate, (Homophone Error)/(Total errors)')
            print('')
            f.write('Total Errors： ' + str(total_samples) + ' samples，' + str(samples_number[1]) + ' error samples, ' + str(samples_number[2]) + ' errors.\n')
            f.write('    T-HSR: Total Homophone-error Samples Rate, (Homophone Error samples)/(Total samples)\n')
            f.write('    HSR: Homophone-error Samples Rate, (Homophone Error samples)/(Error samples)\n')
            f.write('    HER: Homophone Error Rate, (Homophone Error)/(Total errors)\n')

            T_HSRs, HSRs, HERs = get_HSR(samples_number, error_type_count, error_sample_count)
            print('Homophone Error:      '+str(error_sample_count[0])+' error samples, '+str(error_type_count[0])+' errors;')
            print('                      T-HSR='+str(T_HSRs[0])[:8]+', HSR='+str(HSRs[0])[:8]+', HER='+str(HERs[0])[:8]+';')
            f.write('Homophone Error:      '+str(error_sample_count[0])+' error samples, '+str(error_type_count[0])+' errors;\n')
            f.write('                      T-HSR=' + str(T_HSRs[0])[:8] + ', HSR=' + str(HSRs[0])[:8] + ', HER=' + str(HERs[0])[:8] + ';\n')

            print('Near Homophone Error: ' + str(error_sample_count[1]) + ' error samples, ' + str(error_type_count[1]) + ' errors;')
            print('                      T-HSR=' + str(T_HSRs[1])[:8] + ', HSR=' + str(HSRs[1])[:8] + ', HER=' + str(HERs[1])[:8] + ';')
            f.write('Near Homophone Error: ' + str(error_sample_count[1]) + ' error samples, ' + str(error_type_count[1]) + ' errors;\n')
            f.write('                      T-HSR=' + str(T_HSRs[1])[:8] + ', HSR=' + str(HSRs[1])[:8] + ', HER=' + str(HERs[1])[:8] + ';\n')

            print('Additional Error:     ' + str(error_sample_count[2]) + ' error samples, ' + str(error_type_count[2]) + ' errors;')
            print('                      T-HSR=' + str(T_HSRs[2])[:8] + ', HSR=' + str(HSRs[2])[:8] + ', HER=' + str(HERs[2])[:8] + ';')
            f.write('Additional Error:     ' + str(error_sample_count[2]) + ' error samples, ' + str(error_type_count[2]) + ' errors;\n')
            f.write('                      T-HSR=' + str(T_HSRs[2])[:8] + ', HSR=' + str(HSRs[2])[:8] + ', HER=' + str(HERs[2])[:8] + ';\n')

            print('Missing Error:        ' + str(error_sample_count[3]) + ' error samples, ' + str(error_type_count[3]) + ' errors;')
            print('                      T-HSR=' + str(T_HSRs[3])[:8] + ', HSR=' + str(HSRs[3])[:8] + ', HER=' + str(HERs[1])[:8] + ';')
            f.write('Missing Error:        ' + str(error_sample_count[3]) + ' error samples, ' + str(error_type_count[3]) + ' errors;\n')
            f.write('                      T-HSR=' + str(T_HSRs[3])[:8] + ', HSR=' + str(HSRs[3])[:8] + ', HER=' + str(HERs[3])[:8] + ';\n')

            print('Unrecognized Error:   ' + str(error_sample_count[4]) + ' error samples, ' + str(error_type_count[4]) + ' errors;')
            print('                      T-HSR=' + str(T_HSRs[4])[:8] + ', HSR=' + str(HSRs[4])[:8] + ', HER=' + str(HERs[4])[:8] + ';')
            f.write('Unrecognized Error:   ' + str(error_sample_count[4]) + ' error samples, ' + str(error_type_count[4]) + ' errors;\n')
            f.write('                      T-HSR=' + str(T_HSRs[4])[:8] + ', HSR=' + str(HSRs[4])[:8] + ', HER=' + str(HERs[4])[:8] + ';\n')

            # #################### 计算level errors #################
            print('\n\n\n')
            # 1. 根据level划分actual和predict
            actuals_level, predicts_level = res_cerDivide(actuals, predicts, datatype='zh')
            for i in range(5):
                csv_path = path.replace('.txt', '_level'+str(i)+'.csv')
                resAna_creErrorCSV(csv_path, actuals_level[i], predicts_level[i])
                samples_number, error_type_count, error_sample_count = resAna_error(csv_path)

                print('Level '+str(i)+' Total Errors： ' + str(len(actuals_level[i])) + ' samples，' + str(samples_number[1]) + ' error samples, ' + str(samples_number[2]) + ' errors.')
                f.write('Level '+str(i)+' Total Errors： ' + str(len(actuals_level[i])) + ' samples，' + str(samples_number[1]) + ' error samples, ' + str(samples_number[2]) + ' errors.\n')

                print('Level '+str(i)+'  Homophone Error:      ' + str(error_sample_count[0]) + ' error samples, ' + str(error_type_count[0]) + ' errors;')
                f.write('Level '+str(i)+'  Homophone Error:      ' + str(error_sample_count[0]) + ' error samples, ' + str(error_type_count[0]) + ' errors;\n')

                print('Level '+str(i)+'  Near Homophone Error: ' + str(error_sample_count[1]) + ' error samples, ' + str(error_type_count[1]) + ' errors;')
                f.write('Level '+str(i)+'  Near Homophone Error: ' + str(error_sample_count[1]) + ' error samples, ' + str(error_type_count[1]) + ' errors;\n')

                print('Level '+str(i)+'  Additional Error:     ' + str(error_sample_count[2]) + ' error samples, ' + str(error_type_count[2]) + ' errors;')
                f.write('Level '+str(i)+'  Additional Error:     ' + str(error_sample_count[2]) + ' error samples, ' + str(error_type_count[2]) + ' errors;\n')

                print('Level '+str(i)+'  Missing Error:        ' + str(error_sample_count[3]) + ' error samples, ' + str(error_type_count[3]) + ' errors;')
                f.write('Level '+str(i)+'  Missing Error:        ' + str(error_sample_count[3]) + ' error samples, ' + str(error_type_count[3]) + ' errors;\n')

                print('Level '+str(i)+'  Unrecognized Error:   ' + str(error_sample_count[4]) + ' error samples, ' + str(error_type_count[4]) + ' errors;')
                f.write('Level '+str(i)+'  Unrecognized Error:   ' + str(error_sample_count[4]) + ' error samples, ' + str(error_type_count[4]) + ' errors;\n')
                print('\n')
                f.write('\n')

            # #################### draw png —— single model #################
            # 1. draw word_acc_visMatrix.png
            # 根据error_csv，统计正确识别的字
            word_num_list, word_num_list_error = resAna_wordACC(csv_path, char_list)
            word_acc = np.zeros(len(word_num_list))
            for i in range(len(word_num_list)):
                if word_num_list[i] == 0:
                    word_acc[i] = 1.0
                else:
                    word_acc[i] = 1 - word_num_list_error[i] / word_num_list[i]
            save_path = path.replace('.txt', '_word_acc_visMatrix.png')
            word_num_visMatrix(save_path, word_acc)

            # 2. draw count_cer_hist.png
            save_path = path.replace('.txt', '_count_cer_hist.png')
            cers_visHistchart(save_path, CERs)

            # 3. draw error_samples_rate_pie.png
            save_path = path.replace('.txt', '_error_samples_rate_pie.png')
            draw_error_samples_rate_pie(save_path, T_HSRs, title=['Error samples', 'Total samples'])

        f.close()


