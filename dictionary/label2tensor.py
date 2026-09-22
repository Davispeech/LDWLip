import os
import csv


def rm_space(data_path, csv_path):
    with open(csv_path, 'r') as f:
        contents = f.readlines()
        for line in contents:
            dataset_name, sample_path, frames, _ = line.split(',')

            f2_path = data_path + dataset_name + '/' + sample_path.replace('video', 'text')[:-4] + '.txt'

            with open(f2_path, 'r') as f2:
                label = f2.readline().replace(' ', '')
                f2.close()
            if os.path.exists(f2_path):
                os.remove(f2_path)
            with open(f2_path, 'w') as f2:
                f2.write(label)
                f2.close()


def get_token(label, char_list):
    token = ''
    now_token = ''
    for i in range(len(label)):
        for j in range(len(char_list)):
            if label[i] == char_list[j].replace('\n', ''):
                now_token = str(j) + ' '
                break
        if now_token:
            token = token + now_token
            now_token = ''
        else:
            token = token + '1 '  # 1 unknow
            now_token = ''
    # token = token + str(len(char_list)-1) + '\n'

    return token


def label_transfer(data_path, csv_path, char_list):
    if os.path.exists(csv_path[:-4]+'_new.csv'):
        os.remove(csv_path[:-4]+'_new.csv')
    f = open(csv_path[:-4]+'_new.csv', "w")
    with open(csv_path, 'r') as f1:
        contents = f1.readlines()
        for line in contents:
            dataset_name, sample_path, frames, _ = line.split(',')

            f2_path = data_path + dataset_name + '/' + sample_path.replace('video', 'text')[:-4] + '.txt'

            with open(f2_path, 'r') as f2:
                label = f2.readline().replace(' ', '')
                f2.close()

            # get token
            token = get_token(label, char_list)

            f.write(
                "{}\n".format(
                    f"{dataset_name},{sample_path},{frames},{token}"
                )
            )

        f.close()


data_path = '/home/ldw/Projects/Lipreading/CSLip/auto_avsr/'
csv_path = '/home/ldw/Projects/Lipreading/CSLip/auto_avsr/labels/cmlr/cmlr_train_transcript_lengths_seg24s.csv'

rm_space(data_path, csv_path)

dict_path = '/home/ldw/Projects/Lipreading/CSLip/auto_avsr/dictionary/zh-cn/cmlr_icslr.txt'
char_list = open(dict_path, 'r').readlines()

label_transfer(data_path, csv_path, char_list)
