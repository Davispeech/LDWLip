import os


def search_unkown(data_path, csv_path, char_list):
    unkown = ''
    with open(csv_path, 'r') as f:
        contents = f.readlines()
        for line in contents:
            dataset_name, sample_path, frames, _ = line.split(',')

            f2_path = data_path + dataset_name + '/' + sample_path.replace('video', 'text')[:-4] + '.txt'

            with open(f2_path, 'r') as f2:
                label = f2.readline().replace(' ', '')
                f2.close()

            for i in range(len(label)):
                now_token = False
                for j in range(len(char_list)):
                    if label[i] == char_list[j].replace('\n', ''):
                        now_token = True
                        break
                    if j == len(char_list)-1 and not now_token:
                        if label[i] not in unkown:
                            unkown = unkown + label[i]
                            char_list.append(label[i] + '\n')
        f.close()
    return char_list, unkown

'''
data_path = '/home/ldw/Projects/Lipreading/CSLip/auto_avsr/'
csv_path = '/home/ldw/Projects/Lipreading/CSLip/auto_avsr/labels/icslr/icslr_test_transcript_lengths_seg24s.csv'

dict_path = '/home/ldw/Projects/Lipreading/CSLip/auto_avsr/dictionary/zh-cn/mpcVSR-zh_cn.txt'
char_list = open(dict_path, 'r').readlines()

unkown = rm_space(data_path, csv_path, char_list)
print(unkown)
'''


def cre_mpcVSR_zh_cn():
    contents = open('/home/ldw/Projects/Lipreading/CSLip/auto_avsr/dictionary/zh-cn/mpcVSR-zh_cn.txt', 'r').readlines()
    char_list = []
    for line in contents:
        if line.replace('\n', '') != '<eos>':
            char_list.append(line.split(' ')[-1])

    data_path = '/home/ldw/Projects/Lipreading/CSLip/auto_avsr/'
    csv_path = '/home/ldw/Projects/Lipreading/CSLip/auto_avsr/labels/icslr/icslr_test_transcript_lengths_seg24s.csv'
    char_list, unkown = search_unkown(data_path, csv_path, char_list)
    char_list.append(contents[-1])
    print(char_list, len(char_list), unkown)
    with open('/home/ldw/Projects/Lipreading/CSLip/auto_avsr/dictionary/zh-cn/cmlr_icslr.txt', 'w') as f:
        for char_ in char_list:
            f.write(char_)
        f.close()


def re_icslr_zh_cn():
    char_list = []
    cts = open('/home/ldw/Projects/Lipreading/CSLip/auto_avsr/dictionary/zh-cn/mpcVSR-zh_cn.txt', 'r').readlines()
    char_list.append(cts[0].split(' ')[-1])
    char_list.append(cts[1].split(' ')[-1])

    contents = open('/home/ldw/Projects/Lipreading/CSLip/auto_avsr/dictionary/zh-cn/self_dataset_vocab_map.txt', 'r').readlines()
    for i in range(4, len(contents)):
        line = contents[i]
        if line.replace('\n', '') != '<eos>':
            char_list.append(line.split(' ')[-1])

    data_path = '/home/ldw/Projects/Lipreading/CSLip/auto_avsr/'
    csv_path = '/home/ldw/Projects/Lipreading/CSLip/auto_avsr/labels/icslr/icslr_test_transcript_lengths_seg24s.csv'
    char_list, unkown = search_unkown(data_path, csv_path, char_list)
    char_list.append(cts[-1])
    print(char_list, len(char_list), unkown)
    with open('/home/ldw/Projects/Lipreading/CSLip/auto_avsr/dictionary/zh-cn/icslr_zhcn_new.txt', 'w') as f:
        for char_ in char_list:
            f.write(char_)
        f.close()


re_icslr_zh_cn()
