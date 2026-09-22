import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用无GUI的后端
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import matplotlib.patches as patches
import matplotlib


def word_num_visMatrix(save_path, word_num_list):
    m = int(math.sqrt(len(word_num_list)/1.5))+1
    n = len(word_num_list)/m
    n = int(n+1) if int(n) < n else int(n)
    for i in range(m*n-len(word_num_list)):
        word_num_list = np.append(word_num_list, 0)
    matrix = word_num_list.reshape(m, n)
    vmin = np.min(matrix)
    vmax = np.max(matrix)

    colormap = cm.viridis
    plt.figure(figsize=(16, 8))  # 设置图形大小
    plt.imshow(matrix, interpolation='nearest', cmap=colormap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Word/Character Frequency')

    # 没有x轴，y轴刻度
    # plt.xticks(np.arange(matrix.shape[1]), labels=[f'{i + 1}' for i in range(matrix.shape[1])])
    # plt.yticks(np.arange(matrix.shape[0]), labels=[f'{i + 1}' for i in range(matrix.shape[0])])

    plt.title('the Statistics of Character Frequency')
    plt.savefig(save_path, dpi=300)


def sentenceLength_num_visColumnchart(save_path, sentenceLength_num_dict):
    sentenceLength_num_list = [0]
    now_max = 0

    for i,(key, value) in enumerate(sentenceLength_num_dict.items()):
        if int(key) > now_max:
            for j in range(now_max+1, int(key)+1):
                sentenceLength_num_list.append(0)
                now_max = int(key)
        sentenceLength_num_list[int(key)] = int(value)
    for i in range(len(sentenceLength_num_list)):
        now_min = i
        if sentenceLength_num_list[i] > 0:
            break
    sentenceLength_num_list = sentenceLength_num_list[now_min:]

    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.size'] = 24  # 设置全局字体大小为18

    plt.figure(figsize=(12, 6))  # 注意：这应该在绘图命令之前调用
    plt.subplots_adjust(left=0.1, right=0.99)
    plt.subplots_adjust(bottom=0.13, top=0.95)

    x = np.arange(now_min, now_max+1, 1)
    y = sentenceLength_num_list
    colors = (90 / 255.0, 150 / 255.0, 210 / 255.0)

    # 创建条形图
    plt.bar(x, y, width=0.3, color=colors)

    # 设置标签和标题
    plt.xlabel('Corpus length', fontsize=24)
    plt.ylabel('Number of samples', fontsize=24)

    y_min = min(sentenceLength_num_list)
    y_max = max(sentenceLength_num_list)
    if int((y_max-y_min)/8) > y_min:
        y_start = y_min
    else:
        y_start = 0
    # 设置y轴刻度和网格
    plt.yticks(np.arange(y_start, y_start+9*int((y_max-y_min)/8), int((y_max-y_min)/8)))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 设置x轴刻度标签旋转
    plt.xticks(np.arange(now_min, now_max+1, 1))

    # 设置x,y轴范围
    plt.ylim(y_start, y_start+9*int((y_max-y_min)/8))

    # 保存图像为PNG格式，分辨率为300 dpi
    plt.savefig(save_path, dpi=300)


def frames_visHistchart(save_path, frames_list):

    plt.figure(figsize=(12, 6))

    plt.hist(frames_list, bins=30, color='skyblue', edgecolor='black')

    # 添加标题和标签
    plt.title('seconds')
    plt.xlabel('video frames')
    plt.ylabel('frequency')

    # 保存图像为PNG格式，分辨率为300 dpi
    plt.savefig(save_path)
    # plt.show()


def cers_visHistchart(save_path, cers_list):
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = 20  # 设置全局字体大小为18

    plt.figure(figsize=(12, 6))

    plt.hist(cers_list, bins=200, color='skyblue', edgecolor='black')

    # 添加标题和标签
    # plt.title('seconds')
    plt.xlabel('CERs')
    plt.ylabel('Frequency')

    # 保存图像为PNG格式，分辨率为300 dpi
    plt.savefig(save_path)
    # plt.show()


def draw_error_samples_rate_pie(save_path, rate, title=['Error samples', 'Total samples']):
    # matplotlib.rcParams['font.family'] = 'Times New Roman'
    # matplotlib.rcParams['font.size'] = 16  # 设置全局字体大小为18

    # 示例数据：每个饼图的比例
    # data1 = [18.6, 81.4]  # 100% - 18.6% = 81.4%
    # data2 = [36.9, 63.1]  # 100% - 36.9% = 63.1%
    # data3 = [59.6, 40.4]  # 100% - 59.6% = 40.4%
    # data4 = [50.4, 49.6]  # 100% - 50.4% = 49.6%
    # data5 = [40.4, 59.6]
    datas = []
    for i in range(len(rate)):
        a = float(str(rate[i])[:6])*100
        b = 100 - a
        datas.append([a, b])
    data1, data2, data3, data4, data5 = datas

    # 标签
    labels = [title[0], title[1]]

    # 设置子图布局为1行4列，并调整图形大小
    fig, axes = plt.subplots(1, 5, figsize=(14, 4))
    plt.subplots_adjust(left=0.0, right=0.99)
    plt.subplots_adjust(bottom=0.1, top=0.85)

    # 绘制饼图
    colors = [(240 / 255.0, 128 / 255.0, 127 / 255.0), (135 / 255.0, 206 / 255.0, 250 / 255.0)]  # 你可以自定义颜色
    titles = ['(a)', '(b)', '(c)', '(d)', '(e)']
    explode = (0, 0.1)
    for ax, data, title in zip(axes, [data1, data2, data3, data4, data5], titles):
        # ax.pie(data, explode=explode, labels=None, autopct='%1.1f%%', startangle=140, colors=colors)
        ax.pie(data, explode=explode, labels=None, colors=colors, autopct='%1.1f%%', shadow=False, startangle=140)
        ax.axis('equal')  # 确保饼图是圆形的
        ax.set_title(title, y=-0.1)  # 设置子图标题

    # 隐藏子图的x轴和y轴标签
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    legend_patches = [patches.Patch(color=colors[0], label=labels[0]),
                      patches.Patch(color=colors[1], label=labels[1])]
    # 在图形外部添加自定义的总图例
    # 这里我们选择在图形的右侧添加图例，但你可以根据需要调整位置
    legend = plt.legend(handles=legend_patches, ncol=2, loc='upper right', bbox_to_anchor=(-0.5, 1.2), frameon=True)

    # 设置图形标题（可选）
    # plt.suptitle('Pie Charts with Specific Percentages')

    # 显示图形
    # plt.tight_layout()  # 调整子图之间的间距
    plt.savefig(save_path, dpi=300)
    # plt.show()

