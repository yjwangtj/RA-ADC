import csv
from rouge import Rouge

# 初始化ROUGE对象
rouge = Rouge()

# 读取参考文本和生成文本
with open('../answers/sorted_annotation.txt', 'r', encoding='utf-8') as f_ref:
    reference_lines = f_ref.readlines()

with open('../answers/internvl/without_rag_internvl-40b.txt', 'r', encoding='utf-8') as f_gen:
    generated_lines = f_gen.readlines()

# 存储 ROUGE 的结果
rouge_1_results = []
rouge_2_results = []
rouge_l_results = []

# 累积 F, P, R 值以计算平均值
rouge_1_f_sum = 0
rouge_1_p_sum = 0
rouge_1_r_sum = 0

rouge_2_f_sum = 0
rouge_2_p_sum = 0
rouge_2_r_sum = 0

rouge_l_f_sum = 0
rouge_l_p_sum = 0
rouge_l_r_sum = 0

# 逐行计算 ROUGE 分数
for ref, gen in zip(reference_lines, generated_lines):
    ref = ref.strip()  # 去除末尾的换行符
    gen = gen.strip()

    # 计算 ROUGE 分数
    scores = rouge.get_scores([gen], [ref])[0]

    # 提取ROUGE-1, ROUGE-2, ROUGE-L的F, P, R值
    rouge_1 = scores['rouge-1']
    rouge_2 = scores['rouge-2']
    rouge_l = scores['rouge-l']

    # 将结果添加到对应的列表中
    rouge_1_results.append([rouge_1['f'], rouge_1['p'], rouge_1['r']])
    rouge_2_results.append([rouge_2['f'], rouge_2['p'], rouge_2['r']])
    rouge_l_results.append([rouge_l['f'], rouge_l['p'], rouge_l['r']])

    # 累积 F, P, R 值
    rouge_1_f_sum += rouge_1['f']
    rouge_1_p_sum += rouge_1['p']
    rouge_1_r_sum += rouge_1['r']

    rouge_2_f_sum += rouge_2['f']
    rouge_2_p_sum += rouge_2['p']
    rouge_2_r_sum += rouge_2['r']

    rouge_l_f_sum += rouge_l['f']
    rouge_l_p_sum += rouge_l['p']
    rouge_l_r_sum += rouge_l['r']

# 计算平均值
num_lines = len(reference_lines)

rouge_1_avg = [rouge_1_f_sum / num_lines, rouge_1_p_sum / num_lines, rouge_1_r_sum / num_lines]
rouge_2_avg = [rouge_2_f_sum / num_lines, rouge_2_p_sum / num_lines, rouge_2_r_sum / num_lines]
rouge_l_avg = [rouge_l_f_sum / num_lines, rouge_l_p_sum / num_lines, rouge_l_r_sum / num_lines]

# 将 ROUGE-1 结果保存到CSV文件
with open('rouge_1.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['f', 'p', 'r'])  # 写入表头
    writer.writerows(rouge_1_results)  # 写入数据
    writer.writerow(['Average', *rouge_1_avg])  # 写入平均值

# 将 ROUGE-2 结果保存到CSV文件
with open('rouge_2.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['f', 'p', 'r'])  # 写入表头
    writer.writerows(rouge_2_results)  # 写入数据
    writer.writerow(['Average', *rouge_2_avg])  # 写入平均值

# 将 ROUGE-L 结果保存到CSV文件
with open('rouge_l.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['f', 'p', 'r'])  # 写入表头
    writer.writerows(rouge_l_results)  # 写入数据
    writer.writerow(['Average', *rouge_l_avg])  # 写入平均值

# 打印出每个文件的平均值
print("ROUGE-1 Average (f, p, r):", rouge_1_avg)
print("ROUGE-2 Average (f, p, r):", rouge_2_avg)
print("ROUGE-L Average (f, p, r):", rouge_l_avg)

print("ROUGE scores have been saved to 'rouge_1.csv', 'rouge_2.csv', and 'rouge_l.csv'.")