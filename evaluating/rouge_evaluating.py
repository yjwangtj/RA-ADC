import csv
from rouge import Rouge


rouge = Rouge()


with open('../answers/sorted_annotation.txt', 'r', encoding='utf-8') as f_ref:
    reference_lines = f_ref.readlines()

with open('../answers/internvl/without_rag_internvl-40b.txt', 'r', encoding='utf-8') as f_gen:
    generated_lines = f_gen.readlines()


rouge_1_results = []
rouge_2_results = []
rouge_l_results = []


rouge_1_f_sum = 0
rouge_1_p_sum = 0
rouge_1_r_sum = 0

rouge_2_f_sum = 0
rouge_2_p_sum = 0
rouge_2_r_sum = 0

rouge_l_f_sum = 0
rouge_l_p_sum = 0
rouge_l_r_sum = 0


for ref, gen in zip(reference_lines, generated_lines):
    ref = ref.strip() 
    gen = gen.strip()

   
    scores = rouge.get_scores([gen], [ref])[0]

  
    rouge_1 = scores['rouge-1']
    rouge_2 = scores['rouge-2']
    rouge_l = scores['rouge-l']


    rouge_1_results.append([rouge_1['f'], rouge_1['p'], rouge_1['r']])
    rouge_2_results.append([rouge_2['f'], rouge_2['p'], rouge_2['r']])
    rouge_l_results.append([rouge_l['f'], rouge_l['p'], rouge_l['r']])

 
    rouge_1_f_sum += rouge_1['f']
    rouge_1_p_sum += rouge_1['p']
    rouge_1_r_sum += rouge_1['r']

    rouge_2_f_sum += rouge_2['f']
    rouge_2_p_sum += rouge_2['p']
    rouge_2_r_sum += rouge_2['r']

    rouge_l_f_sum += rouge_l['f']
    rouge_l_p_sum += rouge_l['p']
    rouge_l_r_sum += rouge_l['r']


num_lines = len(reference_lines)

rouge_1_avg = [rouge_1_f_sum / num_lines, rouge_1_p_sum / num_lines, rouge_1_r_sum / num_lines]
rouge_2_avg = [rouge_2_f_sum / num_lines, rouge_2_p_sum / num_lines, rouge_2_r_sum / num_lines]
rouge_l_avg = [rouge_l_f_sum / num_lines, rouge_l_p_sum / num_lines, rouge_l_r_sum / num_lines]

with open('rouge_1.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['f', 'p', 'r'])
    writer.writerows(rouge_1_results)
    writer.writerow(['Average', *rouge_1_avg]) 


with open('rouge_2.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['f', 'p', 'r'])  
    writer.writerows(rouge_2_results)  
    writer.writerow(['Average', *rouge_2_avg])  


with open('rouge_l.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['f', 'p', 'r']) 
    writer.writerows(rouge_l_results)  
    writer.writerow(['Average', *rouge_l_avg])  


print("ROUGE-1 Average (f, p, r):", rouge_1_avg)
print("ROUGE-2 Average (f, p, r):", rouge_2_avg)
print("ROUGE-L Average (f, p, r):", rouge_l_avg)

print("ROUGE scores have been saved to 'rouge_1.csv', 'rouge_2.csv', and 'rouge_l.csv'.")
