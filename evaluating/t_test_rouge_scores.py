import matplotlib.pyplot as plt
from scipy.stats import ttest_rel


open_rag_f = [0.2459, 0.2385, 0.2608, 0.2545, 0.2475, 0.2406, 0.2381, 0.2470, 0.2374, 0.2441, 0.2378]
closed_rag_f = [0.2045, 0.2098, 0.1776, 0.1769, 0.1769, 0.2102, 0.2097, 0.2053, 0.2271, 0.2217, 0.2311]
open_rag_p = [0.2244, 0.2124, 0.2207, 0.2058, 0.1988, 0.2183, 0.2175, 0.2455, 0.2180, 0.2324, 0.2196]
closed_rag_p = [0.1845, 0.1853, 0.1319, 0.1278, 0.1276, 0.1825, 0.1811, 0.2103, 0.2238, 0.2117, 0.2109]
open_rag_r = [0.2720, 0.2720, 0.3188, 0.3335, 0.3279, 0.2679, 0.2630, 0.2485, 0.2605, 0.2571, 0.2593]
closed_rag_r = [0.2294, 0.2417, 0.2717, 0.2874, 0.2883, 0.2477, 0.2490, 0.2006, 0.2305, 0.2327, 0.2556]


plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.boxplot([open_rag_f, closed_rag_f], labels=['With RAG - F1', 'Without RAG - F1'], showmeans=True)
plt.title('ROUGE-L F1-Score Comparison', fontsize=18) 
plt.ylabel('F-Score', fontsize=18)
plt.xticks(fontsize=18)  
plt.yticks(fontsize=18)  
plt.grid(axis='y')


plt.subplot(3, 1, 2)
plt.boxplot([open_rag_p, closed_rag_p], labels=['With RAG - P', 'Without RAG - P'], showmeans=True)
plt.title('ROUGE-L Precision Comparison', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(axis='y')


plt.subplot(3, 1, 3)
plt.boxplot([open_rag_r, closed_rag_r], labels=['With RAG - R', 'Without RAG - R'], showmeans=True)
plt.title('ROUGE-L Recall Comparison', fontsize=18)
plt.ylabel('Recall', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(axis='y')


plt.tight_layout()
plt.show()

# T-test
t_stat, p_value = ttest_rel(open_rag_f, closed_rag_f)
print("t-statistic:", t_stat)
print("p-value:", p_value)
