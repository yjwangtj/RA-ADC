from scipy.stats import ttest_rel
import matplotlib.pyplot as plt


open_rag = [0.6855, 0.6828, 0.6970, 0.7048, 0.7081, 0.6798, 0.6851, 0.6904, 0.6991, 0.6989, 0.7146]
closed_rag = [0.6873, 0.6882, 0.6702, 0.6753, 0.6730, 0.6848, 0.6835, 0.6731, 0.6968, 0.6921, 0.7024]

# 绘制箱线图
plt.figure(figsize=(8, 6))
plt.boxplot(
    [open_rag, closed_rag],
    labels=['With RAG', 'Without RAG'],
    showmeans=True
)
plt.title('Comparison of Cosine Similarity: With RAG vs Without RAG')
plt.ylabel('Cosine Similarity')
plt.grid(axis='y')

# 显示图表
plt.show()

t_stat, p_value = ttest_rel(open_rag, closed_rag)
print("t-statistic:", t_stat)
print("p-value:", p_value)