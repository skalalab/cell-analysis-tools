import pandas as pd
import seaborn as sns
import scipy
import matplotlib.pylab as plt

combined = pd.read_csv("jcb_202001064_datas1.txt")
sns.set(style="whitegrid"); ReplicateAverages = combined.groupby(['Treatment','Replicate'], as_index=False).agg({'Speed': "mean"})
ReplicateAvePivot = ReplicateAverages.pivot_table(columns='Treatment', values='Speed', index="Replicate")

statistic, pvalue = scipy.stats.ttest_rel(ReplicateAvePivot['Control'], ReplicateAvePivot['Drug'])

P_value = str(float(round(pvalue, 3)))
sns.swarmplot(x="Treatment", y="Speed", hue="Replicate", data=combined)
ax = sns.swarmplot(x="Treatment", y="Speed", hue="Replicate", size=15, edgecolor="k", linewidth=2, data=ReplicateAverages)
ax.legend_. remove()
x1, x2 = 0, 1
y, h, col = combined['Speed'].max() + 2, 2, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h*2, "P = "+P_value, ha='center', va='bottom', color=col)