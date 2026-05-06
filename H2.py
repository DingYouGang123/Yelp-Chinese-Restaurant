import pandas as pd
import numpy as np
import os
import warnings
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import accelerate

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False

# ========== 路径配置 ==========
INPUT_EXCEL = "data/100家中餐厅评论结果（精简版）.xlsx"
BASE_RESTAURANTS = "data/100家中餐厅结果.csv"
OUTPUT_SENTIMENT = "data/情感分析结果.xlsx"
OUTPUT_FINAL = "data/情感结果.xlsx"


# ===================== 第一步：加载数据 & 情感分析 =====================
def preprocess(text):
    """
    预处理文本：替换用户名为 @user，链接为 http
    """
    if not isinstance(text, str):
        return ""
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# 初始化模型和 tokenizer 
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()  # 设置为评估模式

# 确定设备 (如果有GPU则使用GPU，否则CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

def get_sentiment_score(text):
    """
    输入文本，返回 positive_prob - negative_prob 作为情感分数
    范围在 -1 到 1 之间
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    # 1. 预处理
    text = preprocess(text)
    
    # 2. 编码
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    
    # 将输入移动到相同设备
    encoded_input = {key: val.to(device) for key, val in encoded_input.items()}
    
    # 3. 推理
    with torch.no_grad():
        output = model(**encoded_input)
    
    # 4. 计算 softmax 概率
    scores = output[0][0].detach().cpu().numpy()
    scores = softmax(scores)
    
    # 5. 获取标签对应的索引
    label_mapping = config.id2label
    
    neg_idx = None
    pos_idx = None
    
    for idx, label in label_mapping.items():
        if label == 'negative':
            neg_idx = idx
        elif label == 'positive':
            pos_idx = idx
            
    sentiment_score = scores[pos_idx] - scores[neg_idx]    
    return float(sentiment_score)


print("正在加载评论数据...")
data = pd.read_excel(INPUT_EXCEL)

# 检查必要的列是否存在
if 'text' not in data.columns:
    raise ValueError("Excel文件中未找到 'text' 列")
if 'stars.y' not in data.columns:
    raise ValueError("Excel文件中未找到 'stars.y' 列")
    
print(f"Data loaded. Shape: {data.shape}")

# --- 生成 sentiment_score ---
print("Calculating sentiment scores using RoBERTa...")
data['sentiment_score'] = data['text'].apply(get_sentiment_score)

print(f"处理完成！\n原始数据总行数： {len(data)}")
print(f"情感得分列非NA数量： {data['sentiment_score'].notna().sum()}")

# ===================== 第二步：计算情感落差 =====================
# 自动适配 stars.y 或 stars 列名
star_col = 'stars.y' if 'stars.y' in data.columns else 'stars'
data['rating_scaled'] = (data[star_col] - 3) / 2
data['expectation_gap'] = data['sentiment_score'] - data['rating_scaled']

print("\n预览核心字段：")
print(data[['text', star_col, 'sentiment_score', 'rating_scaled', 'expectation_gap']].head())
data.to_excel(OUTPUT_SENTIMENT, index=False)
print(f"情感分析结果已保存至：{OUTPUT_SENTIMENT}")

# ===================== 第三步：按餐厅聚合 & 匹配 =====================
data2 = pd.read_excel(OUTPUT_SENTIMENT)
restaurant_avg = data2.groupby('name', as_index=False).agg(
    avg_expectation_gap=('expectation_gap', 'mean'),
    avg_stars=(star_col, 'mean')
)
print(f"\n共计算了 {len(restaurant_avg)} 家餐厅的均值")

chinese_restaurants = pd.read_csv(BASE_RESTAURANTS)
# 左连接匹配均值（处理可能存在的后缀冲突）
chinese_restaurants = chinese_restaurants.merge(restaurant_avg, on='name', how='left', suffixes=('', '_merged'))
# 清理列名
for col in ['avg_expectation_gap', 'avg_stars']:
    if f"{col}_merged" in chinese_restaurants.columns:
        chinese_restaurants[col] = chinese_restaurants[f"{col}_merged"]
        chinese_restaurants.drop(columns=[f"{col}_merged"], inplace=True)

print("餐厅数据框新增列：avg_expectation_gap（情感落差均值）、avg_stars（评分均值）")
print(chinese_restaurants[['name', 'avg_expectation_gap', 'avg_stars']].head())
chinese_restaurants.to_excel(OUTPUT_FINAL, index=False)
print(f"最终结果已保存至：{OUTPUT_FINAL}")

# ===================== 第四步：回归数据预处理 =====================
# 安全选择列（自动处理 .x/.y 后缀问题）
def safe_select(df, target_name, *candidates):
    for c in candidates:
        if c in df.columns:
            return df[c]
    raise KeyError(f"找不到列：{target_name}，候选：{candidates}")

reg_data = pd.DataFrame({
    'Y': safe_select(chinese_restaurants, 'avg_stars', 'avg_stars', 'avg_stars.x'),
    'X1': safe_select(chinese_restaurants, 'attributes.RestaurantsPriceRange2', 'attributes.RestaurantsPriceRange2'),
    'X2': safe_select(chinese_restaurants, 'avg_expectation_gap', 'avg_expectation_gap', 'avg_expectation_gap.x')
}).dropna()

print(f"\n用于回归分析的有效样本数： {len(reg_data)}")
print(reg_data.describe())

# ===================== 第五步：多元线性回归 =====================
X = sm.add_constant(reg_data[['X1', 'X2']])
y = reg_data['Y']
reg_model = sm.OLS(y, X).fit()

print("\n===== 回归分析结果 =====")
print(reg_model.summary())

r2 = reg_model.rsquared
adj_r2 = reg_model.rsquared_adj
f_pvalue = reg_model.f_pvalue
print(f"\n===== 核心指标总结 =====")
print(f"R²（决定系数）： {r2:.4f} → 模型解释了 {r2*100:.2f}%的评分变异")
print(f"调整后R²： {adj_r2:.4f} → 更准确的模型拟合度（考虑自变量数量）")
print(f"F检验p值： {f_pvalue:.4f} → 若 <0.05，模型整体显著")

print("\n===== 系数表 =====")
coef_table = pd.DataFrame({
    '系数': reg_model.params,
    '标准误': reg_model.bse,
    't值': reg_model.tvalues,
    'p值': reg_model.pvalues
}).round(4)
print(coef_table)

# ===================== 第六步：回归诊断 =====================
residuals = reg_model.resid
shapiro_stat, shapiro_p = shapiro(residuals)
print(f"\n残差正态性检验p值： {shapiro_p:.4f} → 若 >0.05，残差符合正态分布")

# VIF 检验（仅对自变量计算）
X_vif = X.iloc[:, 1:]
vif_data = pd.DataFrame({
    'Variable': X_vif.columns,
    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
print("\n多重共线性检验（VIF）：")
print(vif_data.round(2))

# ===================== 第七步：可视化 & 单变量回归 =====================
plt.figure(figsize=(12, 5))
# 图1：价格 vs 评分
plt.subplot(1, 2, 1)
sns.scatterplot(data=reg_data, x='X1', y='Y', alpha=0.6)
sns.regplot(data=reg_data, x='X1', y='Y', scatter=False, color='red', ci=95)
plt.title('价格与商店评分的关系')
plt.xlabel('价格（X1）')
plt.ylabel('商店评分（Y）')

# 图2：情感落差 vs 评分
plt.subplot(1, 2, 2)
sns.scatterplot(data=reg_data, x='X2', y='Y', alpha=0.6, color='black')
sns.regplot(data=reg_data, x='X2', y='Y', scatter=False, color='lightblue', ci=95)
plt.title('Expectation_gap vs. Average Rating')
plt.xlabel('Expectation_gap')
plt.ylabel('Average Rating')
plt.tight_layout()
plt.show()

# 简单一元回归 Y ~ X1
model_simple = sm.OLS(y, sm.add_constant(reg_data[['X1']])).fit()
print("\n===== 单变量回归 (Y ~ X1) =====")
print(model_simple.summary())

print("\n✅ 全部流程执行完毕。")