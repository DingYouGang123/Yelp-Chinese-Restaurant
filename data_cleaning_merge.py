import pandas as pd
import json
import os

business_path = "data/Yelp JSON/yelp_academic_dataset_business.json"
review_path = "data/Yelp JSON/yelp_academic_dataset_review.json"

output_merged = "data/100家中餐厅评论结果.csv"
output_biz = "data/100家中餐厅结果.csv"

# 自动创建输出目录
for path in [output_merged, output_biz]:
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ========== 1. 读取商家数据 ==========
print("正在读取商家数据...")
business_df = pd.read_json(business_path, lines=True, encoding='utf-8')

# ========== 2. 筛选符合条件的中餐厅 ==========
target_city = "Philadelphia"  

print("正在筛选符合条件的中餐厅...")
mask = (
    business_df['categories'].notna() &
    business_df['categories'].astype(str).str.contains(r'Chinese', case=False, na=False) &
    business_df['city'].notna() &
    (business_df['city'].astype(str).str.strip() == target_city) &
    (business_df['review_count'] > 100)
)

chinese_restaurants = business_df[mask].copy()

# 按评论数降序排序，取前100家
chinese_restaurants = chinese_restaurants.sort_values(by='review_count', ascending=False).head(100)

# 验证筛选结果
print(f"筛选出的符合条件的中餐厅数量：{len(chinese_restaurants)}")
if len(chinese_restaurants) < 100:
    print(f"警告：目标城市 {target_city} 符合条件的中餐厅不足100条，仅返回 {len(chinese_restaurants)} 条")

# 提取 business_id 并构建哈希集合
target_biz_ids = set(chinese_restaurants['business_id'].dropna())

# ========== 3. 逐行读取评论并筛选 ==========
print("正在逐行读取并筛选评论数据...")
matched_reviews = []
counter = 0

with open(review_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
            
        try:
            review = json.loads(line)
            biz_id = review.get('business_id')
            # 快速哈希判断
            if biz_id in target_biz_ids:
                matched_reviews.append({
                    'business_id': biz_id,
                    'review_id': review.get('review_id'),
                    'text': review.get('text'),
                    'stars': review.get('stars'),
                    'date': review.get('date')
                })
                counter += 1
        except json.JSONDecodeError:
            continue

print(f"成功匹配到 {counter} 条评论")

# 转换为 DataFrame
matched_reviews_df = pd.DataFrame(matched_reviews)

# ========== 4. 合并数据并保存 ==========
print("正在合并数据并保存...")
final_result = pd.merge(
    chinese_restaurants,
    matched_reviews_df,
    on='business_id',
    how='left'
)

# 保存结果 
final_result.to_csv(output_merged, index=False, encoding='utf-8-sig')
chinese_restaurants.to_csv(output_biz, index=False, encoding='utf-8-sig')

print(f"匹配到评论数：{len(matched_reviews_df)}")
print(f"结果已保存至：{output_merged}")
print(f"餐厅基础数据已保存至：{output_biz}")
print("成功")
