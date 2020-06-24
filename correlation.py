import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import pymongo

# Connect to mongodb and make collection value
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["final"]
col_sales = mydb["sales"]
col_fpop = mydb["fpop"]

# Define queries to retrieve data from each collection
service_codes = col_sales.distinct("서비스_업종_코드_명")
fpop_query = [{"$group" : {"_id" : "$상권_코드_명", "avg_floatingPop" : {"$avg" : "$총_유동인구_수"},
                           "avg_female" : {"$avg" : "$여성_유동인구_수"}, "avg_age10" : {"$avg" : "$연령대_10_유동인구_수"},
                           "avg_age20" : {"$avg" : "$연령대_20_유동인구_수"}, "avg_age30" : {"$avg" : "$연령대_30_유동인구_수"},
                           "avg_age40" : {"$avg" : "$연령대_40_유동인구_수"}, "avg_age50" : {"$avg" : "$연령대_50_유동인구_수"},
                           "avg_age60" : {"$avg" : "$연령대_60_이상_유동인구_수"}}}]
compare_targets = ['총 유동인구 수', '유동인구 성비율', '10대 유동인구', '20대 유동인구', '30대 유동인구', '40대 유동인구', '50대 유동인구', '60대 유동인구']

# Correlate factors with sales for each service code
total_corr = np.array([])
total_corr_list = []
for service_code in service_codes:
    sales_query = [{"$match" : {"서비스_업종_코드_명": service_code}},
                   {"$group" : {"_id" : "$상권_코드_명", "avg_sales" : {"$avg" : "$당월_매출_금액"}}}]

    training_label_id = []
    training_label = []
    training_set = [np.array([]) for row in range(9)]
    for item in col_sales.aggregate(sales_query):
        training_label_id.append(item["_id"])
        training_label.append(item["avg_sales"])
    for item in col_fpop.aggregate(fpop_query):
        if item["_id"] in training_label_id:
            label_idx = training_label_id.index(item["_id"])
            training_set[0] = np.append(training_set[0], item["avg_floatingPop"])
            training_set[1] = np.append(training_set[1], item["avg_female"] / item["avg_floatingPop"])
            training_set[2] = np.append(training_set[2], item["avg_age10"])
            training_set[3] = np.append(training_set[3], item["avg_age20"])
            training_set[4] = np.append(training_set[4], item["avg_age30"])
            training_set[5] = np.append(training_set[5], item["avg_age40"])
            training_set[6] = np.append(training_set[6], item["avg_age50"])
            training_set[7] = np.append(training_set[7], item["avg_age60"])
            training_set[8] = np.append(training_set[8], training_label[label_idx])
    df = pd.DataFrame(training_set).T
    corr = np.array(df.corrwith(df[8])[:8])
    total_corr = np.append(total_corr, corr)
    corr_list = corr.tolist()
    total_corr_list.append(corr_list)

# Select the top 10 most correlated factors
top10val = np.sort(total_corr)[:-11:-1]
top10idx = np.argsort(total_corr)[:-11:-1]

print('Top10 highly correlated factor with sales in fpop datasets')
i = 1
for idx in top10idx:
    #print("service code index = ", int(idx / 8), ", compare target = ", idx % 8)
    print('#', i, ': ', service_codes[int(idx / 8)], ", ", compare_targets[idx % 8])
    i = i + 1

# Visualize the entire correlation map as heatmap of seaborn
df2 = pd.DataFrame(total_corr_list)
ax = sns.heatmap(df2)
plt.title('heatmap')
plt.show()