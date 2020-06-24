from flask import Flask, render_template, request, jsonify
from sklearn.neighbors import KNeighborsClassifier
import pymongo
import matplotlib.pyplot as plt
from pyproj import Transformer
import numpy as np
import random
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from scipy import stats

# connect to mongodb and make collection value
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["final"]
col_area = mydb["area"]
col_sales = mydb["sales"]
col_fpop = mydb["fpop"]
col_resident = mydb["resident"]
col_worker = mydb["worker"]
col_facility = mydb["facility"]

# Define training data, label
training_data = []
training_label = []

# Store training data, label
for item in col_area.find():
    line = []
    line.append(item["엑스좌표_값"])
    line.append(item["와이좌표_값"])

    training_data.append(line)
    training_label.append(item["상권_코드_명"])

# Make KNN Model & fit
classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(training_data, training_label)

# Define coordinate EPSG: 5181 to WSG84
transformer = Transformer.from_proj(
    "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
    "+proj=tmerc +lat_0=38 +lon_0=127 +k=1 +x_0=200000 +y_0=500000 +ellps=GRS80 +units=m +no_defs",
    always_xy=True,
    skip_equivalent=True
)

# Start flask app
app = Flask(__name__)

# index page
@app.route('/')
def index():
    return render_template('gmap.html')

# Response coordinate
@app.route('/call', methods=['GET'])
def call_addr():
    addr = request.args.get("addr")
    new_addr = addr[1:-1]
    split_str = new_addr.split(', ', 1)

    # Split x, y
    x = float(split_str[1])
    y = float(split_str[0])

    # Transform EPSG:5181 to WSG84
    trans_x, trans_y = transformer.transform(x, y)

    unknown_data = [
        [trans_x, trans_y]
    ]

    # Predict where unknown data is
    guesses = classifier.predict(unknown_data)

    guesses_str = np.array_str(guesses)
    str = guesses_str[2:-2]

    return str


# Response for service code name
@app.route("/init")
def init():
    data = []

    # Store all service code name
    for items in col_sales.distinct("서비스_업종_코드_명"):
        data.append(items)

    # Send all service code name
    return jsonify(data)


# Response analysis result
@app.route("/service", methods=['GET'])
def service():
    area_code = request.args.get("code_name")
    service_code = request.args.get("service_name")

    # Define queries to retrieve data from each collection
    fpop_query = [
        {"$group": {"_id": "$상권_코드_명", "avg_floatingPop": {"$avg": "$총_유동인구_수"}, "avg_female": {"$avg": "$여성_유동인구_수"},
                    "avg_age10": {"$avg": "$연령대_10_유동인구_수"}, "avg_age20": {"$avg": "$연령대_20_유동인구_수"},
                    "avg_age30": {"$avg": "$연령대_30_유동인구_수"}, "avg_age40": {"$avg": "$연령대_40_유동인구_수"},
                    "avg_age50": {"$avg": "$연령대_50_유동인구_수"}, "avg_age60": {"$avg": "$연령대_60_이상_유동인구_수"}}},
        {"$sort": {"_id": 1}}]
    resident_query = [
        {"$group": {"_id": "$상권 코드 명", "avg_resident": {"$avg": "$총 상주인구 수"}, "avg_female": {"$avg": "$여성 상주인구 수"},
                    "avg_age10": {"$avg": "$연령대 10 상주인구 수"}, "avg_age20": {"$avg": "$연령대 20 상주인구 수"},
                    "avg_age30": {"$avg": "$연령대 30 상주인구 수"}, "avg_age40": {"$avg": "$연령대 40 상주인구 수"},
                    "avg_age50": {"$avg": "$연령대 50 상주인구 수"}, "avg_age60": {"$avg": "$연령대 60 이상 상주인구 수"},
                    "avg_household": {"$avg": "$총 가구 수"}}},
        {"$sort": {"_id": 1}}]
    worker_query = [
        {"$group": {"_id": "$상권_코드_명", "avg_worker": {"$avg": "$총_직장_인구_수"}, "avg_female": {"$avg": "$여성_직장_인구_수"},
                    "avg_age10": {"$avg": "$연령대_10_직장_인구_수"}, "avg_age20": {"$avg": "$연령대_20_직장_인구_수"},
                    "avg_age30": {"$avg": "$연령대_30_직장_인구_수"}, "avg_age40": {"$avg": "$연령대_40_직장_인구_수"},
                    "avg_age50": {"$avg": "$연령대_50_직장_인구_수"}, "avg_age60": {"$avg": "$연령대_60_이상_직장_인구_수"}}},
        {"$sort": {"_id": 1}}]
    facility_query = [{"$group": {"_id": "$상권_코드_명", "avg_facility": {"$avg": "$집객시설_수"},
                                  "avg_government_offices": {"$avg": "$관공서_수"},
                                  "avg_bank": {"$avg": "$은행_수"}, "avg_hospital": {"$avg": "$종합병원_수"},
                                  "avg_clinic": {"$avg": "$일반_병원_수"},
                                  "avg_pharmacy": {"$avg": "$약국_수"}, "avg_kindergarten": {"$avg": "$유치원_수"},
                                  "avg_elementary": {"$avg": "$초등학교_수"},
                                  "avg_middle": {"$avg": "$중학교_수"}, "avg_high": {"$avg": "$고등학교_수"},
                                  "avg_university": {"$avg": "$대학교_수"},
                                  "avg_department_store": {"$avg": "$백화점_수"}, "avg_supermarket": {"$avg": "$슈퍼마켓_수"},
                                  "avg_theater": {"$avg": "$극장_수"},
                                  "avg_accommodation": {"$avg": "$숙박_시설_수"}, "avg_railway": {"$avg": "$철도_역_수"},
                                  "avg_terminal": {"$avg": "$버스_터미널_수"},
                                  "avg_subway": {"$avg": "$지하철_역_수"}, "avg_bus_station": {"$avg": "$버스_정거장_수"}}},
                      {"$sort": {"_id": 1}}]
    sales_query = [{"$match": {"서비스_업종_코드_명": service_code}},
                   {"$group": {"_id": "$상권_코드_명", "avg_sales": {"$avg": "$당월_매출_금액"}, "avg_store": {"$avg": "$점포수"}}},
                   {"$sort": {"_id": 1}}]

    # Define the factors used to predict sales
    fpop_factors = ['총 유동인구 수', '유동인구 성비', '10대 유동인구', '20대 유동인구', '30대 유동인구', '40대 유동인구', '50대 유동인구', '60대 이상 유동인구']
    resident_factors = ['총 상주인구 수', '상주인구 성비', '10대 상주인구', '20대 상주인구', '30대 상주인구', '40대 상주인구', '50대 상주인구',
                        '60대 이상 상주인구', '총 가구 수']
    worker_factors = ['총 직장인구 수', '직장인구 성비', '10대 직장인구', '20대 직장인구', '30대 직장인구', '40대 직장인구', '50대 직장인구', '60대 이상 직장인구']
    facility_factors = ['집객시설 수', '관공서 수', '은행 수', '종합병원 수', '일반 병원 수', '약국 수', '유치원 수', '초등학교 수', '중학교 수', '고등학교 수',
                        '대학교 수',
                        '백화점 수', '슈퍼마켓 수', '극장 수', '숙박 시설 수', '철도 역 수', '버스 터미널 수', '지하철 역 수', '버스 정거장 수']
    compare_factors = fpop_factors + resident_factors + worker_factors + facility_factors
    fpop_factors_num = len(fpop_factors)
    resident_factors_num = len(resident_factors)
    worker_factors_num = len(worker_factors)
    facility_factors_num = len(facility_factors)
    compare_factors_num = len(compare_factors)

    # Save values from queries as arrays
    train_fpop = [[] for row in range(fpop_factors_num + 1)]
    for item in col_fpop.aggregate(fpop_query):
        train_fpop[0].append(item["avg_floatingPop"])
        train_fpop[1].append(item["avg_female"] / item["avg_floatingPop"])
        train_fpop[2].append(item["avg_age10"])
        train_fpop[3].append(item["avg_age20"])
        train_fpop[4].append(item["avg_age30"])
        train_fpop[5].append(item["avg_age40"])
        train_fpop[6].append(item["avg_age50"])
        train_fpop[7].append(item["avg_age60"])
        train_fpop[8].append(item["_id"])
    train_resident = [[] for row in range(resident_factors_num + 1)]
    for item in col_resident.aggregate(resident_query):
        train_resident[0].append(item["avg_resident"])
        train_resident[1].append(item["avg_female"] / item["avg_resident"])
        train_resident[2].append(item["avg_age10"])
        train_resident[3].append(item["avg_age20"])
        train_resident[4].append(item["avg_age30"])
        train_resident[5].append(item["avg_age40"])
        train_resident[6].append(item["avg_age50"])
        train_resident[7].append(item["avg_age60"])
        train_resident[8].append(item["avg_household"])
        train_resident[9].append(item["_id"])
    train_worker = [[] for row in range(worker_factors_num + 1)]
    for item in col_worker.aggregate(worker_query):
        train_worker[0].append(item["avg_worker"])
        train_worker[1].append(item["avg_female"] / item["avg_worker"])
        train_worker[2].append(item["avg_age10"])
        train_worker[3].append(item["avg_age20"])
        train_worker[4].append(item["avg_age30"])
        train_worker[5].append(item["avg_age40"])
        train_worker[6].append(item["avg_age50"])
        train_worker[7].append(item["avg_age60"])
        train_worker[8].append(item["_id"])
    train_facility = [[] for row in range(facility_factors_num + 1)]
    for item in col_facility.aggregate(facility_query):
        train_facility[0].append(item["avg_facility"])
        train_facility[1].append(item["avg_government_offices"])
        train_facility[2].append(item["avg_bank"])
        train_facility[3].append(item["avg_hospital"])
        train_facility[4].append(item["avg_clinic"])
        train_facility[5].append(item["avg_pharmacy"])
        train_facility[6].append(item["avg_kindergarten"])
        train_facility[7].append(item["avg_elementary"])
        train_facility[8].append(item["avg_middle"])
        train_facility[9].append(item["avg_high"])
        train_facility[10].append(item["avg_university"])
        train_facility[11].append(item["avg_department_store"])
        train_facility[12].append(item["avg_supermarket"])
        train_facility[13].append(item["avg_theater"])
        train_facility[14].append(item["avg_accommodation"])
        train_facility[15].append(item["avg_railway"])
        train_facility[16].append(item["avg_terminal"])
        train_facility[17].append(item["avg_subway"])
        train_facility[18].append(item["avg_bus_station"])
        train_facility[19].append(item["_id"])

    # make x and y training data
    train_label_id = []
    train_label = []
    stores_num = []
    train_set = [np.array([]) for row in range(compare_factors_num)]
    for item in col_sales.aggregate(sales_query):
        isValErr = False
        if not item["_id"] in train_fpop[fpop_factors_num]: isValErr = True
        if not item["_id"] in train_resident[resident_factors_num]: isValErr = True
        if not item["_id"] in train_worker[worker_factors_num]: isValErr = True
        if not item["_id"] in train_facility[facility_factors_num]: isValErr = True
        if isValErr: continue

        fpop_idx = train_fpop[fpop_factors_num].index(item["_id"])
        resident_idx = train_resident[resident_factors_num].index(item["_id"])
        worker_idx = train_worker[worker_factors_num].index(item["_id"])
        facility_idx = train_facility[facility_factors_num].index(item["_id"])
        train_idx = 0
        for i in range(fpop_factors_num):
            train_set[train_idx] = np.append(train_set[train_idx], train_fpop[i][fpop_idx])
            train_idx = train_idx + 1
        for i in range(resident_factors_num):
            train_set[train_idx] = np.append(train_set[train_idx], train_resident[i][resident_idx])
            train_idx = train_idx + 1
        for i in range(worker_factors_num):
            train_set[train_idx] = np.append(train_set[train_idx], train_worker[i][worker_idx])
            train_idx = train_idx + 1
        for i in range(facility_factors_num):
            train_set[train_idx] = np.append(train_set[train_idx], train_facility[i][facility_idx])
            train_idx = train_idx + 1
        train_label_id.append(item["_id"])
        train_label.append(item["avg_sales"])
        stores_num.append(item["avg_store"])

    # area_code exception handling(About area that do not exist in Sales dataset)
    if not area_code in train_label_id:
        print(area_code, " is not in list")
        return "분석 불가"

    # split the data to train, test
    X_train = [np.array([]) for row in range(compare_factors_num)]
    X_test = [np.array([]) for row in range(compare_factors_num)]
    y_train = [np.array([]) for row in range(compare_factors_num)]
    y_test = [np.array([]) for row in range(compare_factors_num)]

    for i in range(compare_factors_num):
        X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(train_set[i], train_label, test_size=0.2,
                                                                        random_state=4)

    # make Linear Regression models and train the data
    regr = [linear_model.LinearRegression() for i in range(compare_factors_num)]
    for i in range(compare_factors_num):
        regr[i].fit(X_train[i].reshape(-1, 1), y_train[i])

    # predict test data, scores models
    guesses = [np.array([]) for row in range(compare_factors_num)]
    scores = []
    for i in range(compare_factors_num):
        guesses[i] = regr[i].predict(X_test[i].reshape(-1, 1))
        score = metrics.r2_score(y_test[i], guesses[i])
        if score > 0:
            scores.append(score)
        else:
            scores.append(0)

    # Limit the minimum value of score
    for score in scores:
        if score < 0:
            score = 0

    if sum(scores) == 0:
        print('예측 불가!')
        return "분석 불가"

    # Obtain weights from r2 scores so that the sum is 1
    weights = []
    for i in range(compare_factors_num):
        weights.append(scores[i] ** 2 / sum(scores))

    # set unknown target
    result = 0
    target_fpop_idx = train_fpop[fpop_factors_num].index(area_code)
    target_resident_idx = train_resident[resident_factors_num].index(area_code)
    target_worker_idx = train_worker[worker_factors_num].index(area_code)
    target_facility_idx = train_facility[facility_factors_num].index(area_code)
    target_store_idx = train_label_id.index(area_code)
    unknown_points = [0 for row in range(compare_factors_num)]
    target_idx = 0
    for i in range(fpop_factors_num):
        unknown_points[target_idx] = np.array(train_fpop[i][target_fpop_idx])
        target_idx = target_idx + 1
    for i in range(resident_factors_num):
        unknown_points[target_idx] = np.array(train_resident[i][target_resident_idx])
        target_idx = target_idx + 1
    for i in range(worker_factors_num):
        unknown_points[target_idx] = np.array(train_worker[i][target_worker_idx])
        target_idx = target_idx + 1
    for i in range(facility_factors_num):
        unknown_points[target_idx] = np.array(train_facility[i][target_facility_idx])
        target_idx = target_idx + 1
    store_num = stores_num[target_store_idx]

    # predict the unknown target
    for i in range(compare_factors_num):
        temp = regr[i].predict(unknown_points[i].reshape(-1, 1))
        result = result + temp * weights[i]

    # print the result
    top3idx = np.argsort(scores)[:-4:-1]
    top3factor = np.array(compare_factors)[top3idx]
    print('입력 받은 상권 & 서비스 업종: ', area_code, ", ", service_code)
    print('예상 월 매출 액: ₩', f'{int(result):,}')
    print('예상 월 매출 액(점포수로 나눈 값): ₩', f'{int(result / store_num):,}')
    print('가장 설명력이 높았던 가변 인자: ', top3factor)


    '''
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # visualize data
    path = 'C:\\WINDOWS\\Fonts\\KoPubDotumMedium.ttf'
    fontprop = fm.FontProperties(fname=path, size=14)

    fig = []
    figidx = 0
    for i in range(fpop_factors_num + resident_factors_num + worker_factors_num, compare_factors_num):
        fig.append(plt.figure())
        ax = fig[figidx].add_subplot(1, 1, 1)
        fig[figidx].suptitle("독립 변수: " + compare_factors[i], fontproperties=fontprop)
        ax.scatter(X_train[i], y_train[i])
        ax.plot(X_train[i], regr[i].predict(X_train[i].reshape(-1, 1)), color='red')
        figidx = figidx + 1
    plt.show()
    '''

    result_data = []

    result_data.append(f'{int(result / store_num):,}')
    for item in top3factor:
        result_data.append(str(item))

    return jsonify(result_data)


if __name__ == "__main__":
    app.run(debug=True, port=5000, host='127.0.0.1')