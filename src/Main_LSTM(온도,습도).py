import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM
from math import sqrt
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from keras.models import model_from_json
import time
from tensorflow import keras
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import *
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pymongo import MongoClient
from datetime import datetime
import pymongo
import copy
CONNECTION_STRING = "mongodb://netdb:netdb3230!@203.255.77.192:27017/"

client = MongoClient(CONNECTION_STRING)

# def get_database_size(collection_name):

Things_to_refer = "Things_to_refer"
system_model = 1

def compare_MSE_with_system_models_MSE(this_mse,model_name):
    global system_model,model # system_model = 시스템 모델, model = 시스템 모델과 성능 비교 대상이 되는 모델.
    previous_model_collection  = client[Things_to_refer]["Previous_model_features"]
    previous_model_doc = previous_model_collection.find_one()
    system_mse = previous_model_doc["MSE"]

    if this_mse < system_mse:
        system_model = model # system_model은 전역_변수로 필히 선언.
        update_query = {'$set': {'MSE': this_mse}}
        previous_model_collection.update_one({"_id":previous_model_doc['_id']},update_query)

    # 기존 system_model이 탈락하면 모델 별로 재학습된 결과를 저장하는 mongodb의 model_collection에서 system_model과 같은 모델의
    # 재학습된 결과와 기존 탈락한 system_model을 비교해 누구를 모델 대표의 모델 파일로 살릴지 결정.

def match_infacility_with_growth(infacilitys,growth_dbNames): # infacilitys(하우스 번호 리스트)과 growth_dbNames는 1:1 매핑으로 주어져야한다. 각 하우스번호에 맞는 생육 db infacilitys는 GH2에 속함.
    
    # 이게 1분 단위 환경 데이터를 1일 평균 환경 데이터로 만드는 함수라서, 이 환경 데이터의 시작 날짜와 
    # 생육 데이터의 시작 날짜가 동일해야한다. 참고로 (생육:환경 = 1:7) 비율로 매핑.

    def plus(document,temp_prefix_sum,humidity_prefix_sum):
        temp_prefix_sum += document['temp']
        humidity_prefix_sum += document['humidity']
        return temp_prefix_sum,humidity_prefix_sum
    
    day_avg_env_data_for_each_facilitys = {}

    GH2_collection = client["TestAPI"]["GH2"]
    infalicitys = {34:"hydroponics_length1",35:"hydroponics_length2"} # 하우스 번호
    cnt = 0
    for i in range(len(infacilitys)):
        
        infacility = infacilitys[i] # 얘는 디비나 컬렉션 이름이 아닌 쿼리문으로 사용된다.
        
        query = {"inFacilityId":infacility}
        result = GH2_collection.find(query)
        now_date = None
        temp_prefix_sum = 0
        humidity_prefix_sum = 0
        document_cnt = 0

        env_data_1day_avg = {}
        
        for document in result:
            if not cnt%1_000_000:
                print(cnt)
            date = document['sensingAt'].split()[0] # 2023-01-06 00:03:01 공백으로 split후 날짜만 파싱
            if now_date == None:
                now_date = date
                temp_prefix_sum,humidity_prefix_sum = plus(document,temp_prefix_sum,humidity_prefix_sum)
                document_cnt += 1

            elif date != now_date:
                env_data_1day_avg[now_date] = [temp_prefix_sum/document_cnt,humidity_prefix_sum/document_cnt]
                # print(f"{now_date}의 document는 {document_cnt}개 있었고, temp_sum:{round(temp_prefix_sum,3)}, avg :{round(env_data_1day_avg[now_date][0],3)}, humidity_sum:{round(humidity_prefix_sum,3)} ,avg :{round(env_data_1day_avg[now_date][1],3)}")
                now_date = date
                temp_prefix_sum,humidity_prefix_sum = 0,0 # 초기화
                temp_prefix_sum,humidity_prefix_sum = plus(document,temp_prefix_sum,humidity_prefix_sum)
                document_cnt = 1

            else:
                temp_prefix_sum,humidity_prefix_sum = plus(document,temp_prefix_sum,humidity_prefix_sum)
                document_cnt += 1
            cnt += 1

        if now_date != None and now_date not in env_data_1day_avg.keys():
            env_data_1day_avg[now_date] = [temp_prefix_sum/document_cnt,humidity_prefix_sum/document_cnt]
        day_avg_env_data_for_each_facilitys[infacility] = env_data_1day_avg
    week_growth_data_for_each_facilitys = {}
    for i in range(len(growth_dbNames)):
        growth_dbName = growth_dbNames[i]
        week_growth_data_for_each_facilitys[growth_dbName] = {} # 샘플변호별 생육 분류 저장 키 = 샘플 번호 , val = 생육 데이터
        growth_collection = client["TestAPI"][growth_dbName]
        all_document = growth_collection.find()

        for document in all_document:
            length_cm = document['growth length   (cm)']
            # if length_cm < 0: # 생장길이가 음수인 다큐먼트가 있음 {"_id" : ObjectId("64a2659f5bc0a5dcd22ffe92")}
            #     continue
            if document['sample_num'] not in week_growth_data_for_each_facilitys[growth_dbName].keys():
                week_growth_data_for_each_facilitys[growth_dbName][document['sample_num']] = []
            week_growth_data_for_each_facilitys[growth_dbName][document['sample_num']].append(length_cm)


    return day_avg_env_data_for_each_facilitys,week_growth_data_for_each_facilitys
    
def map_env_growth_7vs1(env_data,growth_data): # 모델 재학습 input data 만들기 나중에 test:train 따로 나눠야 함.
    global old_model,scaler
    env_key = []
    growth_key = []

    for ki in env_data.keys():
        env_key.append(ki)
    for ki in growth_data.keys():
        growth_key.append(ki)

    for I in range(len(env_key)):
        x = []
        y = []
        env_data_for_tmp_growth_data = [] # tmp_growth_data는 중복 환경 데이터를 가지고, 이 리스트는 중복 x이다.
        tmp_env_data = [] # tmp_env_data는 인덱스로 쉽게 처리하기 위해 이전의 dct형태의 env_data를 리스트 형식으로 새로 만듬.
        tmp_growth_data =[] # tmp_env_data와 동일한 목적으로 tmp_growth_data 생성.
        # next(iter(딕셔너리 A)) 딕셔너리 A의 첫번째 KEY가져오는 함수
        
        for ki in env_data[env_key[I]].keys(): # ki는 날짜
            env_data_for_tmp_growth_data.append(env_data[env_key[I]][ki])

        for idx in range(len(growth_data[growth_key[I]][next(iter(growth_data[growth_key[I]]))])):
            for sample in growth_data[growth_key[I]].keys():
                tmp_growth_data.append(growth_data[growth_key[I]][sample][idx]) # sample과 idx 순서 뒤바꾼거 맞으니까 헷갈림 주의
                for k in range(7):
                    tmp_env_data.append(env_data_for_tmp_growth_data[idx*7+k])


        # for i in range(len(tmp_growth_data)):
        #     y.append(tmp_growth_data[i])
        #     for k in range(7):
        #         x.append(tmp_env_data[(i//len(tmp_growth_data))*7+k])

        x = np.array(tmp_env_data)
        y = np.array(tmp_growth_data)
        x_scaled = scaler.fit_transform(x)
        shape = (len(x_scaled)//(2*7))*(2*7) # 나머지 컷.
        x_scaled = x_scaled[:shape]
        x_scaled.reshape(len(x_scaled)//7,7,2)
        y_scaled = y

        x_train_size = int(len(x_scaled)*0.8) # 1747.2 = 1747
        y_train_size = int(len(y_scaled)*0.8) # 249.6 = 249

        train_x = x_scaled[:x_train_size,:] # 1747
        test_x = x_scaled[x_train_size:,:] # 437

        train_y = y_scaled[:y_train_size] # 249
        test_y = y_scaled[y_train_size:] # 63
        
        train_reshape1 = x_train_size//7 # 249
        test_reshape1 = len(test_x)//7 # 62
        train_x = train_x[:train_reshape1*7] # 1743
        train_x = train_x.reshape((train_reshape1,7,2))
        test_x = test_x[:test_reshape1*7] # 62
        test_x = test_x.reshape((test_reshape1,7,2))

        test_y_1 = test_y
        train_x = train_x.reshape((train_x.shape[0],train_x.shape[1],2))
        test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], 2))

        train_y = train_y.reshape((train_y.shape[0], 1,1))
        test_y = test_y.reshape((test_y.shape[0], 1,1))

        old_model.fit(train_x,train_y,epochs=500,batch_size=150,verbose=2,shuffle=False)

        size = min(len(test_x),len(test_y_1))
        predictions = old_model.predict(test_x[:size])
        new_loss = old_model.evaluate(test_x[:size],test_y_1[:size])
        for pre_val,act_val in zip(predictions,test_y_1): # 계속 똑같은 예측 값만 나오는 경우가 있는데 그 이유는 predictions에 매개변수로 사용된 test_x[:size]가 중복 데이터를 허용하는 tmp_env_data에서 20%의 데이터를 x_test로 때온 거라서 중복되는 x_test가 많아서 계속 같은 값이 나오는 것이다.
            print("Predicted:", pre_val)
            print("Actual:", act_val)
        new_loss = print(new_loss)
def reload_model(name): # 이전에 학습한 모델을 재학습 시키기 위해 불러오기 # ex) name = "old_model.json"
    ##### 모델 reload #####
    with open(name, "r") as json_file:
        loaded_model_json = json_file.read()
    model = keras.models.model_from_json(loaded_model_json)
    model_weights_path = "old_model.h5"
    model.load_weights(model_weights_path)
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    return model

def call_data_and_refine(env_db,env_collection,infacility_lst,growth_db,growth_collection_lst): # 트리거가 작동하면 몽고디비에서 데이터를 읽어와 모델의 input 형식인 pandas.core.frame.DataFrame로 변환
                            # 여기서 몽고디비 데이터베이스에 저장된 환경 데이터들 중 "새로운 최신" 데이터들만 선택적으로 선별해야함.
                            # 즉, 이미 학습에 쓰였던 환경 데이터는 재학습 때 제외하고, 새로 쌓인 데이터들만 재학습에 사용된다.
    
    db = client[env_db]
    collection = db[env_collection]
    
    Previous_model_features = client[Things_to_refer]["Previous_model_features"].find_one() # Previous_model_features가 한개의 Json document를 포함하니까 .find_one()함수가 가능한거임 만약 여러 document를 포함하면 id로 접근해야 함.
    most_recent_time = Previous_model_features["Most_Recent_date"]
    datetime1 = datetime.strptime(most_recent_time,"%Y-%m-%d %H:%M:%S")
    str_new_most_recent_time = copy.deepcopy(most_recent_time)
    new_most_recent_time = datetime.strptime(str_new_most_recent_time,"%Y-%m-%d %H:%M:%S")
    cnt = 0

    # 생육 데이터를 불러와 모델 재학습시 이용.
    
    growth_collection = "hydroponics_first"
    db_growth = client[growth_db]
    collection_growth = db_growth[growth_collection]

    Previous_model_features_growth = client[Things_to_refer]["Previous_model_features"].find_one() 
    most_recent_time_growth = Previous_model_features_growth["Most_Recent_date(growth)"] # 이거는 int라서 바로 비교가능
    new_most_recent_time_growth = most_recent_time_growth

    try:
        cursor = collection.find()
        while(True):
            cnt += 1
            if not cnt%10_000:
                print(cnt)
                # 70~71코드는 call_data_and_refine()가 데이터베이스의 모든 다큐먼트를 조회하므로 매번 코드를 수행할 때 시간이 너무오래걸려서 임시로 넣은 코드 실제 코드에서는 제거할 것.
                if cnt == 10_000:
                    break
            try:
                document = next(cursor)
                str_datetime2 = document["sensingAt"]
                datetime2 = datetime.strptime(str_datetime2,"%Y-%m-%d %H:%M:%S")

                if datetime1 < datetime2:
                    Temp.append(document["temp"])
                    Humidity.append(document["humidity"])
                    ###### GH1의 데이터 셋을 생성하는 농장에 CO2센서가 없다... ####
                    # CO2.append(document["co2"])
                
                if new_most_recent_time < datetime2:
                    str_new_most_recent_time = str_datetime2
                    new_most_recent_time = datetime.strptime(str_new_most_recent_time,"%Y-%m-%d %H:%M:%S")

            except StopIteration:
                break
            except pymongo.errors.InvalidBSON:
                continue

    except pymongo.errors.PyMongoError as e:
        print("Error occurred:", e)
    
    if new_most_recent_time < datetime1:
        # 여기에 Previous_model_features_growth = client[Things_to_refer]["Previous_model_features"].find_one() 
        # most_recent_time_growth = Previous_model_features_growth["Most_Recent_date(growth)"]
        # 이거를 데이터베이스에 최신화하기 구현. 구현했으면 주석 지우기..
        pass
    try:
        cursor = collection_growth.find()
        no_date = 0 # date 키가 없는 다큐먼트 개수
        while(True):
            try:
                document = next(cursor)
                if "date" in document.keys():
                    growth_date = document["date"]# int 라서 바로 비교가능
                else:
                    no_date += 1
                    continue
                if most_recent_time_growth < growth_date:
                    growth.append(document["plant_height              (㎝)"]) # ? hydroponics1에는 생장길이가 없다??

                if new_most_recent_time_growth < growth_date:
                    new_most_recent_time_growth = growth_date
                

            except StopIteration:
                break
            except pymongo.errors.InvalidName:
                continue

    except pymongo.errors.PyMongoError as e:
        print("Error occurred:", e)

    #여기 생육버전으로 바꾸기.
    if new_most_recent_time < datetime1:
        # 여기에 Previous_model_features_growth = client[Things_to_refer]["Previous_model_features"].find_one() 
        # most_recent_time_growth = Previous_model_features_growth["Most_Recent_date(growth)"]
        # 이거를 데이터베이스에 최신화하기 구현. 구현했으면 주석 지우기..
        pass

    #### 이것도 매번 최신화되는걸 다시 고치기 귀찮으니까 일단 비활성화
    # if datetime1 < new_most_recent_time:
    #     client[Things_to_refer]["Previous_model_features"]["Most_Recent_date"] = str_new_most_recent_time
    
def trigger(): # 여기에 mongodb 데이터가 전 보다 20% 증가하면 데이터를 싹 다 긁어와 재학습 시키는 방식.
    weight = 1.2 # 현재 읽은 데이터가 이전에 가장 최근에 읽었던 시점보다 20%가 더 많으면 재학습한다.
    Things_to_refer = "Things_to_refer"
    dbname = client["TestAPI"] # TestAPI 데이터 베이스에 접속

    current_size = dbname.GH1.estimated_document_count() # 왜 이줄의 코드가 작동하는지는 모르겠지만 컬렉션의 이름을 문자열이 아닌 GH1이름 그대로 넣는다.

    environment_Collections_size =  client[Things_to_refer]["environment_Collections_size"]

    previous_size = environment_Collections_size.find_one({"_id":0})["size"]
    
    if current_size>=weight*previous_size:
        # 이 때는 재학습을 하기 때문에 "environment_Collections_size"를 최신화한다.
        # 혹시라도 재학습 코드에서 environment_Collections_size을 참고하는 코드가 있다면 선 최신화에 대해 발생하는 문제가 없는지 확인 필요.

        #★★★★★★★★★★★★★★매번 덧씌워지는게 귀찮으니까 일단 비활성화 시켰음..★★★★★★★★★★

        # previous_dict = environment_Collections_size.find_one({"_id":0})
        # previous_dict["size"] = current_size
        # environment_Collections_size.replace_one({"_id":0},previous_dict)

        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        return 1
    else:
        # 이 때는 재학습을 하지 않기 때문에 "environment_Collections_size"를 최신화하지 않는다.
        return 0

# while(1): # trigger를 계속 while문으로 돌린다.
if trigger():

    #★ ★ ★ ★ ★ ★ ★ ★ match_infacility_with_growth를 빠르게 체크하기위해 잠시 비활성화 시켰음(별 줄사이의 모든 주석을 해제하면 됨.) ★ ★ ★ ★ ★ ★ ★ ★ ★
    scaler = MinMaxScaler()
    print("재학습 및 데이터베이스 사이즈 최신화 완료..")
    CO2 = []
    Humidity = []
    Temp = []
    growth = []
    # call_data_and_refine("TestAPI","GH1","TestAPI",[34,35]) # 이 함수르 수행하면 CO2,Humidity,Temp 리스트가 채워짐.
    data = {"온도":Temp,"습도":Humidity}

    old_model = reload_model("old_model.json")

    day_avg_env_data_for_each_facilitys,week_growth_data_for_each_facilitys = match_infacility_with_growth([34,35],["hydroponics_length1","hydroponics_length2"])


    # day_avg_env_data_for_each_facilitys = {
    #                                            34:{날짜1:[하루 평균 온도,하루 평균 습도],날짜2:[하루 평균 온도,하루 평균 습도]},
    #                                            35:{"34와 동일한 형태"     } 
    #                                                                           }

    map_env_growth_7vs1(day_avg_env_data_for_each_facilitys,week_growth_data_for_each_facilitys)

    # 컬렉션 가장 첫번째, 마지막 문서 조회하는 코드 (해당 컬렉션 기간 확인을 위함.)
    # query = {"inFacilityId":34}
    
    # print(client["TestAPI"]["GH2"].find_one(query))
    # print(client["TestAPI"]["GH2"].find_one(query,sort=[("$natural", pymongo.DESCENDING)]))
    # print()
    # query = {"inFacilityId":35}
    # print(client["TestAPI"]["GH2"].find_one(query))
    # print(client["TestAPI"]["GH2"].find_one(query,sort=[("$natural", pymongo.DESCENDING)]))
    # print()
    # print(client["TestAPI"]["hydroponics_length1"].find_one())
    # print(client["TestAPI"]["hydroponics_length1"].find_one(sort=[("$natural", pymongo.DESCENDING)]))
    # print()
    # print(client["TestAPI"]["hydroponics_length2"].find_one())
    # print(client["TestAPI"]["hydroponics_length2"].find_one(sort=[("$natural", pymongo.DESCENDING)]))
    # df = pd.DataFrame(data)
    # x_values = df.values
    # y_values = np.array(growth)

    # slice_Int = (len(y_values)*24) # 이 값이 딱 맞아떨어져야함 (4000,2) = reshape(1000,4,2)이런식으로 4000 - 4000 = 0

    # x_values = x_values[:slice_Int]

    # y_train_size = int(len(y_values)*0.8)
    # x_train_size = y_train_size*24*len(data)
    
    # x_scaled = scaler.fit_transform(x_values)
    # x_scaled.reshape(len(y_values),24,len(data))

    # train_x = x_scaled[:x_train_size//len(data),:]
    # test_x = x_scaled[x_train_size//len(data):,:]
    # print(type(y_values))
    # print(y_values.shape)
    # # train_y = y_values[:y_train_size,:]
    # # test_y = y_values[y_train_size:,:]
    # train_y = y_values[:y_train_size]
    # test_y = y_values[y_train_size:]

    # train_reshape1 = x_train_size//(24*len(data))
    # test_reshape1 = (len(x_values) - train_reshape1*24)//24

    # train_x = train_x.reshape((train_reshape1,24,len(data)))
    # test_x = test_x.reshape((test_reshape1,24,len(data)))

    # train_x = train_x.reshape((train_x.shape[0],train_x.shape[1],len(data)))
    # test_x = test_x.reshape((test_x.shape[0],test_x.shape[1],len(data)))
    # train_y = train_y.reshape((train_y.shape[0],1,1))
    # test_y = test_y.reshape((test_y.shape[0],1,1))

    ##### 모델 reload #####
    model_json_path = "old_model.json"
    with open(model_json_path, "r") as json_file:
        loaded_model_json = json_file.read()
    model = keras.models.model_from_json(loaded_model_json)
    model_weights_path = "old_model.h5"
    model.load_weights(model_weights_path)
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    
    # model.fit(train_x,train_y,epochs=500,batch_size=150,verbose=2,shuffle=False)

    # #### 재학습된 모델은 new_model.json과 new_model.h5에 저장한다.
    # y_pred = model.predict(test_x) # x_test(온도,습도) 넣어줄테니까 초장길이 한번 예측해보셈.
    # print(test_y.shape)
    # test_y = np.squeeze(test_y)
    # print(y_pred.shape)
    # print(type(test_y),type(y_pred))
    # for i in y_pred:
    #     print(i)
    # mse = mean_squared_error(test_y,y_pred)
    # print("MSE:",mse)


    # ######## System Model 과 MSE 비교 #########
    # compare_MSE_with_system_models_MSE(mse)

    # y_pred = np.round(y_pred).flatten() # 확률을 0또는 1로 변환
    # accuracy = accuracy_score(test_y,y_pred)
    # print("Accuracy:",accuracy)

    # retrain_model_json = model.to_json()
    # with open("new_model.json", "w") as json_file:
    #     json_file.write(retrain_model_json)
    # model.save_weights("new_model.h5")
    # ############################################################

    # print(1)

    #★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★

else:
    print("아직 재학습이 필요하지 않습니다.")
    # (while문에 해당되는 코드)time.sleep(86400) # 86400초 = 하루 가 지나면 다시 trigger 조회를 한다.