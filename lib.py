import json
import csv
import traceback
import numpy as np
from collections import defaultdict
import os
import pickle
from scipy.spatial.transform import Rotation as R
from scipy.stats import mode, spearmanr, shapiro, ttest_rel, wilcoxon, normaltest, ttest_ind, mannwhitneyu, kruskal, friedmanchisquare
import warnings
import pandas as pd
warnings.filterwarnings("error", category=UserWarning)
warnings.filterwarnings("error", category=RuntimeWarning)

N = 23 + 1
TASK_ORDER = [
    [4, 3, 1, 2], [3, 1, 4, 2], [2, 4, 1, 3], [2, 4, 3, 1], [3, 4, 2, 1], 
    [4, 1, 3, 2], [1, 4, 2, 3], [4, 3, 2, 1], [1, 3, 4, 2], [4, 2, 1, 3],
    [1, 2, 4, 3], [2, 1, 4, 3], [3, 1, 2, 4], [4, 2, 3, 1], [3, 2, 4, 1],
    [1, 3, 2, 4], [2, 3, 4, 1], [2, 1, 3, 4], [4, 1, 2, 3], [3, 2, 1, 4],
    [1, 4, 3, 2], [3, 4, 1, 2], [2, 3, 1, 4], [1, 2, 3, 4]
]

DBDATA = os.path.join("bdata", "{0:02d}")
FGAZE = os.path.join("data", "{0:02d}", "{0:02d}_{1:1d}_gaze.csv")
FPLAYER = os.path.join("data", "{0:02d}", "{0:02d}_{1:1d}_player.csv")
FROBOT = os.path.join("data", "{0:02d}", "{0:02d}_{1:1d}_robot.csv")
FINFO = os.path.join("data", "{0:02d}", "{0:02d}_{1:1d}_taskinfo.json")
FSUBJECT = os.path.join("data", "subject.csv")
FSUBJECT_ATTR = os.path.join("data", "subject_attr.csv")
FSUBJECT_RANK = os.path.join("data", "subject_ranking.csv")
BDATA = os.path.join("bdata", "{0:02d}", "{0:02d}_{1:1d}.pkl")
BDATA_SUBTASK = os.path.join("bdata", "{0:02d}", "{0:02d}_{1:1d}_subtask.pkl")

KEY_INFO = ["taskTimeAll", "taskTimeParts", "taskCollisionAll", "taskCollisionParts", "taskMistakeAll", "taskMistakeParts", "mapId", "userId", "uiId"]
KEY_PLAYER = ["pos", "rot", "bpm", "trigger", "state", "subTask", "warning", "collision"]
KEY_GAZE = ["time", "lg_pos", "lg_rot", "rg_pos", "rg_rot", "obj"]
KEY_ROBOT = ["r_id", "pos", "rot_y"]

COLLISION_FLAG = [
    [True, True, False, False, True, True, True, True, False],
    [False, True, True, True, False, False, True, True, True],
    [False, False, True, False, True, True, True, True, True],
    [True, True, True, False, True, True, True, False, False],
    [True, False]
]

ROBOT_NUM = [1, 3, 3, 1, 3, 3, 1, 2, 2, 2, 2]

"""
allData = [
    [taskData-0, taskData-1, taskData-2, taskData-3], # userId 3
    [taskData-0, taskData-1, taskData-2, taskData-3], # userId 4
    ...
]: np.array(list) | shape(UserNUM, 4)

taskData = [
    subtaskData-0: dict(),
    subtaskData-1: dict(),
    ...
    subtaskData-8: dict()
] : np.array(dict()) | shape(9, )

nullData = [None]*9 : np.array() | shape(9, )

subtaskData = {
    "time": np.array(float) | shape(frameNum, ),

    "pos": np.array(float) | shape(frameNum, 3), # camera position
    "rot": np.array(float) | shape(frameNum, 3), # camera rotation
    "lg_pos": np.array(float) | shape(frameNum, 3), # left eye global position
    "lg_rot": np.array(float) | shape(frameNum, 3), # left eye rotation
    "rg_pos": np.array(float) | shape(frameNum, 3), # right eye position
    "rg_rot": np.array(float) | shape(frameNum, 3), # right eye rotation

    "lg_pupil_d": np.array(float) | shape(frameNum, ), # left eye pupil diameter
    "lg_pupil_pos": np.array(float) | shape(frameNum, 2), # left eye pupil position
    "rg_pupil_d": np.array(float) | shape(frameNum, ), # right eye pupil diameter
    "rg_pupil_pos": np.array(float) | shape(frameNum, 2), # right eye pupil position
    "lg_open": np.array(float) | shape(frameNum, ), # left eye open
    "rg_open": np.array(float) | shape(frameNum, ), # right eye open
    
    "obj": np.array(str) | shape(frameNum, ), # gaze object name
    "bpm": np.array(int) | shape(frameNum, ), # heart rate
    "trigger": np.array(bool) | shape(frameNum, ), # button pressed
    "subTask": np.array(bool) | shape(frameNum, ), # subtask
    "warning": np.array(int) | shape(frameNum, ), # warning robot count
    "warning_filter": np.array(int) | shape(frameNum, ), # warning robot count (mode filter)
    "collision": np.array(bool) | shape(frameNum, ), # collision

    "robot": [robotData] * 6, # robot data
    "robot_cnt": np.array(int) | shape(frameNum, ), # moving robot count
    "min_dist": np.array(float) | shape(frameNum, ), # minimum distance between robot and camera

    "userId": int, # user id
    "uiId": int # ui id
    "taskOrder": int, # task order
    "state": int, # subtask id
    "taskTime": float, # subtask time
    "taskCollision": int, # task collision count
    "taskMistake": int, # task mistake count
    "collision_flag": bool, # robot collision
    "label": str # label for graph plot example: "0-1*"
}: dict 

robotData = {
    "r_id": np.array(int) | shape(frameNum, ), # robot id (0-14, 99:null)
    "pos": np.array(float) | shape(frameNum, 3), # robot position
    "rot_y": np.array(float) | shape(frameNum, ) # robot rotation
}: dict
"""


"""
subjectiveData = {
    "userId": np.array(int) | shape(userNum, 4), # user id
    "uiId": np.array(int) | shape(userNum, 4), # ui id

    "easy": np.array(int) | shape(userNum, 4), # easy to use (1-7, -1: null)
    "annoy": np.array(int) | shape(userNum, 4), # annoying (1-7, -1: null)
    "useful": np.array(int) | shape(userNum, 4), # useful (1-7, -1: null)
    "trust": np.array(int) | shape(userNum, 4), # trust (1-7, -1: null)

    "notice": np.array(int) | shape(userNum, 4), # notice robot (1-7)
    "distance": np.array(int) | shape(userNum, 4), # distance (1-7)
    "direction": np.array(int) | shape(userNum, 4), # direction (1-7)
    "safe": np.array(int) | shape(userNum, 4), # safe (1-7)
    "vr": np.array(int) | shape(userNum, 4), # vr sickness (1-7)
    "avoid": np.array(int) | shape(userNum, 4), # avoid robot (1-7)

    "mental": np.array(int) | shape(userNum, 4), # NASA-TLX mental load (0-100)
    "physical": np.array(int) | shape(userNum, 4), # NASA-TLX physical load (0-100)
    "temporal": np.array(int) | shape(userNum, 4), # NASA-TLX temporal load (0-100)
    "performance": np.array(int) | shape(userNum, 4), # NASA-TLX performance load (0-100)
    "effort": np.array(int) | shape(userNum, 4), # NASA-TLX effort load (0-100)
    "frustration": np.array(int) | shape(userNum, 4), # NASA-TLX frustration load (0-100)
    "score": np.array(float) | shape(userNum, 4), # NASA-TLX score (0-100)

    "comment": np.array(str) | shape(userNum, 4) # comment
}
"""

"""
subjectiveAttrData = {
    "userId": int, # user id
    "faculty": str, # faculty
    "age": int, # age
    "gender": str, # gender (m, f, o)
    "exercise": int, # exercise frequency (0-3)
    "duration": int, # exercise duration (0-3)
    "mobile": int, # mobile game frequency (0-2)
    "console": int, # console game frequency (0-2)
    "vr": int, # vr game frequency (0-2)
    "genre": str, # game genre (mobile, console, vr, none)
    "sleep": float, # sleep duration
    "waking": float, # waking time
    "health": int, # health status (1-5)
    "correction": bool, # correction (True/False)
    "vision": str, # vision
    "refraction": str, # refraction
}: DataFrame.key
"""

"""
subjectiveRankingData = {
    "userId": list(int), # user id
    "easy": numpy.array(int) | shape(userNum, 4), # easy to use ranking (1-4)
    "annoy": numpy.array(int) | shape(userNum, 4), # annoying ranking (1-4)
    "useful": numpy.array(int) | shape(userNum, 4), # useful ranking (1-4)
    "trust": numpy.array(int) | shape(userNum, 4), # trust ranking (1-4)
    "notice": numpy.array(int) | shape(userNum, 4), # notice robot ranking (1-4)
    "distance": numpy.array(int) | shape(userNum, 4), # distance ranking (1-4)
    "direction": numpy.array(int) | shape(userNum, 4), # direction ranking (1-4)
    "safe": numpy.array(int) | shape(userNum, 4), # safe ranking (1-4)
    "vr": numpy.array(int) | shape(userNum, 4), # vr ranking (1-4)
    "avoid": numpy.array(int) | shape(userNum, 4), # avoid robot ranking (1-4)
    "comment": list(str) # comment
}
"""



# csv data to binary ********************
def convert_binary(userId, uiId):
    data = defaultdict(list)

    with warnings.catch_warnings():
        warnings.simplefilter("default", category=UserWarning)
        warnings.simplefilter("default", category=RuntimeWarning)

        head_quats = []
        filename = FPLAYER.format(userId, uiId)
        with open(filename, 'r', encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader) # time, pos_x, pos_y, pos_z, qua_x, qua_y, qua_z, qua_w, bpm, trigger, state, subTask, count(robot), collision
            for row in reader:
                data["pos"].append((float(row[1]), float(row[2]), float(row[3])))
                hq = R.from_quat((float(row[4]), float(row[5]), float(row[6]), float(row[7])))
                head_quats.append(hq)
                data["rot"].append(hq.as_euler("xyz", degrees=True))
                data["bpm"].append(int(row[8]))
                data["trigger"].append(bool(int(row[9])))
                data["state"].append(int(row[10]))
                data["subTask"].append(bool(int(row[11])))
                data["warning"].append(int(row[12]))
                data["collision"].append(bool(int(row[13])))


        filename = FGAZE.format(userId, uiId)
        with open(filename, 'r', encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader) # time, l_pos_x, l_pos_y, l_pos_z, l_qua_x, l_qua_y, l_qua_z, l_qua_w, r_pos_x, r_pos_y, r_pos_z, r_qua_x, r_qua_y, r_qua_z, r_qua_w, obj, l_pupil_d, l_pupil_x, l_pupil_y, r_pupil_d, r_pupil_x, r_pupil_y, l_open, r_open
            for row, hq in zip(reader, head_quats):
                data["time"].append(float(row[0]))
                data["lg_pos"].append((float(row[1]), float(row[2]), float(row[3])))
                lg_local_q = hq.inv() * R.from_quat((float(row[4]), float(row[5]), float(row[6]), float(row[7])))
                data["lg_rot"].append(lg_local_q.as_euler("xyz", degrees=True))
                data["rg_pos"].append((float(row[8]), float(row[9]), float(row[10])))
                rg_local_q = hq.inv() * R.from_quat((float(row[11]), float(row[12]), float(row[13]), float(row[14])))
                data["rg_rot"].append(rg_local_q.as_euler("xyz", degrees=True))
                data["obj"].append(row[15].strip())
                data["lg_pupil_d"].append(float(row[16]))
                data["lg_pupil_pos"].append((float(row[17]), float(row[18])))
                data["rg_pupil_d"].append(float(row[19]))
                data["rg_pupil_pos"].append((float(row[20]), float(row[21])))
                data["lg_open"].append(float(row[22]))
                data["rg_open"].append(float(row[23]))
        
    
    filename = FROBOT.format(userId, uiId)
    with open(filename, 'r', encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader) # time,r1_userId,r1_x,r1_z,r1_rot_y,r2_userId,r2_x,r2_z,r2_rot_y,r3_userId,r3_x,r3_z,r3_rot_y,pr1_userId,pr1_x,pr1_z,pr1_rot_y,pr2_userId,pr2_x,pr2_z,pr2_rot_y,pr3_userId,pr3_x,pr3_z,pr3_rot_y,cnt(robot)
        data["robot"] = [defaultdict(list) for _ in range(6)]
        for row in reader:
            for i in range(0, 6):
                try: 
                    data["robot"][i]["r_id"].append(int(row[1 + i * 4]))
                    data["robot"][i]["pos"].append((float(row[2 + i * 4]), float(0), float(row[3 + i * 4])))
                    data["robot"][i]["rot_y"].append(float(row[4 + i * 4]))
                except:
                    print(row)
            data["robot_cnt"].append(int(row[25]))
    
    for key in data.keys():
        if key == "robot":
            for i in range(6):
                data[key][i]["r_id"] = np.array(data[key][i]["r_id"])
                data[key][i]["pos"] = np.array(data[key][i]["pos"])
                data[key][i]["rot_y"] = np.array(data[key][i]["rot_y"])
        else:
            data[key] = np.array(data[key])

    filename = FINFO.format(userId, uiId)
    with open(filename, "r", encoding="utf-8-sig") as f:
        info = json.load(f)
        data["taskTimeAll"] = info["taskTimeAll"]
        data["taskTimeParts"] = np.array(info["taskTimeParts"])
        data["taskCollisionAll"] = info["taskCollisionAll"]
        data["taskCollisionParts"] = np.array(info["taskCollisionParts"])
        data["taskMistakeAll"] = info["taskMistakeAll"]
        data["taskMistakeParts"] = np.array(info["taskMistakeParts"])
        data["mapId"] = info["mapId"]
        data["userId"] = userId
        data["uiId"] = uiId

    os.makedirs(DBDATA.format(userId), exist_ok=True)
    with open(BDATA.format(userId, uiId), "wb") as f:
        pickle.dump(dict(data), f)

def convert_subject(new=False):
    if not new and os.path.exists(os.path.join("bdata", "subject.pkl")):
        print("subject.pkl already exists")
        return
    
    data = []
    with open(FSUBJECT, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f) # Id	開始時刻	完了時刻	メール	名前	被験者ID	Map ID	UI ID	UIについて.UIが直感的でわかりやすいと感じた	UIについて.UIがタスク中に邪魔に感じた	UIについて.UIがロボットに気づく際に便利に感じた	UIについて.UIがロボットを避けるための支援として信頼できると感じた	ロボットの接近に気づくことができた.接近	接近してくるロボットとの距離を把握することができた.距離	接近してくるロボットの方向を把握することができた.方向	サブタスク中、安全に感じた.安全	VR酔いを感じた.VR酔い	ロボットとの衝突を回避することができた.ロボットとの衝突を回避	作業負荷に関する質問（NASA-TLX）	ロボットがいる環境での作業に対する感想	自由記述
        next(reader)
        for i, row in enumerate(reader):
            if i%4 == 0:
                data.append([dict() for _ in range(4)])
            userId = int(row[5])
            uiId = int(row[7])
            data[i//4][uiId]["userId"] = userId
            data[i//4][uiId]["uiId"] = uiId
            if uiId != 3:
                data[i//4][uiId]["easy"] = int(row[8])
                data[i//4][uiId]["annoy"] = int(row[9])
                data[i//4][uiId]["useful"] = int(row[10])
                data[i//4][uiId]["trust"] = int(row[11])
            else:
                data[i//4][uiId]["easy"] = int(-1)
                data[i//4][uiId]["annoy"] = int(-1)
                data[i//4][uiId]["useful"] = int(-1)
                data[i//4][uiId]["trust"] = int(-1)
            data[i//4][uiId]["notice"] = int(row[12])
            data[i//4][uiId]["distance"] = int(row[13])
            data[i//4][uiId]["direction"] = int(row[14])
            data[i//4][uiId]["safe"] = int(row[15])
            data[i//4][uiId]["vr"] = int(row[16])
            data[i//4][uiId]["avoid"] = int(row[17])
            data[i//4][uiId]["ui-comment"] = row[19]
            data[i//4][uiId]["comment"] = row[20]
            for key, value in json.loads(row[18]).items():
                if key == "score":
                    data[i//4][uiId][key] = float(value)
                else:
                    data[i//4][uiId][key] = int(value[0])
            # data[i//4][uiId]["load"] = json.loads(row[17])
    
    data_ = {
        key:  np.vectorize(lambda x: x[key])(data)
        for key in data[0][0].keys()
    }
    
    os.makedirs(DBDATA.format(0), exist_ok=True)
    with open(os.path.join("bdata", "subject.pkl"), "wb") as f:
        pickle.dump(data_, f)
        print("subject.pkl saved")

def split_subtask(userId, uiId):
    filename = BDATA.format(userId, uiId)
    if not os.path.exists(filename):
        try:
            convert_binary(userId, uiId)
        except Exception as e:
            print("Error {0:02d}_{1:1d}: {2}".format(userId, uiId, e))
            return None
    
    with open(filename, "rb") as f:
        data = pickle.load(f)

    diff = np.diff(data["subTask"].astype(int), append=False)
    start_idxs = np.where(diff == 1)[0] + 1
    end_idxs = np.where(diff == -1)[0] + 1
   
    subtask_data = []
    for i, (st, ed) in enumerate(zip(start_idxs, end_idxs)):
        masked_data = dict()
        for key in data.keys():
            if key == "robot":
                masked_data["robot"] = [dict() for _ in range(6)]
                for j in range(6):
                    for k in data["robot"][j]:
                        masked_data[key][j][k] = data[key][j][k][st:ed]
            elif key not in KEY_INFO:
                masked_data[key] = data[key][st:ed]
        
        min_dist = []
        for j in range(st, ed):
            mdist = 10**10
            for k in data["robot"]:
                if k["r_id"][j] == 99: continue
                mdist = min(distance(data["pos"][j], k["pos"][j])[0], mdist)
            min_dist.append(mdist)
        masked_data["min_dist"] = np.array(min_dist)
        
        masked_data["taskTime"] = data["taskTimeParts"][i]
        masked_data["taskCollision"] = data["taskCollisionParts"][i]
        masked_data["taskMistake"] = data["taskMistakeParts"][i]
        state = data["state"][st]
        masked_data["state"] = state
        masked_data["collision_flag"] = COLLISION_FLAG[data["mapId"]][i]
        masked_data["userId"] = userId
        masked_data["uiId"] = uiId
        masked_data["taskOrder"] = TASK_ORDER[userId][uiId]
        masked_data["label"] =  "{0:02d}-{1}".format(state, ROBOT_NUM[state]) + ("*" if masked_data["collision_flag"] else " ")
        subtask_data.append(masked_data)
    
    for sub in subtask_data:
        sub["warning_filter"] = mode_filter(sub["warning"], 5)
        
    with open(BDATA_SUBTASK.format(userId, uiId), "wb") as f:
        pickle.dump(subtask_data, f)
        
def all_convert_binary(new=False):
    for i in range(0, N):
        for j in range(0, 4):
            try:
                if new or not os.path.exists(BDATA.format(i, j)):
                    convert_binary(i, j)
                    print("Convert {0:02d}_{1:1d}".format(i, j))
                if new or not os.path.exists(BDATA_SUBTASK.format(i, j)):
                    split_subtask(i, j)
                    print("Split {0:02d}_{1:1d}".format(i, j))
            except Exception as e:
                print("Error {0:02d}_{1:1d}: {2}".format(i, j, e))
                traceback.print_exc()

def all_data_concat(new=False):
    all_convert_binary(new)
    data = []
    for i in range(0, N):
        userData = []
        for j in range(0, 4):
            try:
                subtask_data = load_subtask(i, j)
                userData.append(subtask_data)
            except Exception as e:
                print("Error {0:02d}_{1:1d}: {2}".format(i, j, e))
                traceback.print_exc()
        data.append(userData)

    data = np.array(data)
    with open("bdata/all.pkl", "wb") as f:
        pickle.dump(data, f)
        print("all.pkl saved")

def new(new=False):
    all_data_concat(new)
    convert_subject(True)
    convert_subject_attr(True)
    convert_subject_ranking(True)
    print("all data converted")

def convert_subject_attr(new=False):
    if not new and os.path.exists(os.path.join("bdata", "subject_attr.pkl")):
        print("subject_attr.pkl already exists")
        return
    
    data = defaultdict(list)
    
    with open(FSUBJECT_ATTR, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader) # a, b, c, d, e, userId, Faculty, Age, Gender, Exercise, Duration, Mobile, Console, VR, Genre, Sleep, Waking, Health, Correction, Vision, Refraction
    
        for row in reader:
            userId = int(row[5])
            data["userId"].append(userId)
            data["faculty"].append(row[6])
            data["age"].append(int(row[7]))
            if row[8] == "男性":
                data["gender"].append("m")
            elif row[8] == "女性":
                data["gender"].append("f")
            else:
                data["gender"].append("o")
        
            if row[9] == "0日":
                data["exercise"].append(0)
            elif row[9] == "1～3日":
                data["exercise"].append(1)
            elif row[9] == "4～6日":
                data["exercise"].append(2)
            else:
                data["exercise"].append(3)
            
            if row[10] == "0分":
                data["duration"].append(0)
            elif row[10] == "1分～59分":
                data["duration"].append(1)
            elif row[10] == "60分～119分":
                data["duration"].append(2)
            else:
                data["duration"].append(3)
                
            
            if row[11] == "0回":
                data["mobile"].append(0)
            elif row[11] == "1-9回":
                data["mobile"].append(1)
            else:
                data["mobile"].append(2)
            
            if row[12] == "0回":
                data["console"].append(0)
            elif row[12] == "1-9回":
                data["console"].append(1)
            else:
                data["console"].append(2)

            if row[13] == "0回":
                data["vr"].append(0)
            elif row[13] == "1-9回":
                data["vr"].append(1)
            else:
                data["vr"].append(2)
            
            if row[14] == "スマホ・タブレットやNintendo Switch（テーブル・携帯モード）などの小型画面ゲーム機":
                data["genre"].append("mobile")
            elif row[14] == "パソコン・コンシューマーゲーム機（Nintendo Switch TVモード / PlayStation など）などの大型画面ゲーム機":
                data["genre"].append("console")
            elif row[14] == "VRゲーム機":
                data["genre"].append("vr")
            else:
                data["genre"].append("none")
            
            data["sleep"].append(float(row[15]))
            data["waking"].append(float(row[16]))
            data["health"].append(int(row[17]))

            data["correction"].append(True if row[18] == "はい" else False)

            data["vision"].append(row[19])
            data["refraction"].append(row[20])

    os.makedirs("bdata", exist_ok=True)
    with open(os.path.join("bdata", "subject_attr.pkl"), "wb") as f:
        pickle.dump(pd.DataFrame(data), f)
        print("subject_attr.pkl saved")

def convert_subject_ranking(new=False):
    if not new and os.path.exists(os.path.join("bdata", "subject_ranking.pkl")):
        print("subject_ranking.pkl already exists")
        return
    
    data = defaultdict(list)
    
    with open(FSUBJECT_RANK, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader) # a, b, c, d, e, userId, easy, annoy, useful, trust, notice, distance, direction, safe, vr, avoid, comment

        for row in reader:
            userId = int(row[5])
            data["userId"].append(userId)
            data["easy"].append(toList(row[6]))
            data["annoy"].append(toList(row[7]))
            data["useful"].append(toList(row[8]))
            data["trust"].append(toList(row[9]))
            data["notice"].append(toList(row[10]))
            data["distance"].append(toList(row[11]))
            data["direction"].append(toList(row[12]))
            data["safe"].append(toList(row[13]))
            data["vr"].append(toList(row[14]))
            data["avoid"].append(toList(row[15]))
            data["comment"].append(row[16])

    for key in data.keys():
        data[key] = np.array(data[key])

    os.makedirs("bdata", exist_ok=True)
    with open(os.path.join("bdata", "subject_ranking.pkl"), "wb") as f:
        pickle.dump(data, f)
        print("subject_ranking.pkl saved")

def toList(txt):
    rank = txt.split(";")
    l = [0]*4
    for i, r in zip(range(1, 5), rank):
        if r == "矢印＋警告音":
            l[0] = i
        elif r == "矢印のみ":
            l[1] = i
        elif r == "警告音のみ":
            l[2] = i
        else:
            l[3] = i
    
    return l

# data load ********************
def load_binary(userId, uiId):
    filename = BDATA.format(userId, uiId)
    if not os.path.exists(filename):
        try:
            convert_binary(userId, uiId)
        except Exception as e:
            print("Error {0:02d}_{1:1d}: {2}".format(userId, uiId, e))
            return None
    
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def load_subtask(userId, uiId): # load taskData
    filename = BDATA_SUBTASK.format(userId, uiId)
    if not os.path.exists(filename):
        try:
            split_subtask(userId, uiId)
        except Exception as e:
            print("Error {0:02d}_{1:1d}: {2}".format(userId, uiId, e))
            return None
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print("Error {0:02d}_{1:1d}: {2}".format(userId, uiId, e))
        return None
    
    return data

def load_subject(): # load subjective data
    filename = os.path.join("bdata", "subject.pkl")
    if not os.path.exists(filename):
        convert_subject()
    
    with open(filename, "rb") as f:
        data = pickle.load(f)
    
    return data

def load_all(): # load all objective data
    filename = "bdata/all.pkl"
    if not os.path.exists(filename):
        all_data_concat()
    
    with open(filename, "rb") as f:
        data = pickle.load(f)
    
    return data

def load_subject_attr(): # load subjective attribute data
    filename = os.path.join("bdata", "subject_attr.pkl")
    if not os.path.exists(filename):
        convert_subject_attr()
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def load_subject_ranking(): # load subjective ranking data
    filename = os.path.join("bdata", "subject_ranking.pkl")
    if not os.path.exists(filename):
        convert_subject_ranking()
    
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def load_cognition():
    filename = os.path.join("bdata", "cognition_speed.pkl")
    with open(filename, "rb") as f:
        data = pickle.load(f)
    
    return data

# data convert ********************
"""
distance(a, b): calculate distance between two points in xz plane
"""
def distance(a, b): # a, b: numpy.array(shape=(n, 3))
    a *= [1, 0, 1]
    b *= [1, 0, 1]
    if len(a) == 3:
        a = np.array([a])
        b = np.array([b])
    return np.linalg.norm(np.array(a) - np.array(b), axis=1) # np.array(shape=(n, ))

"""
rot2vec(rot): convert rotation to vector
"""
def rot2vec(rot): # rot: numpy.array(shape=(n, 3))
    base_vec = [0, 0, 1]
    vec = R.from_euler("xyz", rot, degrees=True).apply(base_vec)
    return vec # np.array(shape=(n, 3))


def slide(data, n):
    if n <= 0:
        return data
    return np.concatenate([np.zeros(n), data[: -n]])


def mode_filter(arr, window_size):
    """
    1次元配列に対して移動窓の最頻値（モード）を計算してノイズを平滑化する

    Parameters:
    - arr: 入力の1次元np.array
    - window_size: ウィンドウのサイズ（奇数を推奨）

    Returns:
    - 平滑化された配列（np.array）
    """
    half = window_size // 2
    padded = np.pad(arr, (half, half), mode='edge')  # 端を拡張してパディング
    smoothed = np.empty_like(arr)

    for i in range(len(arr)):
        window = padded[i:i + window_size]
        smoothed[i] = mode(window, keepdims=False).mode

    return smoothed

def get_warning_mask(warning, window=5):
    masks = [ [ [] for _ in range(4)] for _ in range(4)]

    diff = np.diff(warning.astype(int))
    l = len(warning)

    for i, d in enumerate(diff):
        if d == 0:
            continue
        p = warning[i]
        n = warning[i+1]
        st = max(0, i-window+1)
        ed = min(l, i+window+1)

        mask = np.zeros(l, dtype=bool)
        mask[st:ed] = True
        masks[p][n].append(mask)

    return masks

def get_corr(data1, data2, axis=0):
    if axis == 1:
        data1 = np.array(data1).T
        data2 = np.array(data2).T
    
    result = []
    for d1, d2 in zip(data1, data2):
        if len(d1) == 0 or len(d2) == 0:
            result.append(np.nan)
            continue
        try:
            corr = spearmanr(d1, d2).statistic
        except ValueError as e:
            print(f"ValueError in get_corr: {e}")
            corr = np.nan
        except Warning as e:
            print(f"Warning in get_corr: {e}")
            corr = np.nan
        result.append(corr)
    
    result = np.array(result)
    return result

def ND_test(data):
    try:
        if len(data) <= 5000:
            statistic, p_value = shapiro(data)
        else:
            statistic, p_value = normaltest(data)
    except UserWarning as e:
        print(f"Warning in ND_test: {e}")
        return 0, 1
    except RuntimeWarning as e:
        print(f"RuntimeWarning in ND_test: {e}")
        return 0, 1
    except Warning as e:
        print(f"Warning in ND_test: {e}")
        return 0, 1

    return statistic, p_value

    if p_value < 0.05:
        return False  # データは正規分布に従わない
    else:
        return True  # データは正規分布に従う

def samples_test_rel(data1, data2, n_parametric=False):
    alpha = 0.05

    try:
        if not n_parametric and ND_test(data1) and ND_test(data2):
            print("parametric test")
            statistic, p_value = ttest_rel(data1, data2)
            r = common_language_effect_rel(data1, data2)
        else:
            statistic, p_value = wilcoxon(data1, data2)
            r = common_language_effect_rel(data1, data2)
    except UserWarning as e:
        print(f"Warning in samples_test_ind: {e}")
        statistic, p_value = 0, 1
        r = -1
    except RuntimeWarning as e:
        # print(f"RuntimeWarning in samples_test_ind: {e}")
        statistic, p_value = 0, 1
        r = -1
    except Warning as e:
        print(f"Warning in samples_test_ind: {e}")
        statistic, p_value = 0, 1
        r = -1

    return statistic, p_value, p_value < alpha, r

def samples_test_ind(data1, data2, n_parametric=False):
    alpha = 0.05

    try:
        if not n_parametric and ND_test(data1) and ND_test(data2):
            print("parametric test")
            statistic, p_value = ttest_ind(data1, data2)
            r = common_language_effect_ind(data1, data2)
        else:
            statistic, p_value = mannwhitneyu(data1, data2)
            r = common_language_effect_ind(data1, data2)
    except UserWarning as e:
        print(f"Warning in samples_test_ind: {e}")
        statistic, p_value = 0, 1
        r = -1
    except RuntimeWarning as e:
        # print(f"RuntimeWarning in samples_test_ind: {e}")
        statistic, p_value = 0, 1
        r = -1
    except Warning as e:
        print(f"Warning in samples_test_ind: {e}")
        statistic, p_value = 0, 1
        r = -1

    return statistic, p_value, p_value < alpha, r

def samples_test_rel_list(datas, n_parametric=False):
    results = [[False]*(len(datas)) for _ in range(len(datas))]
    for i in range(len(datas)):
        for j in range(i + 1, len(datas)):
            result = samples_test_rel(datas[i], datas[j], n_parametric=n_parametric)
            results[i][j] = (result[2], result[3], result[1], result[0])
    return results

def samples_test_ind_list(datas, n_parametric=False):
    results = [[False]*(len(datas)) for _ in range(len(datas))]
    for i in range(len(datas)):
        for j in range(i + 1, len(datas)):
            result = samples_test_ind(datas[i], datas[j], n_parametric=n_parametric)
            results[i][j] = (result[2], result[3], result[1], result[0])
    return results

def cliff_delta(data1, data2):
    """
    Cliff's Deltaを計算する関数
    data1, data2: 比較する2つのデータセット（numpy array）
    """
    if len(data1) == 0 or len(data2) == 0:
        return np.nan
    
    if isinstance(data1, list):
        data1 = np.array(data1)
    if isinstance(data2, list):
        data2 = np.array(data2)
    
    n1 = len(data1)
    n2 = len(data2)
    
    # count = 0
    # for x in data1:
    #     for y in data2:
    #         if x > y:
    #             count += 1
    #         elif x < y:
    #             count -= 1
    count = np.sum(data1[:, None] > data2) - np.sum(data1[:, None] < data2)

    delta = count / (n1 * n2)
    return delta

def common_language_effect_ind(data1, data2):
    """
    Common Language Effect (CLE)を計算する関数
    data1, data2: 比較する2つのデータセット（numpy array）
    """
    if len(data1) == 0 or len(data2) == 0:
        return np.nan
    
    if isinstance(data1, list):
        data1 = np.array(data1)
    if isinstance(data2, list):
        data2 = np.array(data2)
    
    n1 = len(data1)
    n2 = len(data2)
    
    # count = 0
    # for x in data1:
    #     for y in data2:
    #         if x > y:
    #             count += 1
    #         elif x == y:
    #             count += 0.5
    count = np.sum(data1[:, None] > data2) + 0.5 * np.sum(data1[:, None] == data2)
    
    cle = count / (n1 * n2)
    # print(cle)
    return cle

def common_language_effect_rel(data1, data2):
    """
    Common Language Effect (CLE)を計算する関数
    data1, data2: 比較する2つのデータセット（numpy array）
    """
    if len(data1) == 0 or len(data2) == 0:
        return np.nan
    
    if isinstance(data1, list):
        data1 = np.array(data1)
    if isinstance(data2, list):
        data2 = np.array(data2)
    
    difference_scores = data1-data2

    # 差が0より大きい、または0に等しいケースをカウント
    greater_than_zero = np.sum(difference_scores > 0)
    equal_to_zero = np.sum(difference_scores == 0)

    # CLS for paired data
    # P(D > 0) + 0.5 * P(D = 0) の式
    cle = (greater_than_zero + 0.5 * equal_to_zero) / len(difference_scores)

    return cle

def friedman_test(data):
    statistic, pvalue = friedmanchisquare(*data)
    print(f"Friedman test statistic: {statistic}, p-value: {pvalue}")

    UI = ["AV", "VO", "AO", "NO"]
    res = samples_test_rel_list(data, n_parametric=True)
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if res[i][j][0]:
                print(f"{UI[i]}-{UI[j]}({res[i][j][2]}), ", end="")
    print()