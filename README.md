# Multimodal User Interface Program

dataをこのフォルダに保存しておく

```text
mui
├ data
│ ├ 03
│ │ ├ 03_0_gaze.csv
│ │ ├ 03_0_player.csv
│ │ ├ 03_0_robot.csv
│ │ ├ 03_0_taskinfo.json
│ │ └ ...
│ ├ 04
│ │ ├ 04_0_gaze.csv
│ │ └ ...
│ ├ ...
│ └ subject.csv
├ requirements.txt
└ ...
```

## 初期化

テキストデータをバイナリデータに変換

```bash
 pip install -r requirements.txt
 python init.py
```

## データの取得

```python
from lib import *

allData = load_all() # np.array(7, 4, 9)
subjectData = load_subject() # dict(key: np.array(4,4))
```

例

- ユーザー3、ui-0の1つ目のサブタスクのカメラ位置の配列
- ユーザー4、ui-1の5つ目のサブタスクのタスク時間

```python
pos = allData[3, 0, 0]["pos"] # np.array(frameNum, 3)
taskTime = allData[4, 0, 4]["taskTime"] # float
```

## データ構造

### 客観的データ

データはサブタスクごとに区切っているため、(userId, uiId, subtaskId) の3次元配列

```python
allData = [
    [nullData, nullData, nullData, nullData], # userId 0: null
    [nullData, nullData, nullData, nullData], # userId 1: null
    [nullData, nullData, nullData, nullData], # userId 2: null
    [taskData-0, taskData-1, taskData-2, taskData-3], # userId 3
    [taskData-0, taskData-1, taskData-2, taskData-3], # userId 4
    ...
]: np.array(list) | shape(UserNUM, 4) # [userId, uiId, subtaskId]

taskData = [
    subtaskData-0: dict(),
    subtaskData-1: dict(),
    ...
    subtaskData-8: dict()
] : np.array(dict()) | shape(9, )

nullData = [None] * 9 : np.array() | shape(9, )

subtaskData = {
    "time": np.array(float) | shape(frameNum, ),

    "pos": np.array(float) | shape(frameNum, 3), # camera position
    "rot": np.array(float) | shape(frameNum, 3), # camera rotation
    "lg_pos": np.array(float) | shape(frameNum, 3), # left eye global position
    "lg_rot": np.array(float) | shape(frameNum, 3), # left eye rotation
    "rg_pos": np.array(float) | shape(frameNum, 3), # right eye position
    "rg_rot": np.array(float) | shape(frameNum, 3), # right eye rotation
    
    "obj": np.array(str) | shape(frameNum, ), # gaze object name
    "bpm": np.array(int) | shape(frameNum, ), # heart rate
    "trigger": np.array(bool) | shape(frameNum, ), # button pressed
    "subTask": np.array(bool) | shape(frameNum, ), # subtask
    "warning": np.array(int) | shape(frameNum, ), # warning robot count
    "collision": np.array(bool) | shape(frameNum, ), # collision

    "robot": [robotData] * 6, # robot data
    "robot_cnt": np.array(int) | shape(frameNum, ), # moving robot count

    "userId": int, # user id
    "uiId": int # ui id
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
```

ユーザーidとallDataのインデックス番号を合わせるため、allData[0]-allData[2]は空のオブジェクトが入っている

### 主観的データ

```python
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

    "mental": np.array(int) | shape(userNum, 4), # NASA-TLX mental load (0-100)
    "physical": np.array(int) | shape(userNum, 4), # NASA-TLX physical load (0-100)
    "temporal": np.array(int) | shape(userNum, 4), # NASA-TLX temporal load (0-100)
    "performance": np.array(int) | shape(userNum, 4), # NASA-TLX performance load (0-100)
    "effort": np.array(int) | shape(userNum, 4), # NASA-TLX effort load (0-100)
    "frustration": np.array(int) | shape(userNum, 4), # NASA-TLX frustration load (0-100)
    "score": np.array(float) | shape(userNum, 4), # NASA-TLX score (0-100)

    "comment": np.array(str) | shape(userNum, 4) # comment
}
```

## その他ファイル

基本的に自分以外が見ることを想定していないです…

- graph.py
    - barh_plot_add：視線データのラベル割合
    - robot_distance：ロボットとの最短距離のグラフ
    - rot_y：カメラのy軸の回転
    - rot_y_diff：カメラのy軸の回転の絶対変化量（1フレーム）
    - rot_y_diff_n：カメラのy軸の回転の絶対変化量（nフレーム）
    - pos_xz_diff_n：カメラの移動の絶対変化量（nフレーム）
    - twin_xz_roty_diff_n：カメラの移動とy軸の回転の絶対変化量の複合グラフ（nフレーム）
    - pos_xz：カメラの移動パス
- subjective.py
    - box_plot：指定したキーの箱ひげ図
    - scatter_plot：指定した2つのキーの相関図
- server.py, index.html：リプレイ機能・作成中
