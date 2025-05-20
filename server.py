# server.py
import http.server
import socketserver
import json
from lib import *
from urllib.parse import urlparse, parse_qs

PORT = 8000

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/data'):
            # APIルートは許可
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            # --- ここで動的にデータ生成・送信 ---
            query = urlparse(self.path).query
            params = parse_qs(query)
            userId = int(params.get('userId', [0])[0])
            uiId = int(params.get('uiId', [0])[0])
            subtaskId = int(params.get('subtaskId', [0])[0])
            
            self.wfile.write(json.dumps(create_frame(userId, uiId, subtaskId)).encode())

            return

        requested_path = self.path.lstrip('/')
        if requested_path == '' or requested_path == 'index.html':
            # index.html か / の場合のみ許可
            return super().do_GET()
        else:
            self.send_error(403, "Access to this file is forbidden")
            
    
    def end_headers(self):
        # CORS許可：ローカルブラウザ動作のために必要な場合あり
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

def create_frame(userId, uiId, subtaskId):
    # ユーザID、UI-ID、タスクIDに基づいてデータを取得
    data = allData[userId][uiId][subtaskId]
    pos = data["pos"]
    rot = data["rot"]
    time = data["time"]

    # 2Dベクトルに変換
    base_vec = [0, 0, 1]
    vec = [R.from_euler("xyz", r, degrees=True).apply(base_vec) for r in rot]
    frames = [{"x": p[0], "z": p[2], "vx": v[0], "vz": v[2], "time": t} for p, v, t in zip(pos, vec, time)]
    
    return frames

with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
    allData = load_all()
    print(f"Serving at http://localhost:{PORT}")
    httpd.serve_forever()
