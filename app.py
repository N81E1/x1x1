#函式區
#   1. 載入函式庫

from flask import Flask, request, abort

from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent
)
###########################################################################################
import mediapipe as mp  #mediapipe：Google 提供的 AI 偵測工具，可用於手勢辨識
import cv2  #opencv-python：用於處理影像與視訊流
import time #time:提供處理時間的方法，除了可以取得目前的時間或轉換時間，也能夠透過像是 sleep() 的方法將程式暫停 
#from linebot import LineBotApi    #line-bot-sdk：用於發送 Line Bot 訊息與圖片通知。
from linebot.models import TextSendMessage, ImageSendMessage

import os   #os:操作系統中檔案的方法
#           SUPERVISON_DEPRECATION_WARNING:這個環境變數是針對supervision庫的，設定它的值為'0'表示禁用棄用警告。
os.environ['SUPERVISON_DEPRECATION_WARNING'] = '0'  

import supervision as sv    #Supervision：物件追蹤、畫框等輔助工具。

import numpy as np  #Numpy：數據運算庫，方便處理影像資料。
import ultralytics  #ultralytics：YOLOv8 物件偵測與追蹤工具
import sys    #sys:提供了一系列與 Python 解釋器和系統相關的函式和變數
import requests    #requests：用於發送 API 請求（如 Imgur 圖片上傳）。
from datetime import datetime    #datetime:提供不少處理日期和時間的方法，可以取得目前的日期或時間，並進一步進行相關的運算
import threading    #threading:採用「執行緒」的方式，運用多個執行緒，在同一時間內處理多個任務 ( 非同步 )

import pygame
#(不可用)from playsound import playsound    #playsound：用來播放 MP3/WAV 音效。

app = Flask(__name__)

# 定義主目錄並列印它
HOME = os.getcwd()
print(HOME)

# 來源相機索引
SOURCE_CAMERA_INDEX = 0

# 對 Ultralytics 進行初步檢查
ultralytics.checks()

# 印刷監督版
print("supervision.__version__:", sv.__version__)

#   YOLO 模型初始化
#   模型檔案路徑
MODEL = "best.pt"

# 載入YOLO模型
from ultralytics import YOLO
model = YOLO(MODEL)
model.fuse()

# 將 class_id 對應到 class_name
CLASS_NAMES_DICT = model.model.names    # 類別名稱對應

# 感興趣的類別 ID
selected_classes = [0, 1, 2, 3, 4, 5, 6, 7]  # 追蹤的類別

# 建立 BoxAnnotator 實例
box_annotator = sv.BoxAnnotator(thickness=4)  # 註釋器

#   線路位置
#   定義兩條垂直的監測線，可能用於計算物件進入或離開的次數：

LINE_START = sv.Point(200, 0)  # 更改為所需的起點位置
LINE_END = sv.Point(200, 800)  # 更改為所需的終點位置

LINE_START_2 = sv.Point(1000, 0)  # 更改為所需的起點位置
LINE_END_2 = sv.Point(1000, 800)  # 更改為所需的終點位置

# 建立 LineZone 實例
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)
line_zone_2 = sv.LineZone(start=LINE_START_2, end=LINE_END_2)

# 建立註釋器
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)
line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)
line_zone_2_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)


current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

#TARGET_VIDEO_PATH = f"C:/yolov5/tracker/results/video_{current_time}.mp4"

# 截圖保存目錄
SCREENSHOT_DIR = "C:\jt"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# 創建 BYTETracker 實例
byte_tracker = sv.ByteTrack()


# 指示我們是否在 3600 幀之後的標誌
after_3600_frames = False

# 截圖計數器
screenshot_counter = 0
MAX_SCREENSHOTS = 10

# 定義全域幀計數器
frame_counter = 0
error_timer = 0  # 透過「6」類偵測追蹤連續時間的計時器

# 定義 Imgur 客戶端 ID
IMGUR_CLIENT_ID = '44338750d0cf7bd'

# 定義 Line Bot Token 和 User ID
LINE_BOT_TOKEN = Configuration(access_token=os.getenv('CHANNEL_ACCESS_TOKEN'))
LINE_USER_ID =WebhookHandler(os.getenv('CHANNEL_SECRET'))
#line_bot_api = LineBotApi(LINE_BOT_TOKEN)

# 初始化 Mediapipe 的手部模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

hand_features_dict = {}  # 儲存手部特徵

@app.route('/')
def home():
    return"X1X1"

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        LINE_USER_ID.handle(body, signature)
    except InvalidSignatureError:
        app.logger.info("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'

def extract_hand_features(hand_landmarks):
    """提取手部特徵"""
    features = []
    for landmark in hand_landmarks.landmark:
        features.append((landmark.x, landmark.y, landmark.z))  # 儲存 x, y, z 座標
    return np.array(features)

def identify_hand(hand_features):
    """識別手部"""
    for person_id, features in hand_features_dict.items():
        if np.allclose(features, hand_features, atol=0.1):  # 使用 numpy 的 allclose 方法進行比較
            return person_id
    return None  # 如果沒有匹配，返回 None

def is_hand_open(hand_landmarks, imgWidth, imgHeight):
    #   判斷手掌是否張開
    #   取得五根手指的指尖座標
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    #   轉換為圖像座標
    thumb_tip = (int(thumb_tip.x * imgWidth), int(thumb_tip.y * imgHeight))
    index_tip = (int(index_tip.x * imgWidth), int(index_tip.y * imgHeight))
    middle_tip = (int(middle_tip.x * imgWidth), int(middle_tip.y * imgHeight))
    ring_tip = (int(ring_tip.x * imgWidth), int(ring_tip.y * imgHeight))
    pinky_tip = (int(pinky_tip.x * imgWidth), int(pinky_tip.y * imgHeight))

    threshold = 0.03 * imgWidth  # 設定閾值
    return (np.linalg.norm(np.array(thumb_tip) - np.array(index_tip)) > threshold and
            np.linalg.norm(np.array(thumb_tip) - np.array(middle_tip)) > threshold and
            np.linalg.norm(np.array(thumb_tip) - np.array(ring_tip)) > threshold and
            np.linalg.norm(np.array(thumb_tip) - np.array(pinky_tip)) > threshold and
            np.linalg.norm(np.array(index_tip) - np.array(middle_tip)) > threshold and
            np.linalg.norm(np.array(middle_tip) - np.array(ring_tip)) > threshold and
            np.linalg.norm(np.array(ring_tip) - np.array(pinky_tip)) > threshold)


    #播放音頻的函數
def play_sound_in_thread(file_path):
    play_sound_in_thread(file_path)

# 定義上傳圖片到 Imgur 的函數
def upload_to_imgur(image_path):
    headers = {
        'Authorization': f'Client-ID {IMGUR_CLIENT_ID}',
    }
    with open(image_path, 'rb') as image_file:
        response = requests.post('https://api.imgur.com/3/image', headers=headers, files={'image': image_file})
    if response.status_code == 200:
        return response.json()['data']['link']
    else:
        print(f"Imgur 上傳失敗: {response.status_code} {response.text}")
        return None

# 定義發送消息到 Line Bot 的函數
def send_line_message(message):
    LINE_BOT_TOKEN.push_message(LINE_USER_ID, TextSendMessage(text=message))

#   這兩個函式用於傳送文字訊息和圖片至 LINE Bot：
def broadcast_message(Tmessage):
    print(f"輸入的訊息是: {Tmessage}")
    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {
        "Authorization": f"Bearer {LINE_BOT_TOKEN}",
        "Content-Type": "application/json"
    }

    body = {
        "messages":[
            {
                "type": "text",
                "text": Tmessage
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers , json=body)
        response.raise_for_status()
        print(f"消息廣播成功 回應: {response.json()}")

    except requests.exceptions.RequestException as e:
        print(f"無法廣播: {e}")
        if e.response:
            print(f"回應內容: {e.response.text}")



# 定義傳圖片到 line 的函數
def send_image_to_line(imgur_link, line_user_id):
    try:
        image_message = ImageSendMessage(
            original_content_url=imgur_link,
            preview_image_url=imgur_link
        )
        LINE_BOT_TOKEN.push_message(line_user_id, image_message)
    except Exception as e:
        print(f"發送 Line 圖片失敗: {e}")

def broadcast_img(imgur_link):
    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {
        "Authorization": f"Bearer {LINE_BOT_TOKEN}",
        "Content-Type": "application/json"
    }
    body = {
        "messages":[
            {
                "type": "image",
                "originalContentUrl": imgur_link,
                "previewImageUrl":imgur_link
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers , json=body)
        response.raise_for_status()
        print(f"消息廣播成功 回應: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"無法廣播: {e}")
        if e.response:
            print(f"回應內容: {e.response.text}")
            
#上半
##############################################################################################################################################
#下半

# 定義重啟函數
def restart_program():
    
    #   重新啟動當前程序  
    python = sys.executable
    os.execv(python, [python] + sys.argv)

#第二個程式
def run_second_program(cap):
    #   在這個函數中，會進行物體檢測、追蹤、截圖、上傳圖片等操作。                          
    #   啟動音頻播放線程

    #   音效檔案的路徑
    file_path = r'start.mp3'
    print("音效播放成功！")

    if os.path.exists(file_path):
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    else:
        print("檔案不存在！請檢查路徑。")

    def calculate_brightness(image):    

        # 將影像轉換為灰階
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     

        # 計算平均亮度
        average_brightness = np.mean(gray_image)        
        return average_brightness
    
    # 定義清除垂直線的函數
    def clear_vertical_lines():
        global line_zone, line_zone_2

        # 將線條設定到螢幕外的位置
        LINE_S = sv.Point(200, 200)
        LINE_E = sv.Point(200, 1080)

        LINE_S_2 = sv.Point(200, 200)
        LINE_E_2 = sv.Point(200, 1080)
        
        line_zone = sv.LineZone(start=LINE_S, end=LINE_E)
        line_zone_2 = sv.LineZone(start=LINE_S_2, end=LINE_E_2)
        
    # 定義視訊處理中使用的回調函數
    error_timer = 0  # 透過「6」類偵測追蹤連續時間的計時器
    
    # 定義視訊處理中使用的回調函數
    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        global after_3600_frames, error_timer, screenshot_counter
        global line_zone, line_zone_2
        global frame_counter  # 宣告全域變數

        # 初始化帶註解的框架
        annotated_frame = frame.copy()
        
        # 單幀模型預測並轉換為監督檢測
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        #程式會偵測特定物件（如類別 6），如果在 30 秒內持續偵測到錯誤，則進行通知。
        if 6 in detections.class_id:
            error_timer += 1
        else:
            error_timer = 0
        
        # 30 秒持續偵測到錯誤
        if error_timer >= 900:
            #clear_vertical_lines()  # 呼叫函數清除垂直線
            after_3600_frames = True  # 設置標誌
            print("還沒解決錯誤")
        
        #根據我們是否在 3600 幀之後選擇檢測
        if after_3600_frames:
            detections = detections[np.isin(detections.class_id, [0, 2, 4, 5])]
        else:
            detections = detections[~np.isin(detections.class_id, [1])]
    
        # 使用 BYTETracker 更新偵測
                    
        detections = byte_tracker.update_with_detections(detections)
        labels = [
            f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for confidence, class_id, tracker_id
            in zip(detections.confidence, detections.class_id, detections.tracker_id)
        ]
        
        # 註釋框架
        annotated_frame = bounding_box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    
        # 更新線路區域
        line_zone.trigger(detections)
        line_zone_2.trigger(detections)
        
        # 檢查錯誤並進行對應註釋
        if line_zone.in_count > 0 or line_zone_2.in_count > 0:
            cv2.putText(annotated_frame, "Error", (9, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)

            # 當“Error”被檢測到時，並且截圖計數器小於最大值，保存截圖
            if screenshot_counter < MAX_SCREENSHOTS:
                screenshot_filename = os.path.join(SCREENSHOT_DIR, f"error_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                cv2.imwrite(screenshot_filename, annotated_frame)
                screenshot_counter += 1
                
                # 上傳截圖到 Imgur
                imgur_link = upload_to_imgur(screenshot_filename)
                if imgur_link:
                    print(f"截圖上傳到 Imgur: {imgur_link}")

                    # 通過 Line Bot 發送消息
                    broadcast_img(imgur_link)
                    time.sleep(3)
                    wroug_message = "檢測到錯誤"
                    broadcast_message(wroug_message)
                
            # 處理錯誤後重置計數器
            line_zone.in_count = 0
            line_zone_2.in_count = 0
    
        elif after_3600_frames and any(detections):
            cv2.putText(annotated_frame, "Error", (9, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)

            # 當“Error”被檢測到時，並且截圖計數器小於最大值，保存截圖
            screenshot_filename = os.path.join(SCREENSHOT_DIR, f"error_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            cv2.imwrite(screenshot_filename, annotated_frame)
                
            # 上傳截圖到 Imgur
            #imgur_link = upload_to_imgur(screenshot_filename)
            #if imgur_link:
                #print(f"截圖上傳到 Imgur: {imgur_link}")
                
                #通過 Line Bot 發送消息
                #broadcast_img(imgur_link)
        
        # 用線條區域註釋框架
        annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)
        annotated_frame = line_zone_2_annotator.annotate(annotated_frame, line_counter=line_zone_2)
        
        # 計算亮度並檢查是否小於或等於50
        brightness = calculate_brightness(frame)
        #print(f"Frame {index}: Brightness: {brightness:.2f}")
        if brightness <= 10:
            #playsound('C:/yolov5/tracker/end.mp3')
            print("Brightness is too low. Restarting the program...")
            restart_program()  # 重新啟動腳本
        
        return annotated_frame
    
    # 處理來自相機的視訊幀
    def process_camera():
        #cap = cv2.VideoCapture(SOURCE_CAMERA_INDEX , cv2.CAP_MSMF)
        
        #設定框架的寬度和高度
        FRAME_WIDTH = 1920  # 設定所需的寬度
        FRAME_HEIGHT = 1080  # 設定所需的高度
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        frame_index = 0
        
        # 建立一個正常大小的視窗並設定初始大小
        cv2.namedWindow('Annotated Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Annotated Frame', FRAME_WIDTH, FRAME_HEIGHT)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame = callback(frame, frame_index)
            cv2.imshow('Annotated Frame', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
            
            frame_index += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
    process_camera()

def main():
    #主函數中，首先啟動攝影機，並進行手部檢測。如果檢測到兩隻手張開，則進入第二個程序進行物體檢測和追蹤。
    
    #修正羅技攝影機
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
    
    cap = cv2.VideoCapture(SOURCE_CAMERA_INDEX , cv2.CAP_MSMF)
    start_time = None  # 初始化計時器的開始時間
    ok_duration = 0  # 初始化 OK 狀態的持續時間
    #print(cv2.getBuildInformation())
    
    FRAME_WIDTH = 2000  # 設定所需的寬度
    FRAME_HEIGHT = 1100  # 設定所需的高度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("錯誤")
            break
        

        # 轉換顏色空間
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgHeight, imgWidth, _ = frame.shape
        results = hands.process(image_rgb)

        open_hands = 0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if is_hand_open(hand_landmarks, imgWidth, imgHeight):
                    open_hands += 1

                    # 提取手部特徵
                    hand_features = extract_hand_features(hand_landmarks)
                    person_id = identify_hand(hand_features)
                    if person_id is not None:
                        cv2.putText(frame, f"Person ID: {person_id}", (10, 90 + open_hands * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    else:
                        new_person_id = len(hand_features_dict) + 1
                        hand_features_dict[new_person_id] = hand_features
                        cv2.putText(frame, f"New Person ID: {new_person_id}", (10, 90 + open_hands * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # 繪製手部關鍵點和連接線
                mp_drawing.draw_landmarks(frame, hand_landmarks, 
                                          mp_hands.HAND_CONNECTIONS
                                          )

        # 顯示結果
        if open_hands >= 2:
            cv2.putText(frame, "OK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if start_time is None:
                start_time = time.time()  # 開始計時
            else:
                ok_duration = time.time() - start_time  # 計算持續時間
            cv2.putText(frame, f"Duration: {ok_duration:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 如果檢測到 "OK" 的持續時間達到 5 秒，則跳出循環
            if ok_duration >= 5:
                #cap.release()
                #cv2.destroyAllWindows()
                run_second_program(cap)  # 調用第二個程序
                break
        else:
            cv2.putText(frame, "Hand Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            start_time = None  # 重置計時器

        cv2.imshow('Hand Detection', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # 按下 ESC 鍵退出
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
