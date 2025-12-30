import customtkinter as ctk
import os
import subprocess
import sys
import threading
import time
import cv2
import numpy as np
from tkinter import filedialog, messagebox
from send2trash import send2trash
import concurrent.futures
import multiprocessing
import webbrowser # 用於打開下載頁面

# --- 線上更新依賴 ---
try:
    import requests
    from packaging import version as pkg_version # 用於精準比對版本號
    HAS_NETWORK = True
except ImportError:
    HAS_NETWORK = False

# ==================== PyTorch 支援檢測 ====================
try:
    import torch
    import torch.nn.functional as F
    from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
    from PIL import Image as PILImage
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    PILImage = None

# ==================== pHash 支援檢測 ====================
try:
    import imagehash
    from PIL import Image as PILImageHash
    HAS_PHASH = True
except ImportError:
    HAS_PHASH = False
    imagehash = None
    PILImageHash = None

# --- 設定主題 ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# ==========================================
# ⚙️ 設定區 (請修改這裡)
# ==========================================
APP_VERSION = "28.0" # 當前程式版本

# 請將這裡換成您的 GitHub 帳號與倉庫名稱
# 例如: https://github.com/YourName/CleanerAI
GITHUB_USER = "Riridesu" 
GITHUB_REPO = "CleanerAI"

# 構建 API URL (通常不需要改)
UPDATE_API_URL = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/releases/latest"
DOWNLOAD_PAGE_URL = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/releases/latest"

# ==========================================
# 全域變數 & 模型
# ==========================================
ai_model = None
img_transforms = None
image_loader_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def load_ai_model():
    global ai_model, img_transforms
    if ai_model is None and HAS_TORCH:
        print("正在載入 EfficientNet-B3 模型...")
        try:
            weights = EfficientNet_B3_Weights.DEFAULT
            ai_model = efficientnet_b3(weights=weights)
            ai_model.eval()
            img_transforms = weights.transforms()
            if torch.cuda.is_available():
                ai_model = ai_model.cuda()
            print("AI 模型載入完成")
        except Exception as e:
            print(f"模型載入失敗: {e}")
            ai_model = None

# ==========================================
# 工具函式
# ==========================================
def path_fix(path):
    try:
        path = os.path.abspath(path)
        path = path.replace('/', '\\')
        if os.name == 'nt':
            if len(path) > 240 and not path.startswith('\\\\?\\'):
                path = '\\\\?\\' + path
        return path
    except:
        return path

# ==========================================
# 線上更新邏輯
# ==========================================
def check_for_updates_thread(callback_found, callback_error):
    """ 在背景執行緒檢查更新 """
    if not HAS_NETWORK:
        callback_error("Missing 'requests' library")
        return

    try:
        print(f"Checking updates from: {UPDATE_API_URL}")
        response = requests.get(UPDATE_API_URL, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            latest_tag = data.get("tag_name", "").strip().lstrip("vV") # 去除 v 前綴
            
            # 使用 packaging.version 進行嚴謹比對 (例如 28.1 > 28.0)
            if pkg_version.parse(latest_tag) > pkg_version.parse(APP_VERSION):
                release_notes = data.get("body", "No release notes.")
                callback_found(latest_tag, release_notes)
            else:
                print("目前已是最新版本。")
        else:
            print(f"檢查更新失敗，狀態碼: {response.status_code}")
    except Exception as e:
        print(f"檢查更新發生錯誤: {e}")
        # 靜默失敗，不打擾使用者，除非是手動檢查

# ==========================================
# 核心運算
# ==========================================
def calculate_phash_task(file_path):
    if not HAS_PHASH: return None, file_path, "SKIP"
    try:
        stream = np.fromfile(file_path, dtype=np.uint8)
        cv_img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        if cv_img is None: return None, file_path, "CORRUPT"
        
        cv_img_small = cv2.resize(cv_img, (64, 64), interpolation=cv2.INTER_AREA)
        cv_img_rgb = cv2.cvtColor(cv_img_small, cv2.COLOR_BGR2RGB)
        pil_img = PILImageHash.fromarray(cv_img_rgb)
        h = imagehash.phash(pil_img)
        
        del cv_img, cv_img_small, cv_img_rgb, pil_img
        return h, file_path, "OK"
    except Exception:
        return None, file_path, "CORRUPT"

def calculate_ai_vector_single(file_path):
    global ai_model, img_transforms
    if ai_model is None or img_transforms is None: return None, file_path, "SKIP"
    try:
        stream = np.fromfile(file_path, dtype=np.uint8)
        cv_img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        if cv_img is None: return None, file_path, "CORRUPT"
        
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(cv_img)
        
        input_tensor = img_transforms(pil_img).unsqueeze(0)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        with torch.no_grad():
            output = ai_model(input_tensor)
            vector = output[0].cpu().numpy()
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
                
        del cv_img, pil_img, input_tensor
        return vector, file_path, "OK"
    except Exception:
        return None, file_path, "CORRUPT"

# ==========================================
# 資料結構
# ==========================================
class ResultItem:
    def __init__(self, dup_path, org_path, similarity):
        self.dup_path = dup_path
        self.org_path = org_path
        self.similarity = similarity
        self.checked = False

class CorruptItem:
    def __init__(self, path):
        self.path = path
        self.checked = False

# ==========================================
# UI 翻譯與文字
# ==========================================
TRANSLATIONS = {
    "zh_TW": {
        "app_title": f"AI 圖片清理工具 V{APP_VERSION} (線上更新版)",
        "sidebar_title": "CLEANER AI",
        "lbl_lang": "語言 (Language)",
        "btn_folder": "1. 選擇資料夾",
        "lbl_mode": "掃描模式",
        "mode_smart": "智慧掃描 (推薦)",
        "mode_ai": "純 AI 模式 (精準)",
        "mode_phash": "純 pHash (極速)",
        "lbl_threshold": "相似度門檻: {}%",
        "btn_scan": "2. 開始掃描",
        "btn_stop": "⛔ 停止",
        "btn_clear": "🧹 清空結果",
        "lbl_tools": "批次工具",
        "btn_smart": "✨ 全選當前列表",
        "btn_deselect": "取消全選",
        "btn_delete": "3. 刪除勾選項目 (Del)",
        "btn_exit": "退出程式",
        "btn_update": "檢查更新",
        "delete_original_label": "同時刪除原圖 (危險)",
        "status_ready": "就緒 - 請選擇資料夾",
        "status_loading": "正在載入檔案列表...",
        "status_phase1": "階段 1/2: 極速比對中 (pHash)... 進度 {} 張",
        "status_phase2": "階段 2/2: AI 深度分析中... 進度 {} 張",
        "status_scanning": "掃描中... 已分析 {} 張",
        "status_stopping": "正在停止... (完成當前批次後)",
        "status_stopped": "已手動停止",
        "status_done": "掃描完成！共發現 {} 重複，{} 損毀",
        "status_nodup": "掃描完成！未發現重複或損毀圖片",
        "status_resuming": "偵測到上次中斷，正在從第 {} 張繼續...",
        "status_check_update": "正在檢查更新...",
        "btn_load_more": "⬇ 載入更多",
        "msg_success": "成功刪除 {} 個檔案",
        "msg_confirm": "確定要刪除 {} 個檔案嗎？",
        "msg_warn_title": "危險操作確認",
        "msg_warn_content": "警告：您勾選了「同時刪除原圖」！\n\n這將會刪除 {count} 組圖片的「原圖」與「重複圖」。\n此操作無法透過本程式復原。\n\n確定要繼續嗎？",
        "msg_warn_corrupt": "確定要刪除 {count} 個損毀檔案嗎？",
        "msg_new_version": "發現新版本 V{}！\n\n更新內容：\n{}\n\n是否前往下載頁面？",
        "tab_dup": "重複圖片 ({})",
        "tab_bad": "損毀檔案 ({})",
        "col_corrupt": "❌ 檔案損毀",
        "btn_del": "刪除"
    },
    "en_US": {
        "app_title": f"AI Image Cleaner V{APP_VERSION} (Online Update)",
        "sidebar_title": "CLEANER AI",
        "lbl_lang": "Language",
        "btn_folder": "1. Select Folder",
        "lbl_mode": "Scan Mode",
        "mode_smart": "Smart Scan (Rec.)",
        "mode_ai": "AI Only (Precise)",
        "mode_phash": "pHash Only (Speed)",
        "lbl_threshold": "Threshold: {}%",
        "btn_scan": "2. Start Scan",
        "btn_stop": "⛔ Stop",
        "btn_clear": "🧹 Clear Results",
        "lbl_tools": "Batch Tools",
        "btn_smart": "✨ Select All",
        "btn_deselect": "Deselect All",
        "btn_delete": "3. Delete Selected (Del)",
        "btn_exit": "Exit App",
        "btn_update": "Check Update",
        "delete_original_label": "Delete Original (Dangerous)",
        "status_ready": "Ready - Please select a folder",
        "status_loading": "Loading file list...",
        "status_phase1": "Phase 1/2: Fast Scan (pHash)... {} items",
        "status_phase2": "Phase 2/2: AI Analysis... {} items",
        "status_scanning": "Scanning... Analyzed {}",
        "status_stopping": "Stopping... (Finishing batch)",
        "status_stopped": "Stopped manually",
        "status_done": "Done! Found {} duplicates, {} corrupted",
        "status_nodup": "Done! No duplicates or corrupted files found",
        "status_resuming": "Resuming scan from item {}...",
        "status_check_update": "Checking for updates...",
        "btn_load_more": "⬇ Load More",
        "msg_success": "Successfully deleted {} files",
        "msg_confirm": "Are you sure you want to delete {} files?",
        "msg_warn_title": "Dangerous Operation",
        "msg_warn_content": "WARNING: You have checked 'Delete Original'!\n\nThis will delete BOTH the Original and Duplicate files for {count} sets.\nThis cannot be undone within this app.\n\nAre you sure?",
        "msg_warn_corrupt": "Delete {count} corrupted files?",
        "msg_new_version": "New version V{} available!\n\nRelease Notes:\n{}\n\nOpen download page?",
        "tab_dup": "Duplicates ({})",
        "tab_bad": "Corrupted ({})",
        "col_corrupt": "❌ File Corrupted",
        "btn_del": "Del"
    }
}

# ==========================================
# UI 元件
# ==========================================
class LazyRow(ctk.CTkFrame):
    def __init__(self, master, result_item, index, click_callback, lang_code, **kwargs):
        super().__init__(master, **kwargs)
        self.result_item = result_item
        self.index = index
        self.click_callback = click_callback
        self.lang_code = lang_code
        
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(5, weight=1)

        val = "on" if self.result_item.checked else "off"
        self.checkbox_var = ctk.StringVar(value=val)
        
        del_text = TRANSLATIONS[self.lang_code].get("btn_del", "Del")
        self.checkbox = ctk.CTkCheckBox(self, text=del_text, variable=self.checkbox_var, 
                                        onvalue="on", offvalue="off", width=60,
                                        font=("Arial", 12, "bold"), fg_color="#E04F5F",
                                        command=self.on_checkbox_click)
        self.checkbox.grid(row=0, column=0, padx=10, pady=10)

        self.frame_org = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_org.grid(row=0, column=1, padx=5)
        self.thumb_org = ctk.CTkLabel(self.frame_org, text="...", width=90, height=90, fg_color="#333")
        self.thumb_org.pack(pady=(0,2))
        self._create_buttons(self.frame_org, self.result_item.org_path)

        self.lbl_org = ctk.CTkLabel(self, text=self.result_item.org_path, 
                                    text_color="#88AAFF", anchor="w",
                                    wraplength=350, justify="left")
        self.lbl_org.grid(row=0, column=2, padx=5, sticky="ew")

        color = "#00FF00" if result_item.similarity > 95 else "#FFFF00"
        self.lbl_sim = ctk.CTkLabel(self, text=f"◄ {result_item.similarity}% ►", 
                                    text_color=color, font=("Arial", 14, "bold"))
        self.lbl_sim.grid(row=0, column=3, padx=5)

        self.frame_dup = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_dup.grid(row=0, column=4, padx=5)
        self.thumb_dup = ctk.CTkLabel(self.frame_dup, text="...", width=90, height=90, fg_color="#333")
        self.thumb_dup.pack(pady=(0,2))
        self._create_buttons(self.frame_dup, self.result_item.dup_path)

        self.lbl_dup = ctk.CTkLabel(self, text=self.result_item.dup_path, 
                                    text_color="#FF8888", anchor="w",
                                    wraplength=350, justify="left")
        self.lbl_dup.grid(row=0, column=5, padx=5, sticky="ew")

        self._bind_click_events()
        self.load_images_async()

    def _create_buttons(self, parent, path):
        btn_box = ctk.CTkFrame(parent, fg_color="transparent")
        btn_box.pack()
        ctk.CTkButton(btn_box, text="🔍", width=40, height=20, fg_color="#444", 
                      command=lambda: self.open_file(path)).pack(side="left", padx=1)
        ctk.CTkButton(btn_box, text="📂", width=40, height=20, fg_color="#335588",
                      command=lambda: self.open_folder(path)).pack(side="left", padx=1)

    def _bind_click_events(self):
        def handler(event):
            self.click_callback(self.result_item, self.index, event)
        self.bind("<Button-1>", handler)
        self.frame_org.bind("<Button-1>", handler)
        self.frame_dup.bind("<Button-1>", handler)
        self.thumb_org.bind("<Button-1>", handler)
        self.thumb_dup.bind("<Button-1>", handler)
        self.lbl_org.bind("<Button-1>", handler)
        self.lbl_dup.bind("<Button-1>", handler)
        self.lbl_sim.bind("<Button-1>", handler)

    def on_checkbox_click(self):
        self.result_item.checked = (self.checkbox_var.get() == "on")
        self.click_callback(self.result_item, self.index, None)

    def refresh_state(self):
        new_val = "on" if self.result_item.checked else "off"
        if self.checkbox_var.get() != new_val:
            self.checkbox_var.set(new_val)
        bg = "#3A3A3A" if self.result_item.checked else "transparent"
        self.configure(fg_color=bg)

    def load_images_async(self):
        image_loader_pool.submit(self._load_img_task)

    def _load_img_task(self):
        img1 = self._read_and_resize(self.result_item.org_path)
        img2 = self._read_and_resize(self.result_item.dup_path)
        if self.winfo_exists():
            self.after(0, lambda: self._update_thumbs(img1, img2))

    def _read_and_resize(self, path):
        try:
            stream = np.fromfile(path, dtype=np.uint8)
            cv_img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
            if cv_img is None: return None
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w = cv_img.shape[:2]
            scale = 90 / max(h, w)
            cv_img = cv2.resize(cv_img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            pil_img = PILImage.fromarray(cv_img)
            return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
        except: return None

    def _update_thumbs(self, img1, img2):
        if not self.winfo_exists(): return
        self.thumb_org.configure(image=img1 if img1 else None, text="" if img1 else "Error")
        self.thumb_dup.configure(image=img2 if img2 else None, text="" if img2 else "Error")

    def open_file(self, path):
        try: os.startfile(path_fix(path))
        except: pass

    def open_folder(self, path):
        try: subprocess.Popen(f'explorer /select,"{path_fix(path)}"')
        except: pass

class CorruptRow(ctk.CTkFrame):
    def __init__(self, master, corrupt_item, index, click_callback, lang_code, **kwargs):
        super().__init__(master, **kwargs)
        self.corrupt_item = corrupt_item
        self.index = index
        self.click_callback = click_callback
        self.lang_code = lang_code
        
        self.grid_columnconfigure(1, weight=1)

        val = "on" if self.corrupt_item.checked else "off"
        self.checkbox_var = ctk.StringVar(value=val)
        
        del_text = TRANSLATIONS[self.lang_code].get("btn_del", "Del")
        self.checkbox = ctk.CTkCheckBox(self, text=del_text, variable=self.checkbox_var, 
                                        onvalue="on", offvalue="off", width=60,
                                        font=("Arial", 12, "bold"), fg_color="#E04F5F",
                                        command=self.on_checkbox_click)
        self.checkbox.grid(row=0, column=0, padx=10, pady=10)

        self.lbl_path = ctk.CTkLabel(self, text=self.corrupt_item.path, anchor="w", 
                                     text_color="#FF6666", wraplength=600, justify="left")
        self.lbl_path.grid(row=0, column=1, padx=10, sticky="ew")

        status_text = TRANSLATIONS[self.lang_code].get("col_corrupt", "Corrupted")
        self.lbl_status = ctk.CTkLabel(self, text=status_text, text_color="gray")
        self.lbl_status.grid(row=0, column=2, padx=10)

        ctk.CTkButton(self, text="📂", width=40, height=20, fg_color="#335588",
                      command=lambda: self.open_folder(self.corrupt_item.path)).grid(row=0, column=3, padx=5)
        
        self._bind_click_events()

    def _bind_click_events(self):
        def handler(event):
            self.click_callback(self.corrupt_item, self.index, event)
        self.bind("<Button-1>", handler)
        self.lbl_path.bind("<Button-1>", handler)
        self.lbl_status.bind("<Button-1>", handler)

    def on_checkbox_click(self):
        self.corrupt_item.checked = (self.checkbox_var.get() == "on")
        self.click_callback(self.corrupt_item, self.index, None)

    def refresh_state(self):
        new_val = "on" if self.corrupt_item.checked else "off"
        if self.checkbox_var.get() != new_val:
            self.checkbox_var.set(new_val)
        bg = "#3A3A3A" if self.corrupt_item.checked else "transparent"
        self.configure(fg_color=bg)

    def open_folder(self, path):
        try: subprocess.Popen(f'explorer /select,"{path_fix(path)}"')
        except: pass

# ==========================================
# 主程式
# ==========================================
class AsyncCleanerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.current_lang = "zh_TW"
        
        self.bind("<Delete>", lambda event: self.delete_selected())

        self.geometry("1400x850")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.protocol("WM_DELETE_WINDOW", self.safe_exit)

        # 狀態
        self.is_scanning = False
        self.folder_path = ""
        self.last_scan_folder = ""
        
        self.all_results = []
        self.corrupt_results = []
        self.processed_files = set()
        self.found_pairs = set() 
        
        self.last_clicked_index_dup = None
        self.last_clicked_index_bad = None
        
        self.rendered_count_dup = 0
        self.rendered_count_bad = 0
        self.batch_size = 30
        
        self.widget_list_dup = []
        self.widget_list_bad = []

        self.KEY_TAB_DUP = "dup_tab"
        self.KEY_TAB_BAD = "bad_tab"

        self.setup_sidebar()
        self.setup_main_area()
        self.update_ui_text()
        
        # 啟動時自動檢查一次更新
        self.trigger_update_check(silent=True)

    def tr(self, key): 
        return TRANSLATIONS[self.current_lang].get(key, key)

    def setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.lbl_title = ctk.CTkLabel(self.sidebar, text="CLEANER AI", font=ctk.CTkFont(size=24, weight="bold"))
        self.lbl_title.grid(row=0, column=0, padx=20, pady=(30, 10))

        self.lbl_lang = ctk.CTkLabel(self.sidebar, text="Language")
        self.lbl_lang.grid(row=1, column=0, padx=20, pady=(10, 0))
        
        self.lang_var = ctk.StringVar(value="繁體中文")
        self.lang_menu = ctk.CTkOptionMenu(self.sidebar, variable=self.lang_var, 
                                           values=["繁體中文", "English"],
                                           command=self.change_language)
        self.lang_menu.grid(row=2, column=0, padx=20, pady=5)

        self.lbl_mode = ctk.CTkLabel(self.sidebar, text="Mode") 
        self.lbl_mode.grid(row=3, column=0, padx=20, pady=(10, 0))

        self.mode_var = ctk.StringVar(value="Smart")
        self.mode_menu = ctk.CTkOptionMenu(self.sidebar, variable=self.mode_var, fg_color="#2B719E")
        self.mode_menu.grid(row=4, column=0, padx=20, pady=5)

        self.btn_select = ctk.CTkButton(self.sidebar, text="Select Folder", command=self.select_folder)
        self.btn_select.grid(row=5, column=0, padx=20, pady=10)

        self.slider_lbl = ctk.CTkLabel(self.sidebar, text="")
        self.slider_lbl.grid(row=6, column=0)
        self.slider = ctk.CTkSlider(self.sidebar, from_=80, to=100, number_of_steps=40, command=self.update_slider_text)
        self.slider.set(95)
        self.slider.grid(row=7, column=0, padx=20, pady=5)
        
        self.btn_scan = ctk.CTkButton(self.sidebar, text="Scan", fg_color="#E04F5F", command=self.start_scan)
        self.btn_scan.grid(row=8, column=0, padx=20, pady=10)
        self.btn_stop = ctk.CTkButton(self.sidebar, text="Stop", state="disabled", command=self.stop_scan)
        self.btn_stop.grid(row=9, column=0, padx=20, pady=5)
        self.btn_clear = ctk.CTkButton(self.sidebar, text="Clear", command=self.clear_all)
        self.btn_clear.grid(row=10, column=0, padx=20, pady=5)

        self.lbl_tools = ctk.CTkLabel(self.sidebar, text="Tools", font=("Arial", 12, "bold"))
        self.lbl_tools.grid(row=11, column=0, pady=(20,5))
        self.btn_smart = ctk.CTkButton(self.sidebar, text="Select All", fg_color="#2B719E", command=self.smart_select)
        self.btn_smart.grid(row=12, column=0, padx=20, pady=5)
        self.btn_deselect = ctk.CTkButton(self.sidebar, text="Deselect", fg_color="#555", command=self.deselect_all)
        self.btn_deselect.grid(row=13, column=0, padx=20, pady=5)

        self.del_org_var = ctk.BooleanVar(value=False)
        self.chk_del_org = ctk.CTkCheckBox(self.sidebar, text="Delete Original", variable=self.del_org_var, fg_color="#E04F5F")
        self.chk_del_org.grid(row=14, column=0, padx=20, pady=10)

        self.btn_delete = ctk.CTkButton(self.sidebar, text="Delete Selected", state="disabled", fg_color="gray", command=self.delete_selected)
        self.btn_delete.grid(row=15, column=0, padx=20, pady=10)
        
        self.lbl_status = ctk.CTkLabel(self.sidebar, text="Ready", text_color="silver", wraplength=220)
        self.lbl_status.grid(row=16, column=0, padx=20, pady=10, sticky="s")

        # 更新按鈕放在底部
        self.btn_update = ctk.CTkButton(self.sidebar, text="Check Update", fg_color="#333", command=lambda: self.trigger_update_check(False))
        self.btn_update.grid(row=17, column=0, padx=20, pady=(20, 5), sticky="s")

        self.btn_exit = ctk.CTkButton(self.sidebar, text="Exit App", fg_color="#990000", command=self.safe_exit)
        self.btn_exit.grid(row=18, column=0, padx=20, pady=10, sticky="s")

    def setup_main_area(self):
        self.tab_view = ctk.CTkTabview(self, command=self.on_tab_change)
        self.tab_view.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.tab_dup = self.tab_view.add(self.KEY_TAB_DUP)
        self.tab_bad = self.tab_view.add(self.KEY_TAB_BAD)

        self.scroll_dup = ctk.CTkScrollableFrame(self.tab_dup, label_text="Duplicates")
        self.scroll_dup.pack(fill="both", expand=True)
        self.btn_load_more_dup = ctk.CTkButton(self.scroll_dup, text="", command=lambda: self.render_next_batch("dup"))
        
        self.scroll_bad = ctk.CTkScrollableFrame(self.tab_bad, label_text="Corrupted Files")
        self.scroll_bad.pack(fill="both", expand=True)
        self.btn_load_more_bad = ctk.CTkButton(self.scroll_bad, text="", command=lambda: self.render_next_batch("bad"))

    # ==========================
    # 更新邏輯
    # ==========================
    def trigger_update_check(self, silent=False):
        if not HAS_NETWORK:
            if not silent: messagebox.showerror("Error", "Need 'requests' library")
            return
        
        if not silent:
            self.lbl_status.configure(text=self.tr("status_check_update"))
            
        def on_found(tag, notes):
            self.after(0, lambda: self._show_update_dialog(tag, notes))
            
        def on_error(msg):
            if not silent: print(msg)

        threading.Thread(target=check_for_updates_thread, args=(on_found, on_error), daemon=True).start()

    def _show_update_dialog(self, tag, notes):
        msg = self.tr("msg_new_version").format(tag, notes)
        if messagebox.askyesno(f"New Version {tag}", msg):
            webbrowser.open(DOWNLOAD_PAGE_URL)

    # ==========================
    # 介面變更
    # ==========================
    def change_language(self, choice):
        if choice == "繁體中文":
            self.current_lang = "zh_TW"
        else:
            self.current_lang = "en_US"
        
        self.update_ui_text()
        
        for w in self.widget_list_dup:
            txt = TRANSLATIONS[self.current_lang].get("btn_del", "Del")
            w.checkbox.configure(text=txt)
        
        for w in self.widget_list_bad:
            txt = TRANSLATIONS[self.current_lang].get("btn_del", "Del")
            status_txt = TRANSLATIONS[self.current_lang].get("col_corrupt", "Corrupted")
            w.checkbox.configure(text=txt)
            w.lbl_status.configure(text=status_txt)

    def update_ui_text(self):
        self.title(self.tr("app_title"))
        self.lbl_title.configure(text=self.tr("sidebar_title"))
        self.lbl_lang.configure(text=self.tr("lbl_lang"))
        self.btn_select.configure(text=self.tr("btn_folder"))
        self.lbl_mode.configure(text=self.tr("lbl_mode"))
        self.btn_scan.configure(text=self.tr("btn_scan"))
        
        current_mode = self.mode_var.get()
        modes = [self.tr("mode_smart"), self.tr("mode_ai"), self.tr("mode_phash")]
        self.mode_menu.configure(values=modes)
        
        if "Smart" in current_mode or "智慧" in current_mode:
            self.mode_menu.set(self.tr("mode_smart"))
        elif "AI" in current_mode or "Neural" in current_mode:
            self.mode_menu.set(self.tr("mode_ai"))
        else:
            self.mode_menu.set(self.tr("mode_phash"))

        self.update_slider_text(self.slider.get())
        self.btn_stop.configure(text=self.tr("btn_stop"))
        self.btn_clear.configure(text=self.tr("btn_clear"))
        self.lbl_tools.configure(text=self.tr("lbl_tools"))
        self.btn_smart.configure(text=self.tr("btn_smart"))
        self.btn_deselect.configure(text=self.tr("btn_deselect"))
        self.chk_del_org.configure(text=self.tr("delete_original_label"))
        self.btn_delete.configure(text=self.tr("btn_delete"))
        self.btn_exit.configure(text=self.tr("btn_exit"))
        self.btn_update.configure(text=self.tr("btn_update"))
        
        if not self.is_scanning:
            if not self.all_results and not self.corrupt_results:
                self.lbl_status.configure(text=self.tr("status_ready"))
            else:
                msg = self.tr("status_done").format(len(self.all_results), len(self.corrupt_results))
                self.lbl_status.configure(text=msg)
        
        try:
            title_dup = self.tr("tab_dup").format(len(self.all_results))
            title_bad = self.tr("tab_bad").format(len(self.corrupt_results))
            self.tab_view._segmented_button._buttons_dict[self.KEY_TAB_DUP].configure(text=title_dup)
            self.tab_view._segmented_button._buttons_dict[self.KEY_TAB_BAD].configure(text=title_bad)
            
            if self.btn_load_more_dup.winfo_ismapped():
                self.update_load_more_btn("dup")
            if self.btn_load_more_bad.winfo_ismapped():
                self.update_load_more_btn("bad")
        except Exception: 
            pass

    def update_slider_text(self, val):
        self.slider_lbl.configure(text=self.tr("lbl_threshold").format(round(val, 1)))

    def select_folder(self):
        f = filedialog.askdirectory()
        if f: 
            self.folder_path = path_fix(f)
            self.lbl_status.configure(text=f"{os.path.basename(self.folder_path)}")

    def on_tab_change(self):
        self._check_delete_btn_state()

    def handle_row_click(self, item, index, event):
        current_tab = self.tab_view.get()
        target_list = self.all_results if current_tab == self.KEY_TAB_DUP else self.corrupt_results
        last_clicked = self.last_clicked_index_dup if current_tab == self.KEY_TAB_DUP else self.last_clicked_index_bad
        
        if event is None:
            if current_tab == self.KEY_TAB_DUP: self.last_clicked_index_dup = index
            else: self.last_clicked_index_bad = index
            self._check_delete_btn_state()
            self._refresh_visible_rows(current_tab)
            return

        is_shift = (event.state & 1) != 0
        
        if is_shift and last_clicked is not None:
            start = min(last_clicked, index)
            end = max(last_clicked, index)
            target_state = True 
            for i in range(start, end + 1):
                if i < len(target_list):
                    target_list[i].checked = target_state
        else:
            item.checked = not item.checked

        if current_tab == self.KEY_TAB_DUP: self.last_clicked_index_dup = index
        else: self.last_clicked_index_bad = index

        self._check_delete_btn_state()
        self._refresh_visible_rows(current_tab)

    def _refresh_visible_rows(self, tab_key):
        if tab_key == self.KEY_TAB_DUP:
            widgets = self.widget_list_dup
        else:
            widgets = self.widget_list_bad
        for w in widgets:
            w.refresh_state()

    def _check_delete_btn_state(self):
        current_key = self.tab_view.get()
        has_items = False
        if current_key == self.KEY_TAB_DUP and self.all_results: has_items = True
        if current_key == self.KEY_TAB_BAD and self.corrupt_results: has_items = True
        
        if has_items:
            self.btn_delete.configure(state="normal", fg_color="#E04F5F")
        else:
            self.btn_delete.configure(state="disabled", fg_color="gray")

    def start_scan(self):
        if not self.folder_path: 
            messagebox.showwarning("Info", self.tr("status_ready"))
            return
        
        self.lbl_status.configure(text=self.tr("status_loading"))
        
        if self.folder_path != self.last_scan_folder:
            self.clear_all(auto_reset=True)
            self.last_scan_folder = self.folder_path
        else:
            if len(self.processed_files) > 0:
                self.lbl_status.configure(text=self.tr("status_resuming").format(len(self.processed_files)))
        
        self.is_scanning = True
        self.btn_scan.configure(state="disabled")
        self.btn_stop.configure(state="normal", fg_color="#E04F5F")
        self.btn_delete.configure(state="disabled", fg_color="gray")

        mode = self.mode_var.get()
        
        if "Smart" in mode or "智慧" in mode:
            target_func = self.run_smart_scan_thread
        elif "AI" in mode or "Neural" in mode:
            target_func = self.run_ai_process_thread
        else:
            target_func = self.run_phash_process_thread
            
        threading.Thread(target=target_func, daemon=True).start()

    def stop_scan(self):
        self.is_scanning = False
        self.lbl_status.configure(text=self.tr("status_stopping"))

    def get_all_images(self):
        files = []
        exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        for r, _, fs in os.walk(self.folder_path):
            if not self.is_scanning: break
            for f in fs:
                if os.path.splitext(f)[1].lower() in exts:
                    files.append(os.path.join(r, f))
        return files

    # ==========================
    # 智慧掃描核心
    # ==========================
    def run_smart_scan_thread(self):
        # Phase 1: pHash (99% match)
        self.after(0, lambda: self.lbl_status.configure(text=self.tr("status_phase1").format(len(self.processed_files))))
        self._run_core_logic(mode="pHash", threshold_override=99) 
        
        if not self.is_scanning:
            self._on_stop_or_finish()
            return

        # Phase 2: AI (User setting)
        load_ai_model()
        if not ai_model:
            print("AI Load Failed")
            self._on_stop_or_finish()
            return

        self.after(0, lambda: self.lbl_status.configure(text=self.tr("status_phase2").format(len(self.processed_files))))
        self._run_core_logic(mode="AI")
        
        self._on_stop_or_finish()

    def run_phash_process_thread(self):
        self._run_core_logic(mode="pHash")
        self._on_stop_or_finish()

    def run_ai_process_thread(self):
        load_ai_model()
        if not ai_model:
            self.after(0, self.on_scan_finished)
            return
        self._run_core_logic(mode="AI")
        self._on_stop_or_finish()

    def _run_core_logic(self, mode, threshold_override=None):
        all_files = self.get_all_images()
        # 關鍵：只處理「未完成」的檔案
        files_to_process = [f for f in all_files if f not in self.processed_files]
        
        if not files_to_process: return

        stored_data = [] 
        stored_paths = []
        count = len(self.processed_files)
        
        if threshold_override:
            threshold_pct = threshold_override
        else:
            threshold_pct = self.slider.get()

        chunk_size = 50 
        max_cpu = max(1, (os.cpu_count() or 4) - 1)
        
        if mode == "pHash":
            Executor = concurrent.futures.ProcessPoolExecutor
            worker_func = calculate_phash_task
            max_workers = max_cpu
            match_limit = int((100 - threshold_pct) / 100 * 64)
            match_limit = max(0, min(match_limit, 10))
        else:
            Executor = concurrent.futures.ThreadPoolExecutor
            worker_func = calculate_ai_vector_single
            max_workers = 4
            match_limit = threshold_pct / 100.0

        with Executor(max_workers=max_workers) as executor:
            for i in range(0, len(files_to_process), chunk_size):
                if not self.is_scanning: break
                
                chunk = files_to_process[i : i + chunk_size]
                futures = {executor.submit(worker_func, f): f for f in chunk}
                
                for fut in concurrent.futures.as_completed(futures):
                    data, path, status = fut.result()
                    
                    if status == "CORRUPT":
                        self.processed_files.add(path)
                        self.corrupt_results.append(CorruptItem(path))
                        self.after(0, lambda: self.live_update_ui("bad"))
                        count += 1
                        continue
                    
                    if data is None: 
                        self.processed_files.add(path)
                        count += 1
                        continue

                    is_dup = False
                    if stored_data:
                        if mode == "pHash":
                            h_arr = data.hash.flatten()
                            for idx, sh_arr in enumerate(stored_data):
                                diff = np.count_nonzero(h_arr != sh_arr)
                                if diff <= match_limit:
                                    sim = int((1 - diff/64) * 100)
                                    self._add_dup_result(path, stored_paths[idx], sim)
                                    self.processed_files.add(path)
                                    is_dup = True
                                    break
                        else: # AI
                            matrix = np.array(stored_data)
                            sims = np.dot(matrix, data)
                            max_idx = np.argmax(sims)
                            max_sim = sims[max_idx]
                            
                            if max_sim >= match_limit:
                                self._add_dup_result(path, stored_paths[max_idx], int(max_sim * 100))
                                self.processed_files.add(path)
                                is_dup = True

                    if not is_dup:
                        if mode == "pHash":
                            stored_data.append(data.hash.flatten())
                        else:
                            stored_data.append(data)
                        stored_paths.append(path)
                        
                        # 智慧模式判斷：Phase 1 沒找到重複不等於結束
                        current_mode_ui = self.mode_var.get()
                        is_smart_phase_1 = ("Smart" in current_mode_ui or "智慧" in current_mode_ui) and mode == "pHash"
                        
                        if not is_smart_phase_1:
                            self.processed_files.add(path)

                    count += 1
                    if count % 10 == 0:
                        if "Smart" in self.mode_var.get() or "智慧" in self.mode_var.get():
                            phase_txt = "status_phase1" if mode == "pHash" else "status_phase2"
                            txt = self.tr(phase_txt).format(count)
                            self.after(0, lambda t=txt: self.lbl_status.configure(text=t))
                        else:
                            self.after(0, self.update_status_progress, count)

    def _add_dup_result(self, dup, org, sim):
        pair_key = tuple(sorted((dup, org)))
        if pair_key not in self.found_pairs:
            self.found_pairs.add(pair_key)
            item = ResultItem(dup, org, sim)
            self.all_results.append(item)
            self.after(0, lambda: self.live_update_ui("dup"))

    def update_status_progress(self, count):
        self.lbl_status.configure(text=self.tr("status_scanning").format(count))

    def _on_stop_or_finish(self):
        if not self.is_scanning:
            self.after(0, lambda: self.lbl_status.configure(text=self.tr("status_stopped")))
            self.after(0, lambda: self.btn_scan.configure(state="normal"))
            self.after(0, lambda: self.btn_stop.configure(state="disabled", fg_color="gray"))
        else:
            self.after(0, self.on_scan_finished)

    def on_scan_finished(self):
        self.btn_scan.configure(state="normal")
        self.btn_stop.configure(state="disabled", fg_color="gray")
        msg = self.tr("status_done").format(len(self.all_results), len(self.corrupt_results))
        if not self.all_results and not self.corrupt_results:
            msg = self.tr("status_nodup")
        self.lbl_status.configure(text=msg)
        self.update_ui_text() 
        self._check_delete_btn_state()

    def clear_all(self, auto_reset=False):
        self.all_results.clear()
        self.corrupt_results.clear()
        self.found_pairs.clear()
        if hasattr(self, 'smart_ai_paths'): self.smart_ai_paths.clear()
        
        if not auto_reset:
            self.processed_files.clear()
        if auto_reset:
            self.processed_files.clear()

        for w in self.widget_list_dup: w.destroy()
        for w in self.widget_list_bad: w.destroy()
        self.widget_list_dup.clear()
        self.widget_list_bad.clear()
        self.rendered_count_dup = 0
        self.rendered_count_bad = 0
        self.last_clicked_index_dup = None
        self.last_clicked_index_bad = None
        self.btn_load_more_dup.pack_forget()
        self.btn_load_more_bad.pack_forget()
        self.btn_delete.configure(state="disabled", fg_color="gray")
        self.update_ui_text()

    # ... (其餘 UI 邏輯保持不變) ...
    def live_update_ui(self, target_type):
        if target_type == "dup":
            if self.rendered_count_dup < len(self.all_results):
                self.render_next_batch("dup")
        elif target_type == "bad":
            if self.rendered_count_bad < len(self.corrupt_results):
                self.render_next_batch("bad")

    def render_next_batch(self, target_type):
        if target_type == "dup":
            self.btn_load_more_dup.pack_forget()
            start = self.rendered_count_dup
            end = min(start + self.batch_size, len(self.all_results))
            if start >= end: return
            for i in range(start, end):
                item = self.all_results[i]
                row = LazyRow(self.scroll_dup, item, i, self.handle_row_click, self.current_lang)
                row.pack(fill="x", padx=5, pady=5)
                self.widget_list_dup.append(row)
            self.rendered_count_dup = end
            self.update_load_more_btn("dup")
        elif target_type == "bad":
            self.btn_load_more_bad.pack_forget()
            start = self.rendered_count_bad
            end = min(start + self.batch_size, len(self.corrupt_results))
            if start >= end: return
            for i in range(start, end):
                item = self.corrupt_results[i]
                row = CorruptRow(self.scroll_bad, item, i, self.handle_row_click, self.current_lang)
                row.pack(fill="x", padx=5, pady=5)
                self.widget_list_bad.append(row)
            self.rendered_count_bad = end
            self.update_load_more_btn("bad")

    def update_load_more_btn(self, target_type):
        load_txt = self.tr("btn_load_more")
        if target_type == "dup":
            if self.rendered_count_dup < len(self.all_results):
                self.btn_load_more_dup.configure(text=f"{load_txt} ({self.rendered_count_dup}/{len(self.all_results)})")
                self.btn_load_more_dup.pack(pady=10, fill="x")
            else: self.btn_load_more_dup.pack_forget()
        else:
            if self.rendered_count_bad < len(self.corrupt_results):
                self.btn_load_more_bad.configure(text=f"{load_txt} ({self.rendered_count_bad}/{len(self.corrupt_results)})")
                self.btn_load_more_bad.pack(pady=10, fill="x")
            else: self.btn_load_more_bad.pack_forget()

    def smart_select(self):
        current_key = self.tab_view.get()
        if current_key == self.KEY_TAB_DUP:
            for item in self.all_results: item.checked = True
            self._refresh_visible_rows(self.KEY_TAB_DUP)
        elif current_key == self.KEY_TAB_BAD:
            for item in self.corrupt_results: item.checked = True
            self._refresh_visible_rows(self.KEY_TAB_BAD)

    def deselect_all(self):
        current_key = self.tab_view.get()
        if current_key == self.KEY_TAB_DUP:
            for item in self.all_results: item.checked = False
            self._refresh_visible_rows(self.KEY_TAB_DUP)
        elif current_key == self.KEY_TAB_BAD:
            for item in self.corrupt_results: item.checked = False
            self._refresh_visible_rows(self.KEY_TAB_BAD)

    def delete_selected(self):
        if self.btn_delete.cget("state") == "disabled": return
        current_key = self.tab_view.get()
        if current_key == self.KEY_TAB_DUP:
            self.delete_duplicates()
        elif current_key == self.KEY_TAB_BAD:
            self.delete_corrupted()

    def delete_duplicates(self):
        to_del = [x for x in self.all_results if x.checked]
        if not to_del: return
        if self.del_org_var.get():
            title = self.tr("msg_warn_title")
            content = self.tr("msg_warn_content").format(count=len(to_del))
            if not messagebox.askyesno(title, content, icon='warning'): return
        else:
            if not messagebox.askyesno("Confirm", self.tr("msg_confirm").format(len(to_del))): return
        cnt = 0
        deleted_cache = set()
        for item in to_del:
            p_dup = path_fix(item.dup_path)
            if p_dup not in deleted_cache and os.path.exists(p_dup):
                try: 
                    send2trash(p_dup)
                    deleted_cache.add(p_dup)
                    cnt += 1
                except: pass
            if self.del_org_var.get():
                p_org = path_fix(item.org_path)
                if p_org not in deleted_cache and os.path.exists(p_org):
                    try:
                        send2trash(p_org)
                        deleted_cache.add(p_org)
                        cnt += 1
                    except: pass
        self.all_results = [x for x in self.all_results if not x.checked]
        self.last_clicked_index_dup = None 
        for w in self.widget_list_dup: w.destroy()
        self.widget_list_dup.clear()
        self.rendered_count_dup = 0
        self.live_update_ui("dup")
        self.update_ui_text()
        messagebox.showinfo("Done", self.tr("msg_success").format(cnt))

    def delete_corrupted(self):
        to_del = [x for x in self.corrupt_results if x.checked]
        if not to_del: return
        msg = self.tr("msg_warn_corrupt").format(count=len(to_del))
        if not messagebox.askyesno("Confirm", msg): return
        cnt = 0
        for item in to_del:
            p = path_fix(item.path)
            if os.path.exists(p):
                try:
                    send2trash(p)
                    cnt += 1
                except: pass
        self.corrupt_results = [x for x in self.corrupt_results if not x.checked]
        self.last_clicked_index_bad = None
        for w in self.widget_list_bad: w.destroy()
        self.widget_list_bad.clear()
        self.rendered_count_bad = 0
        self.live_update_ui("bad")
        self.update_ui_text()
        messagebox.showinfo("Done", self.tr("msg_success").format(cnt))

    def safe_exit(self):
        self.is_scanning = False
        os._exit(0)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = AsyncCleanerApp()
    app.mainloop()