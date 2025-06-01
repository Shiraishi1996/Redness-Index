# streamlit_app.py
"""
クラスタリングアプリ（画像・動画・PCカメラ・スマホカメラ対応 + GPSマップ）

✅ アップロード処理（画像/動画, 4分割, GPS, AVI出力, CSV）
✅ PCカメラリアルタイム解析（OpenCV, AVI出力, CSV, IP-GPSマップ）
✅ スマホカメラ（WebRTC + JSでカメラ+GPS）

依存:
streamlit, opencv-python, streamlit-webrtc, streamlit-js-eval, streamlit-folium, scikit-learn, Pillow, pandas, matplotlib, folium, geocoder, av
"""

import streamlit as st
import numpy as np
import cv2
import av
import pandas as pd
import geocoder
from matplotlib import cm
from tempfile import TemporaryDirectory
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from streamlit_js_eval import streamlit_js_eval

# UI 選択
st.set_page_config(layout="wide")
st.title("🌱 クラスタリングアプリ - アップロード / PCカメラ / スマホカメラ")
mode = st.radio("モード選択", ["アップロード処理", "PCカメラ", "スマホカメラ"], horizontal=True)

# 共通パラメータ
k = st.slider("クラスタ数", 2, 10, 6)
resize_ratio = st.slider("リサイズ倍率（画像用）", 0.1, 1.0, 1.0)
index_type = st.selectbox("インデックス", ["RI", "VARI", "VWRI"])
cluster_id = st.number_input("対象クラスタ番号", 0, k-1, 0)

# --- ユーティリティ ---
def compute_index(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = [rgb[:, :, i].astype(np.float32) for i in range(3)]
    safe = lambda n, d: n / np.where(d == 0, 1e-5, d)
    if index_type == "RI": return safe(g - r, g + r)
    if index_type == "VARI": return safe(g - r, g + r - b)
    return safe(g - r - b, g + r + b)

def cluster_index(index_map):
    flat = index_map.flatten().reshape(-1, 1)
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(flat).reshape(index_map.shape)
    return labels

def visualize_cluster(index_map, labels):
    mean_vals = sorted([(i, index_map[labels==i].mean()) for i in range(k)], key=lambda x:x[1], reverse=True)
    cmap = cm.get_cmap('RdYlGn', k)
    id2col = {cid: (np.array(cmap(rank/(k-1))[:3]) * 255).astype(np.uint8)
              for rank, (cid, _) in enumerate(mean_vals)}
    vis = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for cid in range(k): vis[labels == cid] = id2col[cid]
    mask = (labels == cluster_id).astype(np.uint8) * 255
    pct = 100 * np.mean(labels == cluster_id)
    return vis, mask, pct

def analyze(img):
    idx = compute_index(img)
    lbl = cluster_index(idx)
    return visualize_cluster(idx, lbl)

def quadrants(img):
    h, w = img.shape[:2]
    return {
        "全体": img,
        "左上": img[:h//2, :w//2],
        "右上": img[:h//2, w//2:],
        "左下": img[h//2:, :w//2],
        "右下": img[h//2:, w//2:]
    }

def exif_gps(path):
    try:
        img = Image.open(path)
        exif = img._getexif() or {}
        gps_info = {GPSTAGS.get(t): v for t, v in exif.get(34853, {}).items()}
        if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
            def dms_to_deg(dms):
                d, m, s = [v[0]/v[1] for v in dms]
                return d + m/60 + s/3600
            lat = dms_to_deg(gps_info['GPSLatitude'])
            if gps_info.get('GPSLatitudeRef') == 'S': lat = -lat
            lon = dms_to_deg(gps_info['GPSLongitude'])
            if gps_info.get('GPSLongitudeRef') == 'W': lon = -lon
            return lat, lon
    except: pass
    return None

def ip_gps():
    g = geocoder.ip('me')
    return g.latlng if g and g.ok else None

# --- スマホカメラモード ---
if mode == "スマホカメラ":
    st.header("📱 スマホカメラ + GPS")
    gps = streamlit_js_eval(js_expressions="navigator.geolocation.getCurrentPosition((p)=>{return {lat:p.coords.latitude, lon:p.coords.longitude};})", key="gps", ttl=1000)
    if isinstance(gps, dict) and 'lat' in gps:
        gps_coord = (gps['lat'], gps['lon'])
        st.success(f"📍 GPS: {gps_coord}")
    else:
        gps_coord = None
        st.warning("📡 位置情報を取得できません（ブラウザの位置許可を確認）")

    class CamProcessor(VideoProcessorBase):
        def __init__(self): self.pct = 0; self.mask = None
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            vis, mask, pct = analyze(img)
            self.pct, self.mask = pct, mask
            return av.VideoFrame.from_ndarray(vis, format="bgr24")

    ctx = webrtc_streamer(
        key="mobile",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=CamProcessor,
        media_stream_constraints={"video": {"facingMode": {"exact": "environment"}}, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        st.metric(f"クラスタ{cluster_id} %", f"{ctx.video_processor.pct:.2f}")
        st.image(ctx.video_processor.mask, caption="Mask", use_column_width=True)
        if gps_coord:
            m = folium.Map(location=gps_coord, zoom_start=15)
            folium.Marker(gps_coord, tooltip="現在地").add_to(m)
            st_folium(m, width=700, height=500)

# --- PCカメラモード ---
elif mode == "PCカメラ":
    st.header("💻 PCカメラ 処理")
    cams = [i for i in range(5) if cv2.VideoCapture(i).read()[0]]
    cam_id = st.selectbox("カメラ選択", cams)
    run = st.checkbox("開始")
    if run:
        cap = cv2.VideoCapture(cam_id)
        out = cv2.VideoWriter("pc_cam.avi", cv2.VideoWriter_fourcc(*'XVID'), 20, (640, 480))
        img_area = st.empty()
        recs = []
        gps = ip_gps()
        while cap.isOpened():
            ok, frm = cap.read()
            if not ok: break
            frm = cv2.resize(frm, (640, 480))
            vis, mask, pct = analyze(frm)
            out.write(vis)
            img_area.image(vis, caption=f"クラスタ{cluster_id}: {pct:.1f}%", use_column_width=True)
            recs.append({"src": "pc_frame", "pct": pct, "gps": gps})
            if st.button("停止"): break
        cap.release(); out.release()
        st.video("pc_cam.avi")
        df = pd.DataFrame(recs); st.dataframe(df)

# --- アップロード処理 ---
else:
    st.header("📁 アップロード処理")
    files = st.file_uploader("画像または動画をアップロード", type=["jpg", "jpeg", "png", "mp4", "avi"], accept_multiple_files=True)
    if files:
        stats, gps_pts = [], []
        with TemporaryDirectory() as tmp:
            for up in files:
                path = os.path.join(tmp, up.name)
                Path(path).write_bytes(up.read())
                ext = Path(path).suffix.lower()
                if ext in [".jpg", ".jpeg", ".png"]:
                    img = cv2.imread(path)
                    if resize_ratio < 1.0:
                        img = cv2.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio)
                    gps = exif_gps(path)
                    for name, region in quadrants(img).items():
                        vis, mask, pct = analyze(region)
                        st.image(vis, caption=f"{up.name} - {name}")
                        stats.append({"src": f"{up.name}_{name}", "pct": pct, "gps": gps})
                        if gps: gps_pts.append((gps, pct))
                elif ext in [".mp4", ".avi"]:
                    cap = cv2.VideoCapture(path)
                    w, h = int(cap.get(3)), int(cap.get(4)); fps = cap.get(5)
                    out_path = os.path.join(tmp, Path(up.name).stem + "_proc.avi")
                    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
                    pbar = st.progress(0); total = int(cap.get(7))
                    for i in range(total):
                        ret, frame = cap.read()
                        if not ret: break
                        vis, mask, pct = analyze(frame)
                        out.write(vis)
                        stats.append({"src": f"{up.name}_fr{i}", "pct": pct, "gps": None})
                        pbar.progress((i+1)/total)
                    cap.release(); out.release()
                    st.video(out_path)
                    with open(out_path, 'rb') as v: st.download_button("動画DL", v.read(), file_name=Path(out_path).name)
        if gps_pts:
            m = folium.Map(location=gps_pts[0][0], zoom_start=6)
            for (latlon, p) in gps_pts:
                folium.Circle(latlon, radius=p*10, color='green', fill=True).add_to(m)
            st_folium(m, width=700, height=500)
        df = pd.DataFrame(stats)
        st.dataframe(df)
        st.download_button("📥 CSVダウンロード", df.to_csv(index=False), "cluster_stats.csv")
