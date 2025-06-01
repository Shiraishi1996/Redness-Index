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

# GPS取得関数（Exif）
def exif_gps(image_path):
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)
        gps_latitude = tags.get('GPS GPSLatitude')
        gps_latitude_ref = tags.get('GPS GPSLatitudeRef')
        gps_longitude = tags.get('GPS GPSLongitude')
        gps_longitude_ref = tags.get('GPS GPSLongitudeRef')
        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            lat_values = [float(x.num) / float(x.den) for x in gps_latitude.values]
            lon_values = [float(x.num) / float(x.den) for x in gps_longitude.values]
            lat = lat_values[0] + lat_values[1]/60 + lat_values[2]/3600
            lon = lon_values[0] + lon_values[1]/60 + lon_values[2]/3600
            if gps_latitude_ref.values[0] != 'N':
                lat = -lat
            if gps_longitude_ref.values[0] != 'E':
                lon = -lon
            return (lat, lon)
    except:
        pass
    return None

# IPからのGPS取得
def ip_gps():
    try:
        g = geocoder.ip('me')
        if g.ok:
            return (g.latlng[0], g.latlng[1])
    except:
        pass
    return None

# ベストなGPS取得
def get_best_gps(exif_path=None):
    if exif_path:
        gps = exif_gps(exif_path)
        if gps: return gps, "EXIF"
    gps = ip_gps()
    if gps: return gps, "IP"
    return None, "Failed"

st.title("📷 画像/動画アップロード + GPS検出")

mode = st.radio("モード選択", ["アップロード", "カメラ"])

gps_coord = None
source = None

if mode == "カメラ":
    pic = st.camera_input("写真を撮ってください")
    if pic:
        with open("camera.jpg", "wb") as f:
            f.write(pic.read())
        gps = streamlit_js_eval(js_expressions="navigator.geolocation.getCurrentPosition", key="loc")
        if isinstance(gps, dict) and 'lat' in gps:
            gps_coord = (gps['lat'], gps['lon'])
            st.success(f"📍 GPS: {gps_coord}（スマホ）")
        else:
            gps_coord, source = get_best_gps("camera.jpg")
            if gps_coord:
                st.warning(f"📍 スマホJS失敗 → 代替位置（{source}）: {gps_coord}")
            else:
                st.error("📡 GPSがどの手段でも取得できませんでした")

else:
    files = st.file_uploader("画像や動画を選んでください", type=["jpg", "jpeg", "png", "mp4", "avi"], accept_multiple_files=True)
    if files:
        tmp = "uploads"
        os.makedirs(tmp, exist_ok=True)
        for up in files:
            path = os.path.join(tmp, up.name)
            Path(path).write_bytes(up.read())
            ext = Path(path).suffix.lower()
            gps, gps_source = get_best_gps(path if ext in [".jpg", ".jpeg", ".png"] else None)
            if gps:
                st.success(f"📍 {up.name} の位置情報 ({gps_source}): {gps}")
            else:
                st.warning(f"📡 {up.name} のGPS情報を取得できませんでした")

use_manual = st.checkbox("📌 手動で位置情報を指定する")
if use_manual:
    lat = st.number_input("緯度", format="%.6f")
    lon = st.number_input("経度", format="%.6f")
    gps_coord = (lat, lon)
    st.success(f"📍 手動設定された座標: {gps_coord}")


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
    st.header("💻 PCカメラ処理 + インデックス抽出")
    idx_type_cam = st.selectbox("📷 使用インデックス", ["VWRI", "VARI", "RI"], key="cam_idx")
    cam_min, cam_max = st.slider("📷 インデックス抽出範囲", -1.0, 1.0, (-0.2, 0.2), step=0.01)
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
            idx_map = compute_index(frm, idx_type_cam)
            mask = ((idx_map >= cam_min) & (idx_map <= cam_max)).astype(np.uint8) * 255
            pct = 100 * np.mean(mask > 0)
            vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            out.write(vis)
            img_area.image(vis, caption=f"{idx_type_cam} 抽出マスク - {pct:.1f}%", use_column_width=True)
            recs.append({"src": f"cam_frame_{len(recs)}", "pct": pct, "gps_lat": gps[0] if gps else None, "gps_lon": gps[1] if gps else None})
            if st.button("停止"): break
        cap.release(); out.release()
        st.video("pc_cam.avi")
        df = pd.DataFrame(recs); st.dataframe(df)
        st.download_button("📥 CSVダウンロード", df.to_csv(index=False), "pc_index_mask.csv")
        if gps:
            m = folium.Map(location=gps, zoom_start=15)
            for row in recs:
                if row["gps_lat"]:
                    folium.Circle([row["gps_lat"], row["gps_lon"]], radius=row["pct"]*10, color='blue', fill=True).add_to(m)
            st.subheader("🗺 PCマップ")
            st_folium(m, width=700, height=500)

# --- スマホカメラモード ---
elif mode == "スマホカメラ":
    st.header("📱 スマホカメラ + GPS + インデックス抽出")
    gps = streamlit_js_eval(js_expressions="navigator.geolocation.getCurrentPosition((p)=>{return {lat:p.coords.latitude, lon:p.coords.longitude};})", key="gps", ttl=1000)
    if isinstance(gps, dict) and 'lat' in gps:
        gps_coord = (gps['lat'], gps['lon'])
        st.success(f"📍 GPS: {gps_coord}")
    else:
        gps_coord = None
        st.warning("📡 位置情報を取得できません（ブラウザの位置許可を確認）")

    idx_type_mobile = st.selectbox("📱 使用インデックス", ["VWRI", "VARI", "RI"], key="mobile_idx")
    mob_min, mob_max = st.slider("📱 インデックス抽出範囲", -1.0, 1.0, (-0.2, 0.2), step=0.01)

    class CamProcessor(VideoProcessorBase):
        def __init__(self): self.pct = 0; self.mask = None; self.history = []
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            idx_map = compute_index(img, idx_type_mobile)
            mask = ((idx_map >= mob_min) & (idx_map <= mob_max)).astype(np.uint8) * 255
            self.pct, self.mask = 100 * np.mean(mask > 0), mask
            if gps_coord:
                self.history.append({"src": f"mobile_frame_{len(self.history)}", "pct": self.pct, "gps_lat": gps_coord[0], "gps_lon": gps_coord[1]})
            return av.VideoFrame.from_ndarray(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), format="bgr24")

    ctx = webrtc_streamer(
        key="mobile",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=CamProcessor,
        media_stream_constraints={"video": {"facingMode": {"exact": "environment"}}, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        st.metric(f"抽出割合", f"{ctx.video_processor.pct:.2f}%")
        st.image(ctx.video_processor.mask, caption="抽出マスク", use_column_width=True)
        df = pd.DataFrame(ctx.video_processor.history)
        if not df.empty:
            st.subheader("📊 スマホ 統計")
            st.dataframe(df)
            st.download_button("📥 CSV", df.to_csv(index=False), "mobile_index_mask.csv")
            if gps_coord:
                m = folium.Map(location=gps_coord, zoom_start=15)
                for _, row in df.iterrows():
                    # 半径を「対象クラスタの割合（%）」に基づくスケーリングに変更（例: 1% → 半径10m）
                    scaled_radius = row["pct"] * 10  # 任意の係数（調整可）
                    folium.Circle(
                        location=[row["gps_lat"], row["gps_lon"]],
                        radius=scaled_radius,
                        color='red',
                        fill=True,
                        fill_opacity=0.6,
                        tooltip=f"{row['pct']:.1f}%"
                    ).add_to(m)
                st.subheader("🗺 スマホマップ")
                st_folium(m, width=700, height=500)
