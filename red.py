# streamlit_app.py
"""
拡張機能一覧
-------------
- 画像/動画アップロード（複数可）
- 画像は自動４分割（全体＋左上/右上/左下/右下）解析
- 手動リサイズ倍率の指定
- インデックス（RI/VARI/VWRI）計算 → KMeansクラスタリング
- **クラスタ番号指定でマスク抽出**
- 動画は処理済みフレームを逐次 `.avi` 保存＋進捗バー
- GPS(EXIF または IP) 取得 → Folium 円描画
- クラスタ統計を CSV ダウンロード
- **リアルタイムモード**：カメラ選択＋AVI保存＋マスク保存＋CSV＋地図
"""

import streamlit as st
import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import cm
from tempfile import TemporaryDirectory
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import folium
from streamlit_folium import st_folium
import geocoder

# ----------------- UI -----------------
st.set_page_config(layout="wide")
st.title("🌿 クラスタリングアプリ（画像/動画 + 4分割 + CSV/AVI + GPSマップ）")

mode = st.radio("モード選択", ["アップロード処理", "リアルタイム処理"], horizontal=True)

k = st.slider("クラスタ数 (k)", 2, 10, 6)
resize_ratio = st.slider("手動リサイズ倍率 (全体のみ)", 0.1, 1.0, 1.0)
index_type = st.selectbox("インデックス", ["RI", "VARI", "VWRI"])
cluster_id = st.number_input("抽出するクラスタ番号", min_value=0, max_value=k-1, value=0, step=1)

# ----------------- 基本ユーティリティ -----------------

def compute_index(img: np.ndarray, ind: str) -> np.ndarray:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = [rgb[:,:,i].astype(np.float32) for i in range(3)]
    safe = lambda n, d: n / np.where(d == 0, 1e-5, d)
    if ind == "RI":
        return safe(g - r, g + r)
    if ind == "VARI":
        return safe(g - r, g + r - b)
    if ind == "VWRI":
        return safe(g - r - b, g + r + b)
    raise ValueError

def kmeans_cluster(index_map: np.ndarray, k: int):
    flat = index_map.reshape(-1,1)
    km = KMeans(n_clusters=k, random_state=42).fit(flat)
    return km.labels_.reshape(index_map.shape)

def visualize(labels: np.ndarray, index_map: np.ndarray, k: int):
    means = sorted([(i,index_map[labels==i].mean()) for i in range(k)], key=lambda x:x[1], reverse=True)
    cmap = cm.get_cmap('RdYlGn', k)
    id2col = {cid:(np.array(cmap(rank/(k-1))[:3])*255).astype(np.uint8) for rank,(cid,_) in enumerate(means)}
    vis = np.zeros((*labels.shape,3), np.uint8)
    for cid in range(k):
        vis[labels==cid] = id2col[cid]
    return vis

def analyze(img: np.ndarray):
    idx = compute_index(img, index_type)
    labels = kmeans_cluster(idx, k)
    vis = visualize(labels, idx, k)
    mask = (labels==cluster_id).astype(np.uint8)*255
    pct = 100*np.sum(labels==cluster_id)/labels.size
    return vis, mask, pct

def quadrants(img):
    h,w = img.shape[:2]
    return {
        "全体": img,
        "1_左上": img[:h//2,:w//2],
        "2_右上": img[:h//2,w//2:],
        "3_左下": img[h//2:,:w//2],
        "4_右下": img[h//2:,w//2:]
    }

# ---- GPS helpers ----

def dms2deg(dms, ref):
    d,m,s = [v[0]/v[1] for v in dms]
    deg = d+m/60+s/3600
    return -deg if ref in ['S','W'] else deg

def exif_gps(path):
    try:
        img = Image.open(path)
        exif = img._getexif() or {}
        gps_info = {GPSTAGS.get(t):v for t,v in exif.get(next((k for k,v in TAGS.items() if v=='GPSInfo'),0), {}).items()}
        if {'GPSLatitude','GPSLongitude'}.issubset(gps_info):
            lat = dms2deg(gps_info['GPSLatitude'], gps_info.get('GPSLatitudeRef','N'))
            lon = dms2deg(gps_info['GPSLongitude'], gps_info.get('GPSLongitudeRef','E'))
            return lat,lon
    except: pass
    return None

def current_ip_gps():
    g = geocoder.ip('me')
    return g.latlng if g and g.ok else None

# ---- camera list ----

def list_cams(max_dev=5):
    cams=[]
    for i in range(max_dev):
        cap=cv2.VideoCapture(i)
        if cap.read()[0]: cams.append(i)
        cap.release()
    return cams

# ----------------- リアルタイム処理 -----------------
if mode=="リアルタイム処理":
    cams = list_cams()
    cam_id = st.selectbox("カメラ選択", cams, format_func=lambda x:f"Camera {x}")
    start = st.checkbox("リアルタイム開始")
    stop = st.button("⏹ 停止")
    if stop: st.session_state["rt_stop"] = True
    recs=[]
    if start and not st.session_state.get("rt_stop", False):
        st.session_state["rt_stop"] = False
        cap=cv2.VideoCapture(cam_id)
        h,w = 480,640
        out=cv2.VideoWriter("realtime.avi",cv2.VideoWriter_fourcc(*'XVID'),20,(w,h))
        f1,f2,f3 = st.empty(),st.empty(),st.empty()
        gps_rt=current_ip_gps()
        fid=0
        with TemporaryDirectory() as tmp:
            while cap.isOpened():
                r,frm=cap.read()
                if not r: break
                frm=cv2.resize(frm,(w,h))
                vis,mask,pct=analyze(frm)
                out.write(vis)
                f1.image(cv2.cvtColor(frm,cv2.COLOR_BGR2RGB),use_column_width=True)
                f2.image(vis,use_column_width=True)
                f3.image(mask,use_column_width=True)
                cv2.imwrite(os.path.join(tmp,f"mask_{fid}.png"),mask)
                recs.append({"src":f"cam_{fid}","pct":pct,"gps":gps_rt})
                fid+=1
                if st.session_state.get("rt_stop",False): break
        cap.release(); out.release()
        if gps_rt:
            m = folium.Map(location=gps_rt, zoom_start=6)
            for r in recs:
                if r["gps"] is not None:
                    folium.Circle(
                        r["gps"], radius=r["pct"]*10,
                        color='blue', fill=True
                    ).add_to(m)
            st_folium(m, width=700, height=500)
            st.video("realtime.avi")
            df=pd.DataFrame(recs); st.dataframe(df)
            st.download_button("CSV",df.to_csv(index=False),"rt_stats.csv")

        else:
            st.warning("⚠️ IPからGPS位置が取得できませんでした。地図は表示されません。")


# ----------------- アップロード処理 -----------------
if mode=="アップロード処理":
    files=st.file_uploader("画像/動画をアップロード",type=["jpg","jpeg","png","mp4","avi"],accept_multiple_files=True)
    if files:
        stats=[]; gps_points=[]
        with TemporaryDirectory() as tmp:
            for f in files:
                p=os.path.join(tmp,f.name); Path(p).write_bytes(f.read())
                ext=Path(f.name).suffix.lower()
                if ext in {'.jpg','.jpeg','.png'}:
                    img=cv2.imread(p)
                    if resize_ratio!=1.0:
                        img=cv2.resize(img,(0,0),fx=resize_ratio,fy=resize_ratio)
                    gps=exif_gps(p)
                    for name,reg in quadrants(img).items():
                        vis,mask,pct=analyze(reg)
                        st.image(vis,caption=f"{f.name}-{name}")
                        cv2.imwrite(os.path.join(tmp,f"mask_{f.name}_{name}.png"),mask)
                        stats.append({"src":f"{f.name}_{name}","pct":pct,"gps":gps})
                    if gps: gps_points.append((gps,pct))
                elif ext in {'.mp4','.avi'}:
                    cap=cv2.VideoCapture(p)
                    w,h=int(cap.get(3)),int(cap.get(4))
                    fps=cap.get(5)
                    out_path=os.path.join(tmp,Path(f.name).stem+'_proc.avi')
                    out=cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*'XVID'),fps,(w,h))
                    prog=st.progress(0); cnt=int(cap.get(7))
                    for i in range(cnt):
                        r,fr=cap.read()
                        if not r: break
                        vis,mask,pct=analyze(fr)
                        out.write(vis)
                        stats.append({"src":f"{f.name}_fr{i}","pct":pct,"gps":None})
                        prog.progress((i+1)/cnt)
                    cap.release(); out.release()
                    st.video(out_path)
                    with open(out_path,'rb') as v: 
                        st.download_button("動画DL",v.read(),file_name=Path(out_path).name)
        if gps_points:
            m=folium.Map(location=gps_points[0][0],zoom_start=6)
            for (latlon,p) in gps_points:
                folium.Circle(latlon,radius=p*10,color='green',fill=True).add_to(m)
            st_folium(m,width=700,height=500)
        df=pd.DataFrame(stats); st.dataframe(df)
        st.download_button("CSV",df.to_csv(index=False),"upload_stats.csv")
