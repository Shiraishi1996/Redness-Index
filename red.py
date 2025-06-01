# streamlit_app.py
"""
拡張機能一覧（完全版）
----------------------
✅ 画像 / 動画 アップロード処理（4分割解析・GPSマップ・AVI 書出し・CSV）
✅ PC カメラリアルタイム処理（クラスタリング・AVI 保存・CSV・GPS/IP マップ）
✅ **スマホブラウザ用 カメラ + GPS リアルタイム処理**（streamlit‑webrtc / JS Geolocation）

追加依存:
```
streamlit-webrtc
av
streamlit-js-eval
```
"""

import streamlit as st
import numpy as np
import cv2
import pandas as pd
import av
from matplotlib import cm
from sklearn.cluster import KMeans
from tempfile import TemporaryDirectory
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import folium
from streamlit_folium import st_folium
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from streamlit_js_eval import streamlit_js_eval
import geocoder

# ----------------- UI 共通 -----------------
st.set_page_config(layout="wide")
st.title("🌿 クラスタリングアプリ（アップロード / PCカメラ / スマホカメラ + GPS）")

mode = st.radio("モード選択", ["アップロード処理", "PCカメラ", "スマホカメラ"], horizontal=True)

k = st.slider("クラスタ数 (k)", 2, 10, 6)
resize_ratio = st.slider("手動リサイズ倍率 (画像のみ)", 0.1, 1.0, 1.0)
index_type = st.selectbox("インデックス", ["RI", "VARI", "VWRI"])
cluster_id = st.number_input("抽出クラスタ番号", 0, k-1, 0)

# ----------------- 共通処理 -----------------

def compute_index(img: np.ndarray):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = [rgb[:,:,i].astype(np.float32) for i in range(3)]
    safe = lambda n,d: n/np.where(d==0,1e-5,d)
    if index_type == "RI":
        return safe(g-r, g+r)
    if index_type == "VARI":
        return safe(g-r, g+r-b)
    return safe(g-r-b, g+r+b)  # VWRI

def kmeans_label(idx):
    flat = idx.reshape(-1,1)
    lbl = KMeans(n_clusters=k, random_state=42).fit_predict(flat).reshape(idx.shape)
    return lbl

def visualize(idx,lbl):
    order = sorted([(i,idx[lbl==i].mean()) for i in range(k)], key=lambda x:x[1], reverse=True)
    cmap=cm.get_cmap('RdYlGn',k)
    id2col={cid:(np.array(cmap(r/(k-1))[:3])*255).astype(np.uint8) for r,(cid,_) in enumerate(order)}
    vis=np.zeros((*lbl.shape,3),np.uint8)
    for cid in range(k): vis[lbl==cid]=id2col[cid]
    mask=(lbl==cluster_id).astype(np.uint8)*255
    pct=100*np.mean(lbl==cluster_id)
    return vis,mask,pct

def analyze(img):
    idx=compute_index(img)
    lbl=kmeans_label(idx)
    return visualize(idx,lbl)

def quadrants(img):
    h,w=img.shape[:2]
    return {
        "全体":img,
        "1_左上":img[:h//2,:w//2],
        "2_右上":img[:h//2,w//2:],
        "3_左下":img[h//2:,:w//2],
        "4_右下":img[h//2:,w//2:]
    }

# ---- GPS helper ----

def exif_gps(path:str):
    try:
        img=Image.open(path); exif=img._getexif() or {}
        gps_info={GPSTAGS.get(t):v for t,v in exif.get(next((k for k,v in TAGS.items() if v=='GPSInfo'),0),{}).items()}
        if {'GPSLatitude','GPSLongitude'}.issubset(gps_info):
            dms=lambda x: x[0][0]/x[0][1]+x[1][0]/x[1][1]/60+x[2][0]/x[2][1]/3600
            lat=dms(gps_info['GPSLatitude']); lon=dms(gps_info['GPSLongitude'])
            if gps_info.get('GPSLatitudeRef','N')=='S': lat=-lat
            if gps_info.get('GPSLongitudeRef','E')=='W': lon=-lon
            return lat,lon
    except: pass
    return None

def ip_gps():
    g=geocoder.ip('me'); return g.latlng if g and g.ok else None

# ----------------- アップロード処理 -----------------
if mode=="アップロード処理":
    st.header("📂 アップロード処理")
    files=st.file_uploader("画像/動画をアップロード",type=["jpg","jpeg","png","mp4","avi"],accept_multiple_files=True)
    if files:
        stats=[]; gps_pts=[]
        with TemporaryDirectory() as tmp:
            for up in files:
                p=os.path.join(tmp,up.name); Path(p).write_bytes(up.read())
                ext=Path(p).suffix.lower()
                if ext in {'.jpg','.jpeg','.png'}:
                    img=cv2.imread(p)
                    if resize_ratio!=1.0: img=cv2.resize(img,(0,0),fx=resize_ratio,fy=resize_ratio)
                    gps=exif_gps(p)
                    for nm,rg in quadrants(img).items():
                        vis,mask,pct=analyze(rg)
                        st.image(vis,caption=f"{up.name}-{nm}")
                        cv2.imwrite(os.path.join(tmp,f"mask_{up.name}_{nm}.png"),mask)
                        stats.append({"src":f"{up.name}_{nm}","pct":pct,"gps":gps})
                    if gps: gps_pts.append((gps,pct))
                elif ext in {'.mp4','.avi'}:
                    cap=cv2.VideoCapture(p)
                    w,h=int(cap.get(3)),int(cap.get(4)); fps=cap.get(5)
                    out_p=os.path.join(tmp,Path(up.name).stem+"_proc.avi")
                    out=cv2.VideoWriter(out_p,cv2.VideoWriter_fourcc(*'XVID'),fps,(w,h))
                    prog=st.progress(0); total=int(cap.get(7))
                    for i in range(total):
                        r,fr=cap.read();
                        if not r: break
                        vis,mask,pct=analyze(fr); out.write(vis)
                        stats.append({"src":f"{up.name}_fr{i}","pct":pct,"gps":None})
                        prog.progress((i+1)/total)
                    cap.release(); out.release()
                    st.video(out_p)
                    with open(out_p,'rb') as vf: st.download_button("処理動画DL",vf.read(),file_name=Path(out_p).name)
        # 地図を上に表示
        if gps_pts:
            st.subheader("📍 GPS マップ")
            m=folium.Map(location=gps_pts[0][0],zoom_start=6)
            for (latlon,p) in gps_pts:
                folium.Circle(latlon,radius=p*10,color='green',fill=True).add_to(m)
            st_folium(m,width=700,height=500)
        df=pd.DataFrame(stats); st.dataframe(df)
        st.download_button("CSV",df.to_csv(index=False),"upload_stats.csv")

# ----------------- PC カメラ -----------------
elif mode=="PCカメラ":
    st.header("💻 PC カメラ処理")
    cams=[i for i in range(5) if cv2.VideoCapture(i).read()[0]]
    cam_id=st.selectbox("カメラ選択",cams)
    run=st.checkbox("開始")
    if run:
        cap=cv2.VideoCapture(cam_id)
        out=cv2.VideoWriter("pccam.avi",cv2.VideoWriter_fourcc(*'XVID'),20,(640,480))
        img_area=st.empty(); stop=st.button("停止")
        rt_gps=ip_gps(); recs=[]; fid=0
        while cap.isOpened() and not stop:
            ok,frm=cap.read();
            if not ok: break
            frm=cv2.resize(frm,(640,480))
            vis,mask,pct=analyze(frm)
            out.write(vis)
            img_area.image(vis,caption=f"pct={pct:.1f}%",use_column_width=True)
            recs.append({"src":f"pc_{fid}","pct":pct,"gps":rt_gps}); fid+=1
        cap.release(); out.release()
        st.video("pccam.avi")
        if rt_gps:
            m=folium.Map(location=rt_gps,zoom_start=6); folium.Marker(rt_gps).add_to(m)
            st_folium(m,width=700,height=500)

# ----------------- スマホカメラ -----------------
else:
    st.header("📱 スマホカメラ + GPS")
    gps_js=streamlit_js_eval(js_expressions="navigator.geolocation.getCurrentPosition((p)=>{return {lat:p.coords.latitude, lon:p.coords.longitude};})",key="gps",ttl=1000)
    current_gps=None
    if isinstance(gps_js,dict) and 'lat' in gps_js:
        current_gps=(gps_js['lat'],gps_js['lon'])
        st.success(f"GPS: {current_gps}")
    else:
        st.warning("位置情報の許可をしてください")

    class CamProcessor(VideoProcessorBase):
        def __init__(self):
            self.pct=0; self.mask=None
        def recv(self,frame):
            img=frame.to_ndarray(format="bgr24")
            vis,mask,pct=analyze(img)
            self.pct,self.mask=pct,mask
            return av.VideoFrame.from_ndarray(vis,format="bgr24")

    ctx=webrtc_streamer(key="mobile",mode=WebRtcMode.SENDRECV,video_processor_factory=CamProcessor,media_stream_constraints={"video":True,"audio":False},async_processing=True)

    if ctx.video_processor:
        st.metric(f"クラスタ{cluster_id} %",f"{ctx.video_processor.pct:.2f}")
        st.image(ctx.video_processor.mask,caption="Mask",use_column_width=True)
    if current_gps:
        m=folium.Map(location=current_gps,zoom_start=15); folium.Marker(current
