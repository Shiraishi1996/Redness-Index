// Cesium Viewerの初期化
Cesium.Ion.defaultAccessToken = 'YOUR_CESIUM_ION_ACCESS_TOKEN'; // 必要に応じてAPIキーを修正
const viewer = new Cesium.Viewer('cesiumContainer', {
  terrainProvider: Cesium.createWorldTerrain()
});

// 3Dタイルセットの読み込み
const tilesetUrl = './tileset.json'; // tileset.jsonのパス
const tileset = new Cesium.Cesium3DTileset({ url: tilesetUrl });
viewer.scene.primitives.add(tileset);

// 現在地を示すエンティティ
let userMarker = viewer.entities.add({
  position: Cesium.Cartesian3.fromDegrees(0, 0, 0),
  point: {
    pixelSize: 10,
    color: Cesium.Color.BLUE,
  },
  label: {
    text: '現在地',
    font: '14pt sans-serif',
    fillColor: Cesium.Color.WHITE,
    showBackground: true
  }
});

// 最寄りの建物を示すエンティティ
let nearestBuildingHighlight = null;

// タイルセットの準備完了後に処理を開始
tileset.readyPromise
  .then(() => {
    console.log("タイルセットが読み込まれました");

    // タイルセットを中心にズーム
    viewer.zoomTo(tileset);

    // 現在地をリアルタイムで取得
    if (navigator.geolocation) {
      navigator.geolocation.watchPosition(
        (position) => {
          const userLat = position.coords.latitude;
          const userLon = position.coords.longitude;
          const userHeight = 0; // 必要に応じて高さを設定

          // 現在地を更新
          const userPosition = Cesium.Cartesian3.fromDegrees(userLon, userLat, userHeight);
          userMarker.position = userPosition;

          // 最も近い建物を計算
          findNearestBuilding(userPosition, tileset);

          // カメラを現在地に追従させる
          viewer.camera.flyTo({
            destination: userPosition,
            duration: 1.5, // カメラ移動の時間
          });
        },
        (error) => {
          console.error("位置情報の取得に失敗しました:", error.message);
        },
        { enableHighAccuracy: true, maximumAge: 0, timeout: 5000 }
      );
    } else {
      console.error("このブラウザでは位置情報がサポートされていません");
    }
  })
  .otherwise((error) => {
    console.error("タイルセットの読み込み中にエラーが発生しました:", error);
  });

// 最も近い建物を特定してハイライトする関数
function findNearestBuilding(userPosition, tileset) {
  let nearestFeature = null;
  let minDistance = Infinity;

  // タイル内の建物をチェック
  tileset.tileVisible.addEventListener((tile) => {
    if (tile.content.featuresLength) {
      for (let i = 0; i < tile.content.featuresLength; i++) {
        const feature = tile.content.getFeature(i);
        const boundingVolume = feature.boundingVolume;

        // バウンディングボリュームがBoundingSphereの場合
        if (boundingVolume instanceof Cesium.BoundingSphere) {
          const center = Cesium.Matrix4.multiplyByPoint(
            tile.content.boundingVolume.boundingVolumeMatrix,
            boundingVolume.center,
            new Cesium.Cartesian3()
          );

          const distance = Cesium.Cartesian3.distance(userPosition, center);

          if (distance < minDistance) {
            minDistance = distance;
            nearestFeature = feature;
          }
        }
      }
    }
  });

  // 最も近い建物をハイライト
  setTimeout(() => {
    if (nearestFeature) {
      // 以前のハイライトを削除
      if (nearestBuildingHighlight) {
        viewer.scene.primitives.remove(nearestBuildingHighlight);
      }

      // 新しいハイライトを追加
      nearestBuildingHighlight = new Cesium.Primitive({
        geometryInstances: new Cesium.GeometryInstance({
          geometry: nearestFeature.geometry,
          attributes: {
            color: Cesium.ColorGeometryInstanceAttribute.fromColor(
              Cesium.Color.RED.withAlpha(0.6)
            )
          }
        })
      });

      viewer.scene.primitives.add(nearestBuildingHighlight);
      console.log("最寄りの建物がハイライトされました", nearestFeature, `距離: ${minDistance.toFixed(2)} m`);
    } else {
      console.log("最寄りの建物が見つかりませんでした");
    }
  }, 1000);
}
