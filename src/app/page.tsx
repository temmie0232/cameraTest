"use client"
import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocossd from '@tensorflow-models/coco-ssd';
import { IoCameraReverseOutline } from 'react-icons/io5';
import { ClipboardList, Camera, Play, Save } from 'lucide-react';

// 検出されたオブジェクトの型定義
type DetectedObject = {
  class: string;
  score: number;
  bbox?: [number, number, number, number]; // [x, y, width, height]
};

const MobileObjectDetection: React.FC = () => {
  // ===== DOM要素への参照 =====
  const videoRef = useRef<HTMLVideoElement>(null); // カメラ映像表示用
  const canvasRef = useRef<HTMLCanvasElement>(null); // バウンディングボックス描画用

  // ===== 状態管理 =====
  const [model, setModel] = useState<cocossd.ObjectDetection | null>(null); // TensorFlowモデル
  const [isLoading, setIsLoading] = useState(true); // モデル読み込み中フラグ
  const [error, setError] = useState<string | null>(null); // エラーメッセージ
  const [isFrontCamera, setIsFrontCamera] = useState(false); // フロント/リアカメラ切り替えフラグ
  const streamRef = useRef<MediaStream | null>(null); // カメラストリーム参照（クリーンアップ用）
  const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([]); // 検出結果リスト
  const [isPaused, setIsPaused] = useState(false); // カメラ一時停止フラグ（キャプチャ時）
  const [capturedImage, setCapturedImage] = useState<string | null>(null); // キャプチャ画像のBase64データ
  const [lastPredictions, setLastPredictions] = useState<cocossd.DetectedObject[]>([]); // 最新の検出結果（キャプチャ時に使用）

  // ===== カメラセットアップ関数 =====
  // フロント/リアカメラの切り替えとストリーム管理を担当
  const setupCamera = useCallback(async (useFrontCamera = false) => {
    try {
      // 既存のストリームがあれば停止（メモリリーク防止）
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }

      // カメラ制約設定
      // facingMode: "user" = フロントカメラ, "environment" = リアカメラ
      const constraints = {
        video: {
          facingMode: useFrontCamera ? "user" : { ideal: "environment" }
        }
      };

      // カメラストリーム取得
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      // ビデオ要素とキャンバス要素が存在する場合の処理
      if (videoRef.current && canvasRef.current) {
        videoRef.current.srcObject = stream;

        // ビデオメタデータ読み込み完了時の処理
        // レスポンシブ対応：アスペクト比を維持しながらコンテナにフィット
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current && canvasRef.current) {
            const videoWidth = videoRef.current.videoWidth;
            const videoHeight = videoRef.current.videoHeight;

            // 親コンテナのサイズを取得してレスポンシブ対応
            const container = videoRef.current.parentElement;
            if (container) {
              const containerWidth = container.clientWidth;
              const containerHeight = container.clientHeight;

              // アスペクト比を維持しながら最適なスケールを計算
              const scale = Math.min(
                containerWidth / videoWidth,
                containerHeight / videoHeight
              );

              const scaledWidth = videoWidth * scale;
              const scaledHeight = videoHeight * scale;

              // CSS表示サイズを設定（レスポンシブ）
              videoRef.current.style.width = `${scaledWidth}px`;
              videoRef.current.style.height = `${scaledHeight}px`;
              canvasRef.current.style.width = `${scaledWidth}px`;
              canvasRef.current.style.height = `${scaledHeight}px`;

              // キャンバスの実際の描画解像度を設定（高解像度維持）
              canvasRef.current.width = videoWidth;
              canvasRef.current.height = videoHeight;
            }
          }
        };
      }
      setError(null); // エラー状態をクリア
    } catch (err) {
      console.error('カメラエラー:', err);

      // リアカメラで失敗した場合、フロントカメラで再試行
      // モバイルデバイスでリアカメラが利用できない場合の対応
      if (!useFrontCamera) {
        console.log('前面カメラを試行中...');
        try {
          await setupCamera(true);
          setIsFrontCamera(true);
        } catch {
          setError('カメラの起動に失敗しました');
        }
      } else {
        setError('カメラの起動に失敗しました');
      }
    }
  }, []);

  // ===== カメラ切り替えハンドラー =====
  // フロント/リアカメラの切り替えを行う
  const handleToggleCamera = useCallback(async () => {
    const newIsFrontCamera = !isFrontCamera;
    setIsFrontCamera(newIsFrontCamera);
    await setupCamera(newIsFrontCamera);
  }, [isFrontCamera, setupCamera]);

  // ===== バウンディングボックス描画関数 =====
  // 検出されたオブジェクトの周りに枠とラベルを描画
  const drawDetections = useCallback((ctx: CanvasRenderingContext2D, predictions: cocossd.DetectedObject[]) => {
    // 前回の描画をクリア
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    predictions.forEach(prediction => {
      const [x, y, width, height] = prediction.bbox;

      // 緑色のバウンディングボックスを描画
      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);

      // ラベル背景（半透明の黒）を描画
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(x, y - 25, width, 25);

      // ラベルテキスト（オブジェクト名 + 信頼度）を描画
      ctx.fillStyle = '#ffffff';
      ctx.font = '16px "Roboto Mono", monospace';
      ctx.fillText(
        `${prediction.class} ${Math.round(prediction.score * 100)}%`,
        x + 5,
        y - 7
      );
    });
  }, []);

  // ===== 画像キャプチャ機能 =====
  // 現在のビデオフレーム + バウンディングボックスを画像として保存
  const captureImage = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) {
      console.error('ビデオまたはキャンバスの参照が見つかりません');
      return;
    }

    // ビデオが完全に読み込まれているかチェック
    if (videoRef.current.readyState !== 4) {
      console.error('ビデオの準備ができていません');
      return;
    }

    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    if (!context) {
      console.error('キャンバスコンテキストを取得できません');
      return;
    }

    try {
      // キャプチャ専用の一時キャンバスを作成
      // メイン表示用キャンバスを直接操作しないことで表示の安定性を確保
      const captureCanvas = document.createElement('canvas');
      captureCanvas.width = canvas.width;
      captureCanvas.height = canvas.height;
      const captureContext = captureCanvas.getContext('2d');

      if (!captureContext) {
        console.error('キャプチャキャンバスのコンテキストを取得できません');
        return;
      }

      // 現在のビデオフレームをキャプチャキャンバスに描画
      captureContext.drawImage(
        videoRef.current,
        0, 0,
        captureCanvas.width,
        captureCanvas.height
      );

      // 検出結果がある場合はバウンディングボックスも描画
      if (lastPredictions.length > 0) {
        drawDetections(captureContext, lastPredictions);
      }

      // キャンバス内容をBase64画像データに変換
      const imageData = captureCanvas.toDataURL('image/png');

      // 画像データが正常に生成されたかチェック
      if (imageData && imageData !== 'data:,') {
        setCapturedImage(imageData);
        setIsPaused(true); // カメラ一時停止

        // 検出されたオブジェクトの情報を保存（リスト表示用）
        setDetectedObjects(lastPredictions.map(pred => ({
          class: pred.class,
          score: pred.score,
          bbox: pred.bbox
        })));

        console.log('画像のキャプチャが成功しました');
      } else {
        console.error('画像データの生成に失敗しました');
        setError('画像のキャプチャに失敗しました');
      }
    } catch (err) {
      console.error('キャプチャ中にエラーが発生しました:', err);
      setError('画像のキャプチャ中にエラーが発生しました');
    }
  }, [lastPredictions, drawDetections]);

  // ===== カメラ再開機能 =====
  // キャプチャモードからライブカメラモードに戻る
  const resumeCamera = useCallback(() => {
    setIsPaused(false);
    setCapturedImage(null);
  }, []);

  // ===== 画像・データ保存機能 =====
  // キャプチャした画像と検出データをダウンロード
  const saveImage = useCallback(() => {
    if (capturedImage) {
      // 画像ファイルのダウンロード
      const link = document.createElement('a');
      link.href = capturedImage;
      link.download = `object-detection-${Date.now()}.png`;
      link.click();

      // 検出データ（JSON）のダウンロード
      // 後でデータ分析やデバッグに使用可能
      const detectionData = {
        timestamp: new Date().toISOString(),
        detections: detectedObjects
      };
      const dataStr = JSON.stringify(detectionData, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      const dataUrl = URL.createObjectURL(dataBlob);
      const dataLink = document.createElement('a');
      dataLink.href = dataUrl;
      dataLink.download = `object-detection-${Date.now()}.json`;
      dataLink.click();
      URL.revokeObjectURL(dataUrl); // メモリリーク防止
    }
  }, [capturedImage, detectedObjects]);

  // ===== 初期化Effect: カメラセットアップ =====
  useEffect(() => {
    setupCamera(false); // デフォルトでリアカメラを使用

    // クリーンアップ関数：コンポーネント破棄時にカメラストリーム停止
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [setupCamera]);

  // ===== 初期化Effect: TensorFlow.jsとCOCO-SSDモデル読み込み =====
  useEffect(() => {
    const initTF = async () => {
      try {
        // TensorFlow.jsの初期化完了を待機
        await tf.ready();

        // COCO-SSDモデルを読み込み
        // lite_mobilenet_v2: 軽量版（モバイル向け）、速度重視
        // mobilenet_v2: 高精度版、精度重視（重い）
        const loadedModel = await cocossd.load({
          base: 'lite_mobilenet_v2'
        });
        setModel(loadedModel);
        setIsLoading(false);
      } catch (err) {
        console.error('初期化エラー:', err);
        setError(err instanceof Error ? err.message : '不明なエラーが発生しました');
        setIsLoading(false);
      }
    };

    initTF();
  }, []);

  // ===== メインループ: リアルタイム物体検出 =====
  useEffect(() => {
    // 必要な要素とモデルが準備できていない場合は処理しない
    if (!model || !videoRef.current || !canvasRef.current || isPaused) return;

    let animationId: number;

    // 物体検出の実行関数
    const detectObjects = async () => {
      try {
        // ビデオが再生可能状態かチェック（readyState === 4）
        if (videoRef.current?.readyState === 4) {
          // COCO-SSDモデルで物体検出実行
          const predictions = await model.detect(videoRef.current);
          setLastPredictions(predictions); // キャプチャ用に保存

          // キャンバスにバウンディングボックスを描画
          const ctx = canvasRef.current?.getContext('2d');
          if (!ctx) return;

          drawDetections(ctx, predictions);

          // UIリスト更新用に検出結果を整形
          setDetectedObjects(predictions.map(pred => ({
            class: pred.class,
            score: pred.score,
            bbox: pred.bbox
          })));
        }

        // 次のフレームで再実行（60FPS目安）
        animationId = requestAnimationFrame(detectObjects);
      } catch (err) {
        console.error('検出エラー:', err);
        setError('物体検出中にエラーが発生しました');
      }
    };

    // 検出ループ開始
    detectObjects();

    // クリーンアップ：アニメーションフレームをキャンセル
    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [model, isPaused, drawDetections]);

  // ===== UIレンダリング =====
  return (
    <div className="fixed inset-0 bg-gradient-to-b from-gray-900 to-black" style={{ fontFamily: '"Roboto Mono", monospace' }}>
      {/* ===== ヘッダーバー ===== */}
      <div className="absolute top-0 left-0 right-0 h-16 bg-gradient-to-b from-black/80 to-transparent z-10 flex items-center justify-between px-4">
        {/* アプリタイトル */}
        <div className="flex items-center space-x-2 text-white/80">
          <Camera className="w-6 h-6" />
          <span className="font-medium">物体検出 (COCO-SSD)</span>
        </div>

        {/* カメラ切り替えボタン */}
        <button
          onClick={handleToggleCamera}
          className="p-2 rounded-full bg-white/10 hover:bg-white/20 transition-all"
          disabled={isPaused} // キャプチャ中は無効
        >
          <IoCameraReverseOutline className="w-6 h-6 text-white" />
        </button>
      </div>

      {/* ===== メイン表示エリア（ビデオ + キャンバス） ===== */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="relative rounded-lg overflow-hidden shadow-2xl w-full h-full max-w-3xl max-h-[80vh]">
          {/* ビデオ要素：カメラ映像表示 */}
          <video
            ref={videoRef}
            autoPlay
            playsInline // モバイル対応：インライン再生
            className="absolute inset-0 object-contain"
            style={{
              display: isPaused ? 'none' : 'block' // キャプチャ中は非表示
            }}
          />

          {/* キャンバス要素：バウンディングボックス描画 */}
          <canvas
            ref={canvasRef}
            className="absolute inset-0 object-contain"
          />
        </div>
      </div>

      {/* ===== 下部コントロールパネル ===== */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
        <div className="max-w-md mx-auto">
          {/* 検出結果リスト */}
          <div className="bg-white/10 backdrop-blur-md rounded-lg p-3 mb-4">
            <div className="flex items-center space-x-2 mb-2">
              <ClipboardList className="w-5 h-5 text-white" />
              <span className="text-white/90 text-sm font-medium">検出オブジェクト</span>
            </div>

            {/* 検出されたオブジェクトのリスト表示 */}
            <div className="space-y-1">
              {detectedObjects.map((obj, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between text-sm px-2 py-1 rounded bg-white/5"
                >
                  <span className="text-white/80">{obj.class}</span>
                  <span className="text-white">{Math.round(obj.score * 100)}%</span>
                </div>
              ))}
            </div>
          </div>

          {/* ボタンコントロール */}
          <div className="flex justify-center items-center space-x-4">
            {!isPaused ? (
              // ライブモード：キャプチャボタンのみ表示
              <button
                onClick={captureImage}
                className="w-16 h-16 rounded-full bg-white/20 hover:bg-white/30 flex items-center justify-center transition-all"
              >
                <Camera className="w-8 h-8 text-white" />
              </button>
            ) : (
              // キャプチャモード：再生・無効化されたキャプチャ・保存ボタン
              <div className="flex items-center space-x-4">
                {/* カメラ再開ボタン */}
                <button
                  onClick={resumeCamera}
                  className="w-12 h-12 rounded-full bg-white/20 hover:bg-white/30 flex items-center justify-center transition-all"
                >
                  <Play className="w-6 h-6 text-white" />
                </button>

                {/* 無効化されたキャプチャボタン */}
                <button
                  className="w-16 h-16 rounded-full bg-gray-500/50 flex items-center justify-center cursor-not-allowed"
                  disabled
                >
                  <Camera className="w-8 h-8 text-gray-400" />
                </button>

                {/* 保存ボタン */}
                <button
                  onClick={saveImage}
                  className="w-12 h-12 rounded-full bg-white/20 hover:bg-white/30 flex items-center justify-center transition-all"
                >
                  <Save className="w-6 h-6 text-white" />
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ===== ローディング画面 ===== */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="text-center text-white">
            <div className="w-12 h-12 border-4 border-white border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="font-medium">読み込み中...</p>
          </div>
        </div>
      )}

      {/* ===== エラー表示 ===== */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="bg-gray-900/90 backdrop-blur-md text-white p-4 rounded-lg mx-4 text-center">
            <p className="font-medium">エラーが発生しました</p>
            <p className="text-sm mt-2 text-white/80">{error}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default MobileObjectDetection;