"use client"
import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocossd from '@tensorflow-models/coco-ssd';
import { IoCameraReverseOutline } from 'react-icons/io5';
import { ClipboardList, Camera, Play, Save } from 'lucide-react';

type DetectedObject = {
  class: string;
  score: number;
  bbox?: [number, number, number, number];
};

const MobileObjectDetection: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [model, setModel] = useState<cocossd.ObjectDetection | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isFrontCamera, setIsFrontCamera] = useState(false);
  const streamRef = useRef<MediaStream | null>(null);
  const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([]);
  const [isPaused, setIsPaused] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [lastPredictions, setLastPredictions] = useState<cocossd.DetectedObject[]>([]);

  const setupCamera = useCallback(async (useFrontCamera = false) => {
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }

      const constraints = {
        video: {
          facingMode: useFrontCamera ? "user" : { ideal: "environment" }
        }
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current && canvasRef.current) {
        videoRef.current.srcObject = stream;

        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current && canvasRef.current) {
            const videoWidth = videoRef.current.videoWidth;
            const videoHeight = videoRef.current.videoHeight;

            // コンテナのサイズを取得
            const container = videoRef.current.parentElement;
            if (container) {
              const containerWidth = container.clientWidth;
              const containerHeight = container.clientHeight;

              // アスペクト比を維持しながら、コンテナにフィットするサイズを計算
              const scale = Math.min(
                containerWidth / videoWidth,
                containerHeight / videoHeight
              );

              const scaledWidth = videoWidth * scale;
              const scaledHeight = videoHeight * scale;

              // ビデオとキャンバスのサイズを設定
              videoRef.current.style.width = `${scaledWidth}px`;
              videoRef.current.style.height = `${scaledHeight}px`;
              canvasRef.current.style.width = `${scaledWidth}px`;
              canvasRef.current.style.height = `${scaledHeight}px`;

              // キャンバスの実際の解像度を設定
              canvasRef.current.width = videoWidth;
              canvasRef.current.height = videoHeight;
            }
          }
        };
      }
      setError(null);
    } catch (err) {
      console.error('カメラエラー:', err);
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

  const handleToggleCamera = useCallback(async () => {
    const newIsFrontCamera = !isFrontCamera;
    setIsFrontCamera(newIsFrontCamera);
    await setupCamera(newIsFrontCamera);
  }, [isFrontCamera, setupCamera]);

  const drawDetections = useCallback((ctx: CanvasRenderingContext2D, predictions: cocossd.DetectedObject[]) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    predictions.forEach(prediction => {
      const [x, y, width, height] = prediction.bbox;

      // 緑色のバウンディングボックス
      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);

      // 半透明の黒背景（ラベル用）
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(x, y - 25, width, 25);

      // 白色のテキスト
      ctx.fillStyle = '#ffffff';
      ctx.font = '16px "Roboto Mono", monospace';
      ctx.fillText(
        `${prediction.class} ${Math.round(prediction.score * 100)}%`,
        x + 5,
        y - 7
      );
    });
  }, []);

  const captureImage = useCallback(async () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      if (context) {
        // 一時的なキャンバスを作成
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = canvas.width;
        tempCanvas.height = canvas.height;
        const tempContext = tempCanvas.getContext('2d');

        if (tempContext) {
          // まず、ビデオフレームを一時的なキャンバスに描画
          tempContext.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

          // メインのキャンバスをクリア
          context.clearRect(0, 0, canvas.width, canvas.height);

          // 一時的なキャンバスの内容をメインのキャンバスにコピー
          context.drawImage(tempCanvas, 0, 0);

          // バウンディングボックスを描画
          if (lastPredictions.length > 0) {
            drawDetections(context, lastPredictions);
          }

          const imageData = canvas.toDataURL('image/png');
          setCapturedImage(imageData);
          setIsPaused(true);

          setDetectedObjects(lastPredictions.map(pred => ({
            class: pred.class,
            score: pred.score,
            bbox: pred.bbox
          })));
        }
      }
    }
  }, [lastPredictions, drawDetections]);

  const resumeCamera = useCallback(() => {
    setIsPaused(false);
    setCapturedImage(null);
  }, []);

  const saveImage = useCallback(() => {
    if (capturedImage) {
      const link = document.createElement('a');
      link.href = capturedImage;
      link.download = `object-detection-${Date.now()}.png`;
      link.click();

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
      URL.revokeObjectURL(dataUrl);
    }
  }, [capturedImage, detectedObjects]);

  useEffect(() => {
    setupCamera(false);

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [setupCamera]);

  useEffect(() => {
    const initTF = async () => {
      try {
        await tf.ready();
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

  useEffect(() => {
    if (!model || !videoRef.current || !canvasRef.current || isPaused) return;

    let animationId: number;
    const detectObjects = async () => {
      try {
        if (videoRef.current?.readyState === 4) {
          const predictions = await model.detect(videoRef.current);
          setLastPredictions(predictions);

          const ctx = canvasRef.current?.getContext('2d');
          if (!ctx) return;

          drawDetections(ctx, predictions);

          setDetectedObjects(predictions.map(pred => ({
            class: pred.class,
            score: pred.score,
            bbox: pred.bbox
          })));
        }

        animationId = requestAnimationFrame(detectObjects);
      } catch (err) {
        console.error('検出エラー:', err);
        setError('物体検出中にエラーが発生しました');
      }
    };

    detectObjects();

    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [model, isPaused, drawDetections]);

  return (
    <div className="fixed inset-0 bg-gradient-to-b from-gray-900 to-black" style={{ fontFamily: '"Roboto Mono", monospace' }}>
      <div className="absolute top-0 left-0 right-0 h-16 bg-gradient-to-b from-black/80 to-transparent z-10 flex items-center justify-between px-4">
        <div className="flex items-center space-x-2 text-white/80">
          <Camera className="w-6 h-6" />
          <span className="font-medium">物体検出 (COCO-SSD)</span>
        </div>
        <button
          onClick={handleToggleCamera}
          className="p-2 rounded-full bg-white/10 hover:bg-white/20 transition-all"
          disabled={isPaused}
        >
          <IoCameraReverseOutline className="w-6 h-6 text-white" />
        </button>
      </div>

      <div className="absolute inset-0 flex items-center justify-center">
        <div className="relative rounded-lg overflow-hidden shadow-2xl w-full h-full max-w-3xl max-h-[80vh]">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="absolute inset-0 object-contain"
            style={{
              display: isPaused ? 'none' : 'block'
            }}
          />
          <canvas
            ref={canvasRef}
            className="absolute inset-0 object-contain"
          />
        </div>
      </div>

      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
        <div className="max-w-md mx-auto">
          <div className="bg-white/10 backdrop-blur-md rounded-lg p-3 mb-4">
            <div className="flex items-center space-x-2 mb-2">
              <ClipboardList className="w-5 h-5 text-white" />
              <span className="text-white/90 text-sm font-medium">検出オブジェクト</span>
            </div>
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

          <div className="flex justify-center items-center space-x-4">
            {!isPaused ? (
              <button
                onClick={captureImage}
                className="w-16 h-16 rounded-full bg-white/20 hover:bg-white/30 flex items-center justify-center transition-all"
              >
                <Camera className="w-8 h-8 text-white" />
              </button>
            ) : (
              <div className="flex items-center space-x-4">
                <button
                  onClick={resumeCamera}
                  className="w-12 h-12 rounded-full bg-white/20 hover:bg-white/30 flex items-center justify-center transition-all"
                >
                  <Play className="w-6 h-6 text-white" />
                </button>
                <button
                  className="w-16 h-16 rounded-full bg-gray-500/50 flex items-center justify-center cursor-not-allowed"
                  disabled
                >
                  <Camera className="w-8 h-8 text-gray-400" />
                </button>
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

      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="text-center text-white">
            <div className="w-12 h-12 border-4 border-white border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="font-medium">読み込み中...</p>
          </div>
        </div>
      )}

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