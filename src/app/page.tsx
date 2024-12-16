"use client"
import React, { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocossd from '@tensorflow-models/coco-ssd';
import { IoCameraReverseOutline } from 'react-icons/io5';

const MobileObjectDetection = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isFrontCamera, setIsFrontCamera] = useState(false);
  const streamRef = useRef(null);

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
        setError(err.message);
        setIsLoading(false);
      }
    };

    initTF();
  }, []);

  const setupCamera = async (useFrontCamera = false) => {
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }

      const constraints = {
        video: {
          facingMode: useFrontCamera ? "user" : { ideal: "environment" },
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        }
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;

        videoRef.current.onloadedmetadata = () => {
          if (canvasRef.current) {
            canvasRef.current.width = videoRef.current.videoWidth;
            canvasRef.current.height = videoRef.current.videoHeight;
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
        } catch (frontErr) {
          setError('カメラの起動に失敗しました');
        }
      } else {
        setError('カメラの起動に失敗しました');
      }
    }
  };

  useEffect(() => {
    setupCamera(false);

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const toggleCamera = async () => {
    setIsFrontCamera(!isFrontCamera);
    await setupCamera(!isFrontCamera);
  };

  useEffect(() => {
    if (!model || !videoRef.current || !canvasRef.current) return;

    let animationId;
    const detectObjects = async () => {
      try {
        if (videoRef.current.readyState === 4) {
          const predictions = await model.detect(videoRef.current);

          const ctx = canvasRef.current.getContext('2d');
          ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

          predictions.forEach(prediction => {
            const [x, y, width, height] = prediction.bbox;

            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 4;
            ctx.strokeRect(x, y, width, height);

            ctx.fillStyle = 'rgba(0, 255, 0, 0.3)';
            ctx.fillRect(x, y - 30, width, 30);

            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 20px Arial';
            ctx.fillText(
              `${prediction.class} ${Math.round(prediction.score * 100)}%`,
              x + 5,
              y - 8
            );
          });
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
  }, [model]);

  return (
    <div className="fixed inset-0 bg-black">
      <div className="relative w-full h-full">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          className="absolute inset-0 w-full h-full object-cover"
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full object-cover"
        />

        {/* カメラ切り替えボタン */}
        <button
          onClick={toggleCamera}
          className="absolute top-4 right-4 z-10 p-3 bg-black bg-opacity-50 rounded-full hover:bg-opacity-70 transition-all"
        >
          <IoCameraReverseOutline className="w-6 h-6 text-white" />
        </button>

        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50">
            <div className="text-center text-white">
              <div className="w-12 h-12 border-4 border-white border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
              <p>読み込み中...</p>
            </div>
          </div>
        )}

        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50">
            <div className="bg-red-500 text-white p-4 rounded-lg mx-4 text-center">
              <p>エラーが発生しました</p>
              <p className="text-sm mt-2">{error}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MobileObjectDetection;