import {
  useEffect,
  useRef,
} from "react";

import type {
  CameraViewMode,
} from "../App";

import type {
  Detection,
  Face,
} from "../services/api";

import {
  detectObjects,
  recognizeFaces,
} from "../services/api";


type Props = {
  cameraRunning:
    boolean;

  setCameraRunning:
    React.Dispatch<
      React.SetStateAction<boolean>
    >;

  cameraStatus:
    string;

  setCameraStatus:
    React.Dispatch<
      React.SetStateAction<string>
    >;

  detections:
    Detection[];

  setDetections:
    React.Dispatch<
      React.SetStateAction<Detection[]>
    >;

  faces:
    Face[];

  setFaces:
    React.Dispatch<
      React.SetStateAction<Face[]>
    >;

  detectionFPS:
    number;

  setDetectionFPS:
    React.Dispatch<
      React.SetStateAction<number>
    >;

  cameraViewMode:
    CameraViewMode;

  setCameraViewMode:
    React.Dispatch<
      React.SetStateAction<CameraViewMode>
    >;

  onEnroll:
    () => void;
};


const DETECTION_INTERVAL =
  300;


const DETECTION_CONFIDENCE =
  0.40;


export default function CameraPanel({
  cameraRunning,
  setCameraRunning,
  cameraStatus,
  setCameraStatus,
  detections,
  setDetections,
  faces,
  setFaces,
  detectionFPS,
  setDetectionFPS,
  cameraViewMode,
  setCameraViewMode,
  onEnroll,
}: Props) {

  const videoRef =
    useRef<HTMLVideoElement | null>(
      null
    );


  const canvasRef =
    useRef<HTMLCanvasElement | null>(
      null
    );


  const streamRef =
    useRef<MediaStream | null>(
      null
    );


  const animationFrameRef =
    useRef<number | null>(
      null
    );


  const detectionRunningRef =
    useRef(false);


  const lastDetectionRef =
    useRef(0);


  const fpsCountRef =
    useRef(0);


  const fpsStartRef =
    useRef(performance.now());


  /* =======================================================
     START CAMERA
  ======================================================= */

  async function startCamera() {

    try {

      const stream =
        await navigator
          .mediaDevices
          .getUserMedia({

            video: {

              width: {
                ideal: 1280,
              },

              height: {
                ideal: 720,
              },

              facingMode:
                "user",

            },

            audio:
              false,

          });


      streamRef.current =
        stream;


      if (
        videoRef.current
      ) {

        videoRef.current.srcObject =
          stream;

        await videoRef.current.play();

      }


      setCameraRunning(true);

      setCameraStatus(
        "🟢 Camera running"
      );


      setDetections([]);

      setFaces([]);

      setDetectionFPS(0);


      fpsCountRef.current =
        0;

      fpsStartRef.current =
        performance.now();

      lastDetectionRef.current =
        0;


      animationFrameRef.current =
        requestAnimationFrame(
          detectionLoop
        );

    } catch (error) {

      console.error(
        "Camera error:",
        error
      );


      setCameraStatus(
        "❌ Camera access failed"
      );


      alert(
        "Could not access the camera. Please allow camera permission."
      );

    }

  }


  /* =======================================================
     STOP CAMERA
  ======================================================= */

  function stopCamera() {

    streamRef.current
      ?.getTracks()
      .forEach(
        (track) =>
          track.stop()
      );


    streamRef.current =
      null;


    if (
      videoRef.current
    ) {

      videoRef.current.srcObject =
        null;

    }


    if (
      animationFrameRef.current !==
      null
    ) {

      cancelAnimationFrame(
        animationFrameRef.current
      );

      animationFrameRef.current =
        null;

    }


    detectionRunningRef.current =
      false;


    setCameraRunning(
      false
    );

    setCameraStatus(
      "Camera is off"
    );

    setDetections([]);

    setFaces([]);

    setDetectionFPS(0);


    const canvas =
      canvasRef.current;


    if (
      canvas
    ) {

      canvas
        .getContext("2d")
        ?.clearRect(
          0,
          0,
          canvas.width,
          canvas.height
        );

    }

  }


  /* =======================================================
     DETECTION LOOP
  ======================================================= */

  function detectionLoop(
    timestamp:
      number
  ) {

    if (
      !streamRef.current
    ) {

      return;

    }


    if (
      timestamp -
        lastDetectionRef.current >=
      DETECTION_INTERVAL
    ) {

      lastDetectionRef.current =
        timestamp;


      if (
        !detectionRunningRef.current
      ) {

        void detectFrame();

      }

    }


    animationFrameRef.current =
      requestAnimationFrame(
        detectionLoop
      );

  }


  /* =======================================================
     DETECT FRAME
  ======================================================= */

  async function detectFrame() {

    const video =
      videoRef.current;


    if (
      !video ||
      video.readyState <
        2 ||
      detectionRunningRef.current
    ) {

      return;

    }


    detectionRunningRef.current =
      true;


    try {

      const captureCanvas =
        document.createElement(
          "canvas"
        );


      captureCanvas.width =
        video.videoWidth;


      captureCanvas.height =
        video.videoHeight;


      const context =
        captureCanvas.getContext(
          "2d"
        );


      if (
        !context
      ) {

        return;

      }


      context.drawImage(
        video,
        0,
        0,
        captureCanvas.width,
        captureCanvas.height
      );


      const blob =
        await new Promise<
          Blob | null
        >(
          (resolve) =>
            captureCanvas.toBlob(
              resolve,
              "image/jpeg",
              0.65
            )
        );


      if (
        !blob
      ) {

        return;

      }


      const [
        yoloData,
        faceData,
      ] = await Promise.all([

        detectObjects(
          blob,
          DETECTION_CONFIDENCE
        ),

        recognizeFaces(
          blob
        ),

      ]);


      const newDetections =
        yoloData.detections ||
        [];


      const newFaces =
        faceData.faces ||
        [];


      setDetections(
        newDetections
      );


      setFaces(
        newFaces
      );


      fpsCountRef.current++;


      const now =
        performance.now();


      if (
        now -
          fpsStartRef.current >=
        1000
      ) {

        setDetectionFPS(
          Math.round(
            (
              fpsCountRef.current *
              1000
            ) /
            (
              now -
              fpsStartRef.current
            )
          )
        );


        fpsCountRef.current =
          0;


        fpsStartRef.current =
          now;

      }


      const parts =
        [] as string[];


      if (
        newDetections.length
      ) {

        parts.push(
          `${newDetections.length} object${
            newDetections.length ===
            1
              ? ""
              : "s"
          }`
        );

      }


      if (
        newFaces.length
      ) {

        parts.push(
          `${newFaces.length} face${
            newFaces.length ===
            1
              ? ""
              : "s"
          }`
        );

      }


      setCameraStatus(
        parts.length
          ? `🟢 ${parts.join(
              " · "
            )}`
          : "🟢 Monitoring scene"
      );


      drawDetections(
        newDetections,
        newFaces,
        yoloData.width ||
          video.videoWidth,
        yoloData.height ||
          video.videoHeight
      );

    } catch (error) {

      console.error(
        "Vision error:",
        error
      );


      setCameraStatus(
        "⚠️ Vision temporarily unavailable"
      );

    } finally {

      detectionRunningRef.current =
        false;

    }

  }


  /* =======================================================
     DRAW DETECTIONS
  ======================================================= */

  function drawDetections(
    objectDetections:
      Detection[],

    detectedFaces:
      Face[],

    sourceWidth:
      number,

    sourceHeight:
      number
  ) {

    const canvas =
      canvasRef.current;


    if (
      !canvas
    ) {

      return;

    }


    canvas.width =
      sourceWidth;


    canvas.height =
      sourceHeight;


    const context =
      canvas.getContext(
        "2d"
      );


    if (
      !context
    ) {

      return;

    }


    context.clearRect(
      0,
      0,
      canvas.width,
      canvas.height
    );


    const scaleX =
      canvas.width /
      sourceWidth;


    const scaleY =
      canvas.height /
      sourceHeight;


    context.font =
      "600 15px Arial";


    for (
      const object of
      objectDetections
    ) {

      const box =
        object.box;


      if (
        !box
      ) {

        continue;

      }


      const width =
        (
          box.x2 -
          box.x1
        ) *
        scaleX;


      const height =
        (
          box.y2 -
          box.y1
        ) *
        scaleY;


      const x =
        canvas.width -
        box.x2 *
          scaleX;


      const y =
        box.y1 *
        scaleY;


      context.strokeStyle =
        "#72ffb6";


      context.lineWidth =
        3;


      context.strokeRect(
        x,
        y,
        width,
        height
      );


      const label =
        `${object.class} ${Math.round(
          Number(
            object.confidence ||
            0
          )
        )}%`;


      const labelWidth =
        context.measureText(
          label
        ).width +
        16;


      context.fillStyle =
        "#72ffb6";


      context.fillRect(
        x,
        Math.max(
          0,
          y - 27
        ),
        labelWidth,
        25
      );


      context.fillStyle =
        "#06120d";


      context.fillText(
        label,
        x + 8,
        Math.max(
          17,
          y - 9
        )
      );

    }


    for (
      const face of
      detectedFaces
    ) {

      const box =
        face.box;


      if (
        !box
      ) {

        continue;

      }


      const width =
        (
          box.x2 -
          box.x1
        ) *
        scaleX;


      const height =
        (
          box.y2 -
          box.y1
        ) *
        scaleY;


      const x =
        canvas.width -
        box.x2 *
          scaleX;


      const y =
        box.y1 *
        scaleY;


      const known =
        face.name &&
        face.name !==
          "Unknown";


      context.strokeStyle =
        known
          ? "#68c7ff"
          : "#ff6b7d";


      context.lineWidth =
        3;


      context.strokeRect(
        x,
        y,
        width,
        height
      );


      const label =
        known
          ? `${face.name} ${Math.round(
              Number(
                face.confidence ||
                0
              )
            )}%`
          : "Unknown";


      const labelWidth =
        context.measureText(
          label
        ).width +
        16;


      context.fillStyle =
        known
          ? "#68c7ff"
          : "#ff6b7d";


      context.fillRect(
        x,
        Math.max(
          0,
          y - 27
        ),
        labelWidth,
        25
      );


      context.fillStyle =
        "#061018";


      context.fillText(
        label,
        x + 8,
        Math.max(
          17,
          y - 9
        )
      );

    }

  }


  /* =======================================================
     VIEW CONTROLS
  ======================================================= */

  function decreaseView() {

    setCameraViewMode(
      (current) =>
        current === "large"
          ? "normal"
          : "compact"
    );

  }


  function increaseView() {

    setCameraViewMode(
      (current) =>
        current === "compact"
          ? "normal"
          : "large"
    );

  }


  /* =======================================================
     CLEANUP
  ======================================================= */

  useEffect(() => {

    return () => {

      streamRef.current
        ?.getTracks()
        .forEach(
          (track) =>
            track.stop()
        );


      if (
        animationFrameRef.current !==
        null
      ) {

        cancelAnimationFrame(
          animationFrameRef.current
        );

      }

    };

  }, []);


  return (

    <aside className="camera-panel">

      <div className="camera-heading">

        <div className="camera-heading-left">

          <div className="camera-heading-icon">
            ◎
          </div>

          <div>

            <div className="section-title">
              Vision
            </div>

            <div className="section-caption">
              YOLO · Face Recognition
            </div>

          </div>

        </div>


        <div className="camera-size-controls">

          <button
            className="camera-size-button"
            type="button"
            onClick={
              decreaseView
            }
            disabled={
              cameraViewMode ===
              "compact"
            }
            title="Smaller camera"
          >
            −
          </button>


          <span>
            {
              cameraViewMode
            }
          </span>


          <button
            className="camera-size-button"
            type="button"
            onClick={
              increaseView
            }
            disabled={
              cameraViewMode ===
              "large"
            }
            title="Larger camera"
          >
            +
          </button>

        </div>

      </div>


      <div className="camera-body">

        <div className="camera-frame">

          <video
            ref={
              videoRef
            }
            className="camera-video"
            autoPlay
            muted
            playsInline
          />


          <canvas
            ref={
              canvasRef
            }
            className="camera-overlay"
          />


          {!cameraRunning && (

            <div className="camera-idle">

              <div className="camera-idle-orb">
                ◎
              </div>

              <strong>
                Vision offline
              </strong>

              <span>
                Start your camera to activate
                object and face recognition.
              </span>

            </div>

          )}


          {cameraRunning && (

            <div className="camera-live-badge">
              <span />
              LIVE
            </div>

          )}

        </div>


        <div className="camera-stat-grid">

          <div className="camera-stat">

            <span>
              OBJECTS
            </span>

            <strong>
              {
                detections.length
              }
            </strong>

          </div>


          <div className="camera-stat">

            <span>
              FACES
            </span>

            <strong>
              {
                faces.length
              }
            </strong>

          </div>


          <div className="camera-stat">

            <span>
              FPS
            </span>

            <strong>
              {
                detectionFPS
              }
            </strong>

          </div>

        </div>


        <div
          className={
            `camera-status-chip ${
              cameraRunning
                ? "active"
                : ""
            }`
          }
        >

          <span>
            ●
          </span>

          {
            cameraStatus
          }

        </div>


        <div className="camera-actions">

          <button
            className={
              `camera-main-button ${
                cameraRunning
                  ? "stop"
                  : ""
              }`
            }
            type="button"
            onClick={() => {

              if (
                cameraRunning
              ) {

                stopCamera();

              } else {

                void startCamera();

              }

            }}
          >

            {
              cameraRunning
                ? "⏹ Stop Camera"
                : "◎ Start Camera"
            }

          </button>


          <button
            className="camera-enroll-button"
            type="button"
            onClick={
              onEnroll
            }
            disabled={
              !cameraRunning
            }
          >
            👤 Enroll
          </button>

        </div>

      </div>

    </aside>
  );
}