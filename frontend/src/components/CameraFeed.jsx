import { useEffect, useRef, useState } from "react";
import axios from "axios";

function CameraFeed() {

  const videoRef = useRef(null);

  const canvasRef = useRef(null);

  const [detections, setDetections] = useState([]);

  const [loading, setLoading] = useState(false);

  useEffect(() => {

    const startCamera = async () => {

      try {

        const stream =
          await navigator.mediaDevices.getUserMedia({
            video: true
          });

        if (videoRef.current) {

          videoRef.current.srcObject =
            stream;
        }

      } catch (error) {

        console.error(
          "Camera access denied:",
          error
        );
      }
    };

    startCamera();

  }, []);

  const analyzeFrame = async () => {

    if (!videoRef.current) return;

    setLoading(true);

    const video = videoRef.current;

    const canvas = canvasRef.current;

    canvas.width = video.videoWidth;

    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");

    context.drawImage(
      video,
      0,
      0,
      canvas.width,
      canvas.height
    );

    canvas.toBlob(
      async (blob) => {

        const formData = new FormData();

        formData.append(
          "file",
          blob,
          "frame.jpg"
        );

        try {

          const response =
            await axios.post(
              "http://127.0.0.1:8000/detect",
              formData
            );

          setDetections(
            response.data.detections
          );

        } catch (error) {

          console.error(error);

          alert(
            "Frame analysis failed"
          );
        }

        setLoading(false);

      },
      "image/jpeg"
    );
  };

  return (

    <div className="camera-container">

      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="camera-feed"
      />

      <canvas
        ref={canvasRef}
        style={{ display: "none" }}
      />

      <br />

      <button
        onClick={analyzeFrame}
        disabled={loading}
      >
        {
          loading
            ? "Analyzing..."
            : "Analyze Current Frame"
        }
      </button>

      <div
        style={{
          marginTop: "20px"
        }}
      >

        <h3>
          Camera Detections
        </h3>

        {
          detections.length > 0
            ? (
              <ul>

                {
                  detections.map(
                    (
                      item,
                      index
                    ) => (

                      <li key={index}>

                        <strong>
                          {
                            item.class_name
                          }
                        </strong>

                        <br />

                        Direction:
                        {" "}
                        {
                          item.direction
                        }

                        <br />

                        Distance:
                        {" "}
                        {
                          item.distance
                        }

                        <br />

                        Confidence:
                        {" "}
                        {
                          (
                            item.score * 100
                          ).toFixed(0)
                        }
                        %

                      </li>
                    )
                  )
                }

              </ul>
            )
            : (
              <p>
                No detections yet.
              </p>
            )
        }

      </div>

    </div>

  );
}

export default CameraFeed;