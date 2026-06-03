import { useEffect, useRef, useState } from "react";
import axios from "axios";

function CameraFeed() {

  const videoRef = useRef(null);

  const canvasRef = useRef(null);

  const [detections, setDetections] = useState([]);

  const [loading, setLoading] = useState(false);

  const [ocrText, setOcrText] = useState("");

const [guidance, setGuidance] = useState("");

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

          const ocrFormData = new FormData();

ocrFormData.append(
  "file",
  blob,
  "frame.jpg"
);

const ocrResponse =
  await axios.post(
    "http://127.0.0.1:8000/ocr",
    ocrFormData
  );

setOcrText(
  ocrResponse.data.text
);

const navigationFormData =
  new FormData();

navigationFormData.append(
  "file",
  blob,
  "frame.jpg"
);

const navigationResponse =
  await axios.post(
    "http://127.0.0.1:8000/navigation",
    navigationFormData
  );

setGuidance(
  navigationResponse.data.guidance
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
<div
  style={{
    marginTop: "20px"
  }}
>

  <h3>OCR Text</h3>

  <p>
    {
      ocrText
        ? ocrText
        : "No text detected."
    }
  </p>

</div>

<div
  style={{
    marginTop: "20px"
  }}
>

  <h3>Navigation Guidance</h3>

  <p>
    {
      guidance
        ? guidance
        : "No guidance available."
    }
  </p>

</div>
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