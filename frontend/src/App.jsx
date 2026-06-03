import { useState } from "react";
import axios from "axios";
import "./App.css";
import CameraFeed from "./components/CameraFeed";

function App() {

  const [file, setFile] = useState(null);

  const [imagePreview, setImagePreview] = useState(null);

  const [detections, setDetections] = useState([]);

  const [ocrText, setOcrText] = useState("");

  const [guidance, setGuidance] = useState("");

  const [loading, setLoading] = useState(false);

  const [showCamera, setShowCamera] =
  useState(false);

  const handleFileChange = (event) => {

    const selectedFile = event.target.files[0];

    if (!selectedFile) return;

    setFile(selectedFile);

    setImagePreview(
      URL.createObjectURL(selectedFile)
    );
  };

  const analyzeImage = async () => {

    if (!file) {

      alert("Please select an image");

      return;
    }

    setLoading(true);

    try {

      // DETECT OBJECTS
      const detectFormData = new FormData();

      detectFormData.append("file", file);

      const detectResponse =
        await axios.post(
          "http://127.0.0.1:8000/detect",
          detectFormData
        );

      setDetections(
        detectResponse.data.detections
      );

      // OCR
      const ocrFormData = new FormData();

      ocrFormData.append("file", file);

      const ocrResponse =
        await axios.post(
          "http://127.0.0.1:8000/ocr",
          ocrFormData
        );

      setOcrText(
        ocrResponse.data.text
      );

      // NAVIGATION
      const navigationFormData =
        new FormData();

      navigationFormData.append(
        "file",
        file
      );

      const navigationResponse =
        await axios.post(
          "http://127.0.0.1:8000/navigation",
          navigationFormData
        );

      setGuidance(
        navigationResponse.data.guidance
      );

      setLoading(false);

    } catch (error) {

      console.error(error);

      setLoading(false);

      alert("Backend connection failed");
    }
  };

  return (

    <div className="container">

      <h1>Smart Vision Assistant</h1>

      <p className="subtitle">
        AI-Powered Navigation Assistant for
        Visually Impaired Users
      </p>

      <div className="upload-section">

        <input
          type="file"
          onChange={handleFileChange}
        />

        <button
          onClick={analyzeImage}
          disabled={loading}
        >
          {
            loading
              ? "Analyzing..."
              : "Analyze"
          }
        </button>
<button
  onClick={() =>
    setShowCamera(!showCamera)
  }
>
  {
    showCamera
      ? "Stop Camera"
      : "Start Camera"
  }
</button>

      </div>

      {imagePreview && (

        <div className="preview-container">

          <img
            src={imagePreview}
            alt="Preview"
            className="preview-image"
          />

        </div>

      )}
      {
  showCamera &&
  <CameraFeed />
}

      <div className="results-container">

        <div className="card">

          <h2>
            Detected Objects
            {
              detections.length > 0 &&
              ` (${detections.length})`
            }
          </h2>

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

        <div className="card">

          <h2>OCR Text</h2>

          <p>
            {
              ocrText
                ? ocrText
                : "No text detected."
            }
          </p>

        </div>

        <div className="card">

          <h2>
            Navigation Guidance
          </h2>

          <p>
            {
              guidance
                ? guidance
                : "No guidance available."
            }
          </p>

        </div>

      </div>

    </div>
  );
}

export default App;