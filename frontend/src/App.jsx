import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {

  const [file, setFile] = useState(null);

  const [imagePreview, setImagePreview] = useState(null);

  const [detections, setDetections] = useState([]);

  const [ocrText, setOcrText] = useState("");

  const [guidance, setGuidance] = useState("");

  const handleFileChange = (event) => {

    const selectedFile = event.target.files[0];

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

  const formData = new FormData();
  formData.append("file", file);

  try {

    const detectResponse =
      await axios.post(
        "http://127.0.0.1:8000/detect",
        formData
      );

    console.log(detectResponse.data);

    setDetections(
      detectResponse.data.detections
    );

  } catch (error) {

    console.error(error);

    alert("Backend connection failed");
  }
};

  return (

    <div className="container">

      <h1>Smart Vision Assistant</h1>

      <p className="subtitle">
        AI-Powered Navigation Assistant for Visually Impaired Users
      </p>

      <div className="upload-section">

        <input
          type="file"
          onChange={handleFileChange}
        />

        <button onClick={analyzeImage}>
          Analyze
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

      <div className="results-container">

        <div className="card">

          <h2>Detected Objects</h2>

          {detections.length > 0 ? (

            <ul>

              {detections.map(
                (item, index) => (

                  <li key={index}>

                    {item.class_name}
                    {" "}
                    ({item.score})

                  </li>
                )
              )}

            </ul>

          ) : (

            <p>No detections yet.</p>
          )}

        </div>

        <div className="card">

          <h2>OCR Text</h2>

          <p>
            {ocrText || "No text extracted yet."}
          </p>

        </div>

        <div className="card">

          <h2>Navigation Guidance</h2>

          <p>
            {guidance || "No guidance available."}
          </p>

        </div>

      </div>

    </div>
  );
}

export default App;