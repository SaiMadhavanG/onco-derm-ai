import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./UploadPage.css";

function UploadPage({ handleAnalyze, error }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [selectedExample, setSelectedExample] = useState(null); // Track selected example image
  const navigate = useNavigate();

  const exampleImages = [
    { src: "/examples/0_[5].png", caption: "Melanocytic nevi" },
    { src: "/examples/1_[3].png", caption: "Dermatofibroma" },
    { src: "/examples/2_[4].png", caption: "Melanoma" },
    { src: "/examples/3_[0].png", caption: "Actinic keratoses" },
    { src: "/examples/horse.png", caption: "Out of Distribution" },
  ]; 

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const validTypes = ["image/jpeg", "image/png", "image/jpg"];
      if (validTypes.includes(file.type)) {
        setSelectedFile(file);
        setPreview(URL.createObjectURL(file));
        setSelectedExample(null); // Clear selected example
      } else {
        alert("Unsupported file type. Please upload a JPEG or PNG image.");
        setSelectedFile(null);
        setPreview("");
      }
    } else {
      alert("No file selected. Please upload an image.");
      setPreview("");
    }
  };

  const handleExampleSelect = (example) => {
    setSelectedExample(example.src);
    setSelectedFile(null); // Clear any uploaded file
    setPreview(example.src); // Use example image as preview
    fetch(example.src)
    .then(response => response.blob())
    .then(blob => {
      // Create a new File object from the Blob (we assume it's a jpeg file)
      const file = new File([blob], "example-image.jpg", { type: "image/jpeg" });
      setSelectedFile(file); // Update the selectedFile state with the Blob file
    })
    .catch(error => {
      console.error("Error fetching example image:", error);
      alert("Failed to load example image.");
    });
  };

  const handleSubmit = () => {
    if (selectedFile) {
      handleAnalyze(selectedFile, navigate);
    }  else {
      alert("Please upload an image or select an example.");
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="app-title">OncoDerm AI</h1>
        <p className="app-subtitle">
          Making skin cancer detection accessible with AI
        </p>
      </header>

      <main className="app-main">
        <div className="upload-section">
          <h2>Upload Patient Scan</h2>
          <p>Upload an image or choose an example to get AI-powered predictions.</p>
          <input
            type="file"
            className="upload-input"
            onChange={handleFileChange}
          />
          {error && <p className="error-message">{error}</p>}
          
          <h3>Or Select an Example:</h3>
          <div className="example-gallery">
            {exampleImages.map((example, index) => (
              <div key={index} className="example-item">
                <img
                  src={example.src}
                  alt={`Example ${index + 1}`}
                  className={`example-thumbnail ${
                    selectedExample === example.src ? "selected" : ""
                  }`}
                  onClick={() => handleExampleSelect(example)}
                />
                <p className="example-caption">{example.caption}</p>
              </div>
            ))}
          </div>

          {preview && (
            <div className="image-preview">
              <h3>Image Preview:</h3>
              <img
                src={preview}
                alt="Uploaded Preview"
                className="preview-image"
              />
            </div>
          )}

          <button className="submit-button" onClick={handleSubmit}>
            Analyze Image
          </button>
        </div>

        <div className="disclaimer-section">
          <h3>Disclaimer</h3>
          <ul className="disclaimer-list">
            <li>
              The model was trained on a low-resolution dataset (28x28). Predictions may have significant inaccuracies.
            </li>
            <li>
              View the dataset's <a href="https://github.com/SaiMadhavanG/onco-derm-ai/blob/main/docs/cards/data-card.ipynb" target="_blank" rel="noopener noreferrer">data card</a> for more details.
            </li>
            <li>
              This tool is intended for use by medical professionals for screening purposes only and should not replace a formal diagnosis.
            </li>
          </ul>
        </div>
      </main>
    </div>
  );
}

export default UploadPage;
