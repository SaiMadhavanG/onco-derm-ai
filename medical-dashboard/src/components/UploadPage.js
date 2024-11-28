import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./UploadPage.css";

function UploadPage({ handleAnalyze, error }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState("");
  const navigate = useNavigate(); // Hook for navigation

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const validTypes = ["image/jpeg", "image/png", "image/jpg"];
      if (validTypes.includes(file.type)) {
        setSelectedFile(file); // Valid file
        setPreview(URL.createObjectURL(file)); // Generate preview URL
      } else {
        alert("Unsupported file type. Please upload a JPEG or PNG image.");
        setSelectedFile(null);
        setPreview(""); // Clear preview on invalid file
      }
    } else {
      alert("No file selected. Please upload an image.");
      setPreview(""); // Clear preview when no file is selected
    }
  };

  const handleSubmit = () => {
    handleAnalyze(selectedFile, navigate); // Pass the file and navigate function
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="app-title">AI-Powered Cancer Detection</h1>
        <p className="app-subtitle">
          Harness the power of Artificial Intelligence to support medical diagnoses.
        </p>
      </header>

      <main className="app-main">
        <div className="upload-section">
          <h2>Upload Patient Scan</h2>
          <p>Upload an image to get AI-powered predictions.</p>
          <input type="file" className="upload-input" onChange={handleFileChange} />
          {error && <p className="error-message">{error}</p>} {/* Display error messages */}
          
          {/* Image Preview Section */}
          {preview && (
            <div className="image-preview">
              <h3>Image Preview:</h3>
              <img src={preview} alt="Uploaded Preview" className="preview-image" />
            </div>
          )}

          <button className="submit-button" onClick={handleSubmit}>
            Analyze Image
          </button>
        </div>
      </main>
    </div>
  );
}

export default UploadPage;
