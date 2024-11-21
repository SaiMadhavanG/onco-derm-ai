import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./UploadPage.css";

function UploadPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleAnalyze = () => {
    if (selectedFile) {
      // Pass data to the predictions page (e.g., via a global state or API)
      navigate("/predictions");
    } else {
      alert("Please upload an image before analyzing.");
    }
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
          <button className="submit-button" onClick={handleAnalyze}>
            Analyze Image
          </button>
        </div>
      </main>
    </div>
  );
}

export default UploadPage;
