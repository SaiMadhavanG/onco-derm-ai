import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./UploadPage.css";

function UploadPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState(""); // State to track errors
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const validTypes = ["image/jpeg", "image/png", "image/jpg"];
      if (validTypes.includes(file.type)) {
        setSelectedFile(file); // Valid file
        setError(""); // Clear any previous errors
      } else {
        setError("Unsupported file type. Please upload a JPEG or PNG image."); // Error for unsupported file types
        setSelectedFile(null);
        navigate("/error"); // Redirect to error page
      }
    } else {
      setError("No file selected. Please upload an image."); // Error for no file selected
      navigate("/error"); // Redirect to error page
    }
  };

  const handleAnalyze = () => {
    if (!selectedFile) {
      setError("Please upload an image before analyzing."); // Error for no file uploaded
      return;
    }

    // Simulating API call
    const simulateAnalysis = new Promise((resolve, reject) => {
      setTimeout(() => {
        const isError = Math.random() < 0.2; // Simulate a 20% chance of failure
        isError ? reject("Error during analysis.") : resolve("Analysis successful.");
      }, 1000);
    });

    simulateAnalysis
      .then(() => {
        navigate("/predictions"); // Redirect to predictions page on success
      })
      .catch((errorMessage) => {
        setError(errorMessage); // Set error message
        navigate("/error"); // Redirect to error page
      });
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
          <button className="submit-button" onClick={handleAnalyze}>
            Analyze Image
          </button>
        </div>
      </main>
    </div>
  );
}

export default UploadPage;
