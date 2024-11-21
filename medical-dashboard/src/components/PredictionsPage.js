import React from "react";
import { useNavigate } from "react-router-dom";
import "./PredictionsPage.css";

function PredictionsPage() {
  const navigate = useNavigate();

  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="app-title">Prediction Results</h1>
      </header>

      <main className="app-main">
        <div className="result-section">
          <h2>Analysis Complete</h2>
          <p>Your results are:</p>
          <div className="result-card">
            <p><strong>Diagnosis:</strong> Possible Malignancy Detected</p>
            <p><strong>Confidence:</strong> 87%</p>
          </div>
          <button className="back-button" onClick={() => navigate("/")}>
            Back to Upload
          </button>
        </div>
      </main>
    </div>
  );
}

export default PredictionsPage;
