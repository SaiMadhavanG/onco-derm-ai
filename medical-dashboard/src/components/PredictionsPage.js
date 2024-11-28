import React from "react";
import "./PredictionsPage.css"; // Add styling here for a polished look

const DISEASE_LABELS = {
  0: "Actinic keratoses and intraepithelial carcinoma",
  1: "Basal cell carcinoma",
  2: "Benign keratosis-like lesions",
  3: "Dermatofibroma",
  4: "Melanoma",
  5: "Melanocytic nevi",
  6: "Vascular lesions",
};

function PredictionsPage({ predictions }) {
  if (!predictions) {
    return <p>No predictions available. Please upload an image for analysis.</p>;
  }

  const { predictions: predictionLabels, integrated_gradients: gradients } = predictions;

  return (
    <div className="predictions-container">
      <h1 className="predictions-title">Analysis Results</h1>
      <p className="predictions-description">
        Below are the detected conditions with corresponding explanations.
      </p>
      <div className="predictions-list">
        {predictionLabels.map((label, index) => (
          <div className="prediction-item" key={index}>
            <h2 className="disease-name">{DISEASE_LABELS[label]}</h2>
            <div className="explanation-image">
              <img
                src={`data:image/png;base64,${gradients[index]}`}
                alt={`Explanation for ${DISEASE_LABELS[label]}`}
                className="explanation-img"
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default PredictionsPage;
