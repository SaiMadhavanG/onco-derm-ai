import React from "react";
import "./PredictionsPage.css";

function PredictionsPage({ predictions }) {
  if (!predictions) {
    return <p>No predictions available. Please upload an image to analyze.</p>;
  }

  const { predictions: predictionSet, integrated_gradients: igImages } = predictions;

  const labels = [
    "Actinic keratoses and intraepithelial carcinoma",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
  ];

  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="app-title">Prediction Results</h1>
      </header>

      <main className="app-main">
        <div className="predictions-section">
          <h2>Conformal Predictions</h2>
          {predictionSet.map((prediction, index) => (
            <div key={index} className="prediction-item">
              <h3>{labels[prediction]}</h3>
              <img
                src={`data:image/png;base64,${igImages[index]}`}
                alt={`Integrated Gradients Heatmap for ${labels[prediction]}`}
                className="ig-image"
              />
            </div>
          ))}
        </div>

        <div className="disclaimer-section">
          <h3>Disclaimer</h3>
          <p>
            The predictions above are <strong>conformal predictions</strong>, providing a <strong>90% certainty</strong> that the target class
            is included in the prediction set. Conformal predictions offer an additional layer of reliability for high-stakes scenarios like medical screening.
          </p>
          <p>
            The accompanying images use <strong>Integrated Gradients (IG)</strong>, a method for explaining AI predictions. The IG heatmap highlights areas in the image that most influenced the model's predictions. These visualizations help ensure transparency and enable medical professionals to interpret results effectively.
          </p>
          <p>
            <strong>Note:</strong> This tool is intended for screening purposes only and should not replace a formal diagnosis by a medical professional.
          </p>
        </div>
      </main>
    </div>
  );
}

export default PredictionsPage;
