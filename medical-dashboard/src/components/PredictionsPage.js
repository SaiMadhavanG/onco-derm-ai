import React, { useState } from "react";
import "./PredictionsPage.css";

function PredictionsPage({ predictions }) {
  const [tooltip, setTooltip] = useState({ visible: false, text: "", x: 0, y: 0 });


  if (!predictions) {
    return <p>No predictions available. Please upload an image to analyze.</p>;
  }

  const { predictions: predictionSet, integrated_gradients: igImages } = predictions;

  // Disease details with brief descriptions and Wikipedia links
  const diseaseInfo = [
    {
      label: "Actinic keratoses and intraepithelial carcinoma",
      description: "A rough, scaly patch on your skin that develops from years of sun exposure.",
      wikiLink: "https://en.wikipedia.org/wiki/Actinic_keratosis",
    },
    {
      label: "Basal cell carcinoma",
      description: "A common type of skin cancer that arises in the basal cells of the skin.",
      wikiLink: "https://en.wikipedia.org/wiki/Basal-cell_carcinoma",
    },
    {
      label: "Benign keratosis-like lesions",
      description: "Non-cancerous skin growths resembling keratosis.",
      wikiLink: "https://en.wikipedia.org/wiki/Seborrheic_keratosis",
    },
    {
      label: "Dermatofibroma",
      description: "A benign skin nodule often found on the lower legs.",
      wikiLink: "https://en.wikipedia.org/wiki/Dermatofibroma",
    },
    {
      label: "Melanoma",
      description: "A serious form of skin cancer that begins in melanocytes.",
      wikiLink: "https://en.wikipedia.org/wiki/Melanoma",
    },
    {
      label: "Melanocytic nevi",
      description: "Commonly referred to as moles, usually harmless growths on the skin.",
      wikiLink: "https://en.wikipedia.org/wiki/Melanocytic_nevus",
    },
    {
      label: "Vascular lesions",
      description: "Abnormal blood vessels or lymph vessels, often appearing on the skin.",
      wikiLink: "https://en.wikipedia.org/wiki/Vascular_anomaly",
    },
  ];


  const handleMouseEnter = (e, text) => {
    const rect = e.target.getBoundingClientRect();
    setTooltip({ visible: true, text, x: rect.left + rect.width / 2, y: rect.top - 10 });
  };

  const handleMouseLeave = () => {
    setTooltip({ visible: false, text: "", x: 0, y: 0 });
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="app-title">Prediction Results</h1>
      </header>

      <main className="app-main">
        <div className="predictions-section">
          <h2>Conformal Predictions</h2>
          {predictionSet.map((prediction, index) => {
            const disease = diseaseInfo[prediction];
            return (
              <div key={index} className="prediction-item">
                <h3
                  className="prediction-label"
                  onMouseEnter={(e) => handleMouseEnter(e, disease.description)}
                  onMouseLeave={handleMouseLeave}
                  onClick={() => window.open(disease.wikiLink, "_blank")}
                >
                  {disease.label}
                </h3>
                <img
                  src={`data:image/png;base64,${igImages[index]}`}
                  alt={`Integrated Gradients Heatmap for ${disease.label}`}
                  className="ig-image"
                />
              </div>
            );
          })}
        </div>

        {tooltip.visible && (
          <div
            className="tooltip"
            style={{ top: `${tooltip.y}px`, left: `${tooltip.x}px` }}
          >
            {tooltip.text}
          </div>
        )}

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
