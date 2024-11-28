import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import UploadPage from "./components/UploadPage";
import PredictionsPage from "./components/PredictionsPage";
import ErrorPage from "./components/ErrorPage";
import Base64ToImage from "./components/Base64toImage";
import axios from "axios";

function App() {
  const [predictions, setPredictions] = useState(null); // State to store predictions
  const [error, setError] = useState(""); // State to handle errors

  const handleAnalyze = async (selectedFile, navigate) => {
    if (!selectedFile) {
      setError("Please upload an image before analyzing.");
      navigate("/error");
      return;
    }

    try {
      const formData = new FormData();
      formData.append("image", selectedFile);

      const response = await axios.post("http://localhost:8000/predict", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setPredictions(response.data); // Store predictions from the server
      setError(""); // Clear any errors
      navigate("/predictions"); // Navigate to predictions page on success
    } catch (error) {
      setError(error);
      navigate("/error");
    }
  };

  return (
    <Router>
      <Routes>
        {/* Home page with the image upload */}
        <Route
          path="/"
          element={<UploadPage handleAnalyze={handleAnalyze} error={error} />}
        />
        {/* Predictions page */}
        <Route
          path="/predictions"
          element={<PredictionsPage predictions={predictions} />}
        />
        {/* Fallback error page for undefined routes */}
        <Route path="/error" element={<ErrorPage error={error} />} />
        <Route path="/image" element={<Base64ToImage />} />
      </Routes>
    </Router>
  );
}

export default App;
