import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import UploadPage from "./components/UploadPage";
import PredictionsPage from "./components/PredictionsPage";

function App() {
  return (
    <Router>
      <Routes>
        {/* Home page with the image upload */}
        <Route path="/" element={<UploadPage />} />
        {/* Predictions page */}
        <Route path="/predictions" element={<PredictionsPage />} />
      </Routes>
    </Router>
  );
}

export default App;
