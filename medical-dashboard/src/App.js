import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import UploadPage from "./components/UploadPage";
import PredictionsPage from "./components/PredictionsPage";
import ErrorPage from "./components/ErrorPage"; // Import the ErrorPage component
import Base64ToImage from "./components/Base64toImage"; // Import the Base64ToImage component

function App() {
  return (
    <Router>
      <Routes>
        {/* Home page with the image upload */}
        <Route path="/" element={<UploadPage />} />
        {/* Predictions page */}
        <Route path="/predictions" element={<PredictionsPage />} />
        {/* Fallback error page for undefined routes */}
        <Route path="/error" element={<ErrorPage />} />
        <Route path="/image" element={<Base64ToImage />} />
      </Routes>

    </Router>
  );
}

export default App;
