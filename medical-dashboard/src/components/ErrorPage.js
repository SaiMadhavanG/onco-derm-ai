import React from "react";
import { useNavigate } from "react-router-dom";
import "./ErrorPage.css";

function ErrorPage() {
  const navigate = useNavigate();

  const handleGoBack = () => {
    navigate("/"); // Redirect to the upload page
  };

  return (
    <div className="error-page-container">
      <h1 className="error-page-title">Oops! Something Went Wrong</h1>
      <p className="error-page-message">
        The file you uploaded is not valid. Please try again with a correct file.
      </p>
      <button className="error-page-button" onClick={handleGoBack}>
        Go Back to Upload Page
      </button>
    </div>
  );
}

export default ErrorPage;
