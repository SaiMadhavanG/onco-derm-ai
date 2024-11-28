import React from "react";
import "./ErrorPage.css";

function ErrorPage({ error }) {
  const renderErrorMessage = () => {
    if (!error) {
      return "An unexpected error occurred. Please try again.";
    }
    // Specific handling for OOD error
    if ('response' in error && error.response.data.detail.includes("Out of distribution error")) {
      // const match = error.match(/score (\d+(\.\d+)?) which is above threshold (\d+(\.\d+)?)/);
      
        return (
          <>
            <p className="error-message">⚠️ Out of Distribution Error Detected!</p>
            <p className="error-detail">
              The uploaded image appears to be significantly different from the data the model was trained on.
            </p>
          </>
        );
      
    }
    else if (error.message === "Network Error") {
      return (
        <>
          <p className="error-message">Network Error</p>
          <p className="error-detail">
            Milan down again? :/
          </p>
        </>
      )
    }
    else if (error.status == 500) {
      return (
        <>
          <p className="error-message">Internal Server Error</p>
          <p className="error-detail">
            Our servers encountered an unexpected issue. Please try again later or contact support if the problem persists.
          </p>
        </>
      )
    }

    // Generic error message
    return "An error occurred during the analysis. Please try again.";
  };

  return (
    <div className="error-container">
      <h1 className="error-title">Oops!</h1>
      <div className="error-content">{renderErrorMessage()}</div>
      <button className="retry-button" onClick={() => (window.location.href = "/")}>
        Try Again
      </button>
    </div>
  );
}

export default ErrorPage;
