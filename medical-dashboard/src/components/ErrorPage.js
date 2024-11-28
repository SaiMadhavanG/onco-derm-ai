import React from "react";

function ErrorPage({ error }) {
  return (
    <div>
      <h1>Error</h1>
      <p>{error || "An unknown error occurred."}</p>
    </div>
  );
}

export default ErrorPage;
