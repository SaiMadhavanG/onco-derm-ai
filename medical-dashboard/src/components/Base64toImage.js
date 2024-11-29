import React, { useEffect, useState } from "react";

const Base64ToImage = () => {
    const [imageSrc, setImageSrc] = useState(null);

    useEffect(() => {
        // Path to the .txt file containing the Base64 string
        const filePath = "/base64.txt"; // File should be in the `public` folder

        // Fetch the file contents
        fetch(filePath)
            .then((response) => {
                if (!response.ok) {
                    throw new Error("Failed to fetch the file");
                }
                return response.text();
            })
            .then((base64Image) => {
                // Directly set the Base64 string as the image source
                setImageSrc(base64Image.trim());
                console.log(base64Image.trim());
            })
            .catch((error) => {
                console.error("Error:", error);
            });
    }, []);

    return (
        <div>
            <h1>Base64 Image Viewer</h1>
            {imageSrc ? (
                <img
                    src={imageSrc}
                    alt="Decoded Base64"
                    style={{ maxWidth: "100%", height: "auto" }}
                />
            ) : (
                <p>Loading image...</p>
            )}
        </div>
    );
};

export default Base64ToImage;
