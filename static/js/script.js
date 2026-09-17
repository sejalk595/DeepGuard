const mediaFile = document.getElementById("mediaFile");

const fileName = document.getElementById("fileName");

const analyzeButton = document.getElementById("analyzeButton");


// ==========================================
// SHOW SELECTED FILE
// ==========================================

mediaFile.addEventListener("change", function () {

    if (mediaFile.files.length > 0) {

        const file = mediaFile.files[0];

        fileName.textContent =
            "Selected: " + file.name;

    } else {

        fileName.textContent =
            "No file selected";

    }

});


// ==========================================
// ANALYZE MEDIA
// ==========================================

analyzeButton.addEventListener(
    "click",
    async function () {

        // Check file
        if (mediaFile.files.length === 0) {

            alert(
                "Please select an image or video first."
            );

            return;
        }


        const file = mediaFile.files[0];


        // Create form data
        const formData = new FormData();

        formData.append(
            "file",
            file
        );


        // Change button
        analyzeButton.disabled = true;

        analyzeButton.textContent =
            "Analyzing...";


        try {

            // Send file to Flask
            const response = await fetch(
                "/upload",
                {
                    method: "POST",
                    body: formData
                }
            );


            // Convert response to JSON
            const data =
                await response.json();


            // Check result
            if (data.success) {

                // Go directly to result page
                window.location.href =
                    data.result_url;

            } else {

                alert(
                    "❌ " + data.message
                );

            }

        } catch (error) {

            console.error(error);

            alert(
                "❌ Something went wrong while uploading the file."
            );

        }


        // Restore button
        analyzeButton.disabled = false;

        analyzeButton.textContent =
            "🔍 Analyze Media";

    }
);