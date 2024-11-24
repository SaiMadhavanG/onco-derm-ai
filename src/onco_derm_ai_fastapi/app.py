import base64
import os
from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from kedro_boot.app.booter import boot_package
from kedro_boot.framework.compiler.specs import CompilationSpec
from matplotlib.figure import Figure
from PIL import Image

project_dir = os.environ.get("PROJECT_DIR", ".")

session = boot_package(
    package_name="onco_derm_ai",
    compilation_specs=[
        CompilationSpec(
            inputs=["inference_sample"], outputs=["predictions", "integrated_gradients"]
        )
    ],
    kedro_args={
        "pipeline": "inference",
        "conf_source": os.path.join(project_dir, "conf"),
    },
)

app = FastAPI(title="Onco Derm AI - Skin Cancer Detection")


def serialize_figure(fig: Figure) -> str:
    """
    Convert a Matplotlib figure to a Base64-encoded PNG string.
    """
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        image_content = await image.read()
        pil_image = Image.open(BytesIO(image_content))
    except Exception:
        return JSONResponse(content={"error": "Invalid image file."}, status_code=400)

    results = session.run(inputs={"inference_sample": pil_image})

    gradients_base64 = [
        serialize_figure(fig) for fig in results["integrated_gradients"]
    ]

    predictions = results["predictions"]

    response = {
        "integrated_gradients": gradients_base64,
        "predictions": predictions,
    }

    return JSONResponse(content=response)
