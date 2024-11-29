import base64
import os
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from kedro_boot.app.booter import boot_package
from kedro_boot.framework.compiler.specs import CompilationSpec
from matplotlib.figure import Figure
from PIL import Image
from pydantic import BaseModel

from onco_derm_ai.pipelines.inf_data_preprocessing.nodes import OutOfDistributionError

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

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    predictions: list[int]
    integrated_gradients: list[str]


def serialize_figure(fig: Figure) -> str:
    """
    Convert a Matplotlib figure to a Base64-encoded PNG string.
    """
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> PredictionResponse:
    try:
        image_content = await image.read()
        pil_image = Image.open(BytesIO(image_content))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid Image",
            headers={"X-Error": "InvalidImage"},
        )

    try:
        results = session.run(inputs={"inference_sample": pil_image})
    except OutOfDistributionError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Out of distribution error: {str(e)}",
            headers={"X-Error": "OutOfDistributionError"},
        )
    except Exception as e:
        # For any other unexpected errors, raise a generic 500 Internal Server Error
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: {str(e)}",
            headers={"X-Error": "InternalServerError"},
        )

    gradients_base64 = [
        serialize_figure(fig) for fig in results["integrated_gradients"]
    ]

    predictions = results["predictions"]

    response = {
        "predictions": predictions,
        "integrated_gradients": gradients_base64,
    }

    return JSONResponse(content=response)


@app.get("/")
async def docs_redirect():
    return RedirectResponse(url="/docs")
