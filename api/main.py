"""
main.py
-------
FastAPI backend for the Thyroid Cancer Classification Pipeline.

Endpoints
---------
GET  /              → health check
GET  /health        → detailed system status
GET  /model-info    → model metadata and metrics
POST /predict       → predict single image (benign / malignant)
POST /upload-data   → upload zip of new images for retraining
POST /retrain       → trigger model retraining on uploaded data
GET  /retrain-status → check if retraining is in progress
GET  /metrics       → model performance metrics
"""

import os
import io
import uuid
import shutil
import asyncio
import zipfile
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Local modules ─────────────────────────────────────────────────────────────
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from prediction import (
    predict_from_bytes, get_model_info, reload_model
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
UPLOAD_DIR      = BASE_DIR / 'data' / 'uploads'
RETRAIN_DATA_DIR= BASE_DIR / 'data' / 'retrain'
MODELS_DIR      = BASE_DIR / 'models'

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RETRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title='Thyroid Cancer Classifier API',
    description='Binary classification of thyroid ultrasound images: Benign vs Malignant',
    version='1.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],   # tighten in production
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# ── State ─────────────────────────────────────────────────────────────────────
_state = {
    'retraining'      : False,
    'retrain_started' : None,
    'retrain_finished': None,
    'retrain_error'   : None,
    'prediction_count': 0,
    'startup_time'    : datetime.utcnow().isoformat(),
    'uploaded_batches': 0
}

# ── Pydantic schemas ──────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    label           : str
    confidence      : float
    prob_benign     : float
    prob_malignant  : float
    threshold_used  : float
    inference_time_ms: float
    timestamp       : str

class HealthResponse(BaseModel):
    status          : str
    uptime_since    : str
    model_loaded    : bool
    prediction_count: int
    retraining      : bool
    version         : str

class RetrainStatusResponse(BaseModel):
    retraining  : bool
    started_at  : Optional[str]
    finished_at : Optional[str]
    error       : Optional[str]

class ModelInfoResponse(BaseModel):
    model_version    : str
    class_names      : list
    img_size         : list
    optimal_threshold: float
    metrics          : dict
    model_loaded     : bool


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event('startup')
async def startup_event():
    """Pre-load model on startup so first request is fast."""
    logger.info('Starting Thyroid Classifier API...')
    try:
        info = get_model_info()
        logger.info(f"Model ready — version: {info['model_version']}, "
                    f"AUC: {info['metrics'].get('auc', 'N/A')}")
    except Exception as e:
        logger.warning(f'Model not loaded on startup: {e}')


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get('/', tags=['Health'])
def root():
    """Root endpoint — basic health ping."""
    return {
        'message': 'Thyroid Cancer Classifier API is running',
        'docs'   : '/docs',
        'status' : 'ok'
    }


@app.get('/health', response_model=HealthResponse, tags=['Health'])
def health():
    """
    Detailed health check.
    Used by the UI dashboard to display model uptime and status.
    """
    try:
        info         = get_model_info()
        model_loaded = info['model_loaded']
    except Exception:
        model_loaded = False

    return HealthResponse(
        status          ='ok' if model_loaded else 'degraded',
        uptime_since    =_state['startup_time'],
        model_loaded    =model_loaded,
        prediction_count=_state['prediction_count'],
        retraining      =_state['retraining'],
        version         ='1.0.0'
    )


@app.get('/model-info', response_model=ModelInfoResponse, tags=['Model'])
def model_info():
    """
    Return model metadata: version, class names, metrics, threshold.
    Used by the UI dashboard visualisations section.
    """
    try:
        info = get_model_info()
        return ModelInfoResponse(
            model_version    =info['model_version'],
            class_names      =info['class_names'],
            img_size         =list(info['img_size']),
            optimal_threshold=info['optimal_threshold'],
            metrics          =info['metrics'],
            model_loaded     =info['model_loaded']
        )
    except Exception as e:
        logger.error(f'model-info error: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/metrics', tags=['Model'])
def metrics():
    """
    Return model evaluation metrics.
    Used to populate the metrics cards in the UI dashboard.
    """
    try:
        info = get_model_info()
        return {
            'metrics'  : info['metrics'],
            'threshold': info['optimal_threshold'],
            'version'  : info['model_version']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/predict', response_model=PredictionResponse, tags=['Prediction'])
async def predict(
    file     : UploadFile = File(..., description='Thyroid ultrasound image (.jpg / .png)'),
    threshold: Optional[float] = Query(None, description='Override decision threshold (0.0–1.0)')
):
    """
    Predict whether a thyroid ultrasound image is Benign or Malignant.

    - Accepts: .jpg, .jpeg, .png
    - Returns: label, confidence, both class probabilities, inference time
    - Uses optimal threshold from model metadata by default
    """
    # Validate file type
    allowed = {'image/jpeg', 'image/jpg', 'image/png'}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f'Invalid file type: {file.content_type}. Upload a .jpg or .png image.'
        )

    # Validate threshold range
    if threshold is not None and not (0.0 < threshold < 1.0):
        raise HTTPException(
            status_code=422,
            detail='Threshold must be between 0.0 and 1.0'
        )

    try:
        image_bytes = await file.read()
        result      = predict_from_bytes(image_bytes, threshold=threshold)

        _state['prediction_count'] += 1
        logger.info(
            f"Prediction #{_state['prediction_count']}: "
            f"{result['label']} ({result['confidence']}%) "
            f"in {result['inference_time_ms']}ms"
        )

        return PredictionResponse(
            **result,
            timestamp=datetime.utcnow().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Prediction error: {e}')
        raise HTTPException(status_code=500, detail=f'Prediction failed: {str(e)}')


@app.post('/upload-data', tags=['Retraining'])
async def upload_data(
    file: UploadFile = File(..., description='ZIP file containing benign/ and malignant/ folders')
):
    """
    Upload a ZIP file of new labelled images for retraining.

    Expected ZIP structure:
        upload.zip
        ├── benign/
        │   ├── img1.jpg
        │   └── img2.jpg
        └── malignant/
            ├── img3.jpg
            └── img4.jpg

    Returns counts of uploaded images per class.
    """
    if not file.filename.endswith('.zip'):
        raise HTTPException(
            status_code=422,
            detail='Please upload a .zip file containing benign/ and malignant/ folders.'
        )

    # Save uploaded zip
    batch_id  = str(uuid.uuid4())[:8]
    zip_path  = UPLOAD_DIR / f'upload_{batch_id}.zip'
    dest_dir  = RETRAIN_DATA_DIR / batch_id

    try:
        contents = await file.read()
        with open(zip_path, 'wb') as f:
            f.write(contents)

        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest_dir)

        # Validate structure
        benign_dir    = dest_dir / 'benign'
        malignant_dir = dest_dir / 'malignant'

        if not benign_dir.exists() or not malignant_dir.exists():
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise HTTPException(
                status_code=422,
                detail='ZIP must contain benign/ and malignant/ subfolders.'
            )

        benign_count    = len(list(benign_dir.glob('*.jpg')) + list(benign_dir.glob('*.png')))
        malignant_count = len(list(malignant_dir.glob('*.jpg')) + list(malignant_dir.glob('*.png')))

        if benign_count == 0 and malignant_count == 0:
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise HTTPException(
                status_code=422,
                detail='No valid images found in the ZIP file.'
            )

        _state['uploaded_batches'] += 1
        zip_path.unlink()  # clean up zip

        logger.info(
            f'Upload batch {batch_id}: '
            f'{benign_count} benign, {malignant_count} malignant'
        )

        return {
            'status'         : 'success',
            'batch_id'       : batch_id,
            'benign_count'   : benign_count,
            'malignant_count': malignant_count,
            'total'          : benign_count + malignant_count,
            'message'        : f'Upload successful. Call POST /retrain with batch_id={batch_id} to retrain.'
        }

    except HTTPException:
        raise
    except zipfile.BadZipFile:
        raise HTTPException(status_code=422, detail='Invalid ZIP file.')
    except Exception as e:
        logger.error(f'Upload error: {e}')
        raise HTTPException(status_code=500, detail=f'Upload failed: {str(e)}')


def _run_retrain(batch_id: Optional[str]):
    """Background retraining task."""
    import sys
    sys.path.append(str(BASE_DIR / 'src'))
    from model import retrain

    _state['retraining']       = True
    _state['retrain_started']  = datetime.utcnow().isoformat()
    _state['retrain_error']    = None
    _state['retrain_finished'] = None

    try:
        # Determine data directory
        if batch_id:
            data_dir = str(RETRAIN_DATA_DIR / batch_id)
        else:
            # Use most recently uploaded batch
            batches = sorted(RETRAIN_DATA_DIR.iterdir())
            if not batches:
                raise ValueError('No uploaded data found. Upload data first via POST /upload-data')
            data_dir = str(batches[-1])

        logger.info(f'Retraining started on data: {data_dir}')
        retrain(new_data_dir=data_dir, epochs=10, lr=1e-5)

        # Reload model in memory
        reload_model()

        _state['retrain_finished'] = datetime.utcnow().isoformat()
        logger.info('Retraining completed successfully.')

    except Exception as e:
        _state['retrain_error'] = str(e)
        logger.error(f'Retraining failed: {e}')
    finally:
        _state['retraining'] = False


@app.post('/retrain', tags=['Retraining'])
async def trigger_retrain(
    background_tasks: BackgroundTasks,
    batch_id: Optional[str] = Query(None, description='Specific batch_id from /upload-data response')
):
    """
    Trigger model retraining on uploaded data.

    - Runs in the background so the API stays responsive
    - Use GET /retrain-status to monitor progress
    - Model is hot-swapped after retraining (no restart needed)

    Pass batch_id from the /upload-data response to retrain on a specific upload.
    If omitted, uses the most recently uploaded batch.
    """
    if _state['retraining']:
        raise HTTPException(
            status_code=409,
            detail='Retraining is already in progress. Check GET /retrain-status'
        )

    background_tasks.add_task(_run_retrain, batch_id)

    return {
        'status' : 'started',
        'message': 'Retraining started in background. Monitor via GET /retrain-status',
        'batch_id': batch_id
    }


@app.get('/retrain-status', response_model=RetrainStatusResponse, tags=['Retraining'])
def retrain_status():
    """
    Check the status of the retraining process.
    Poll this endpoint from the UI to show a progress indicator.
    """
    return RetrainStatusResponse(
        retraining  =_state['retraining'],
        started_at  =_state['retrain_started'],
        finished_at =_state['retrain_finished'],
        error       =_state['retrain_error']
    )


@app.get('/stats', tags=['Monitoring'])
def stats():
    """
    Return API usage statistics.
    Used by the UI uptime / monitoring panel.
    """
    return {
        'prediction_count': _state['prediction_count'],
        'uploaded_batches': _state['uploaded_batches'],
        'uptime_since'    : _state['startup_time'],
        'retraining'      : _state['retraining'],
        'last_retrain'    : _state['retrain_finished']
    }