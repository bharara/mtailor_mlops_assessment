# MTailor Model Service

This repository contains the model service for MTailor, including model conversion, testing, and deployment utilities.

## Prerequisites

- Python 3.10
- Poetry (Python package manager)
- Docker (for containerization)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/bharara/mtailor_mlops_assessment
cd MTailor
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Download model weights:
   - Download the required model weights into the `models/` directory
   - Ensure the model weights are in the correct format for PyTorch

## Model Conversion

Convert the PyTorch model to ONNX format:
```bash
poetry run python convert_to_onnx.py
```

## Running the Service Locally

Start the FastAPI server locally:
```bash
poetry run uvicorn main:app --reload
```

The server will start at `http://localhost:8000` with the following endpoints:
- `POST /predict` - Accepts base64 encoded image data and returns the predicted class number (integer) and status code
- `GET /health` - Health check endpoint

Example `/predict` response:
```json
{
    "prediction": 0,  // Integer representing the predicted class
    "status_code": 200
}
```

API documentation is available at `http://localhost:8000/docs`

## Testing

The test suite (`test.py`) covers comprehensive testing of the model pipeline and API endpoints:

### Model and Preprocessor Tests
- Model initialization and session setup
- Preprocessor initialization and configuration
- Input validation and error handling
- Image format handling (RGB, RGBA, Grayscale, Palette)
- Output shape and type verification

### Prediction Tests
- Model prediction shape validation
- Specific class predictions (e.g., tench (class 0) and mud turtle (class 35))
- Full pipeline testing (preprocessing → prediction)
- Probability distribution validation
- Batch processing verification

### API Tests
- Health endpoint validation
- Prediction endpoint with valid image inputs
- Response format validation
- End-to-end prediction accuracy verification

### Image Processing Tests
- Various image format conversions
- Invalid input handling
- Image size normalization
- Channel conversion (RGB, RGBA, Grayscale)

Run the test suite:
```bash
poetry run pytest test.py
```

Test the deployment server:
```bash
poetry run python test_server.py
```

## Project Structure

- `model.py` - Contains the model code after conversion
- `convert_to_onnx.py` - Script to convert PyTorch model to ONNX format
- `test.py` - Pytest file for model testing
- `test_server.py` - Simple server test script
- `main.py` - FastAPI server implementation
- `Dockerfile` - Container configuration
- `cerebrium.toml` - Cerebrium deployment configuration
- `.pre-commit-config.yaml` - Pre-commit hooks for linting and code quality

## Development

The project uses pre-commit hooks for code quality. To set up:
```bash
poetry run pre-commit install
```

## Docker

Build and run the Docker container:
```bash
docker build -t mtailor-model .
docker run -p 8000:8000 mtailor-model
```

## Deployment

The project is configured for deployment using Cerebrium. The configuration is in `cerebrium.toml`.
