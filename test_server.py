import base64

import requests

ENDPOINT_URL = "https://api.cortex.cerebrium.ai/v4/p-b8881314/mtailor/predict"
# ENDPOINT_URL = "http://localhost:8005/predict"
API_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLWI4ODgxMzE0IiwiaWF0IjoxNzQ5MjAyODI1LCJleHAiOjIwNjQ3Nzg4MjV9.0_Ia8klO5jveXLuC5QyFtjFSUjHmOufwmI1T-KdaxWMgXpRo23XmP2Y2bIjgHhw_LtrzVT34t5QzannHnqe5QCXWLXw_txdB0bLEtTEJSul9qYrAjE_9bDcT1zOLbBuPojnAzxYlMcuIl-hbVylyIbfbfsubIfD2t7tYn6knkByh_p0_8LfCt18R8ZdNDKMDj0pNz8PPnRf5mIxmOqEDorJI0Rvm57n1QRV0Z7Ut-Sm_DEJxIc13yMkiqvwMkiRgVnogII-VvxAFU-ef9sn4xS_MezGh9I1P8qI1qQePCuvKwwKWwUx2g0TeRpLdUI_rNByL2fw2uyh_gzJkm6OGNA"
IMG_PATH = "imgs/n01440764_tench.jpeg"


def encode_image(image_path: str) -> str:
    """Convert image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


if __name__ == "__main__":
    image_data = encode_image(IMG_PATH)

    # Prepare request
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    payload = {"image_data": image_data}
    response = requests.post(ENDPOINT_URL, headers=headers, json=payload)
    try:
        result = response.json()
        print(result)
    except requests.exceptions.JSONDecodeError as e:
        print(f"Failed to decode JSON response: {e}")
        print(f"Response content: {response.text}")
    except Exception as e:
        print(f"Unexpected error: {e}")
