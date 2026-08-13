FROM python:3.11.15-slim-trixie

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install --no-install-recommends -y \
      curl \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY paddle_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN python -c \
    "import numpy as np; from paddleocr import PaddleOCR; \
     PaddleOCR(lang='ch', use_textline_orientation=False, \
               use_doc_orientation_classify=False, use_doc_unwarping=False, \
               enable_mkldnn=False).predict(np.full((320, 320, 3), 255, dtype='uint8'))"

COPY paddle_service/src /app/src

WORKDIR /app/src

EXPOSE 8005

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
