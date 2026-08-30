FROM python:3.13-slim

WORKDIR /Assignment1

COPY car_price_model.pkl .
COPY model_features.pkl .

COPY app/code ./app/code

RUN pip install --no-cache-dir pandas numpy joblib scikit-learn dash plotly

EXPOSE 8050

CMD ["python", "app/code/app.py"]
