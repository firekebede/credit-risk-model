from mlflow.tracking import MlflowClient

client = MlflowClient()

model_name = "credit_risk_classifier"
version = 1  # replace with your actual version

# Assign alias 'prod' instead of using stage 'Production'
client.set_registered_model_alias(
    name=model_name,
    alias="prod",
    version=version
)

print(f"✅ Assigned alias 'prod' to {model_name} version {version}")
