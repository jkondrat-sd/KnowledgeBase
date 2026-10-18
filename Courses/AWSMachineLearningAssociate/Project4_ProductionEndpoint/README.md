## Project 4: Production Endpoint Deployment Strategy
**Exam Domain:** Domain 3 (Deployment & Orchestration)

**Goal:** Deploy a model to a SageMaker real-time endpoint, run an A/B test with two production variants, shift traffic between them, and compare real-time with the other inference options (Serverless, Asynchronous, Batch Transform) hands-on.

> **SDK version note:** like Project 3, this uses the SageMaker Python SDK **v2** API (`Model.deploy()`, `Estimator`), pinned with `sagemaker<3`. SDK v3 replaced these classes with `ModelBuilder`/`ModelTrainer`.

### Building blocks

```
Model ──▶ Endpoint Config ──▶ Endpoint ──▶ https://runtime.sagemaker.../endpoints/<name>/invocations
(artifact   (variants: model,      (the running
 + image)    instance type/count,   instances)
             traffic weight)

Endpoint "ab-endpoint"
 ├── VariantA: model A, 1 × ml.m5.large, weight 0.8 ──▶ ~80% of requests
 └── VariantB: model B, 1 × ml.m5.large, weight 0.2 ──▶ ~20% of requests
```

- **Model:** a SageMaker resource that pairs a model artifact (`model.tar.gz` in S3) with an inference container image and an IAM role. It doesn't run anything by itself.
- **Endpoint configuration:** the deployment blueprint. It lists one or more **production variants**, each with a model, instance type, instance count and **traffic weight**. Endpoint configs are immutable, so changing a deployment means creating a new config.
- **Endpoint:** the live HTTPS endpoint built from a config. **`UpdateEndpoint`** swaps it to a new config without downtime. **`UpdateEndpointWeightsAndCapacities`** changes the weights or counts of the existing variants in place.
- **Production variant:** one model + instance fleet behind the endpoint. Traffic splits by weight (`weight / sum of weights`). A caller can also force a variant with `TargetVariant`.
- **Deployment guardrails (blue/green):** an `UpdateEndpoint` with a **deployment config** moves traffic from the old fleet (blue) to the new one (green) **all at once**, as a **canary** (small slice first) or **linearly** (in steps). It **rolls back automatically** if a CloudWatch alarm fires.

**Steps to Execute:**

**Prerequisites:** the Studio domain and JupyterLab space from [Project 3](../Project3_SageMakerPipeline/README.md#1-set-up-sagemaker-studio). Start the space, create a new notebook, and run everything in one region.

> **Cost:** real-time endpoints bill **per instance-hour while they exist**, even with zero traffic. This project runs 2 × `ml.m5.large` (about $0.12/hour each) for about an hour. **Do the teardown the same day.** The total should be about $1.

### 1. Set up and train two model versions

Two quick XGBoost training jobs with different hyperparameters stand in for "current model" (A) and "challenger" (B). Each takes about 3–4 minutes, and they run in parallel.

```python
%pip install -q "sagemaker>=2.200,<3"
```
Restart the kernel, then:

```python
import time
import boto3
import numpy as np
import pandas as pd
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.serializers import CSVSerializer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

session = sagemaker.Session()
role = sagemaker.get_execution_role()
region = session.boto_region_name
bucket = session.default_bucket()
prefix = "mla-project4"
sm = boto3.client("sagemaker")
runtime = boto3.client("sagemaker-runtime")
ts = time.strftime("%Y%m%d-%H%M%S")

# Data: label first, no header (XGBoost CSV format)
df = load_breast_cancer(as_frame=True).frame
df = df[["target"] + [c for c in df.columns if c != "target"]]
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])
train.to_csv("train.csv", header=False, index=False)
test.iloc[:, 1:].to_csv("test_features.csv", header=False, index=False)  # features only, for batch
train_uri = session.upload_data("train.csv", bucket=bucket, key_prefix=f"{prefix}/train")
test_features_uri = session.upload_data("test_features.csv", bucket=bucket, key_prefix=f"{prefix}/batch-input")

xgb_image = sagemaker.image_uris.retrieve("xgboost", region, version="1.7-1")

def make_estimator(max_depth, num_round):
    est = Estimator(image_uri=xgb_image, role=role, instance_type="ml.m5.large", instance_count=1,
                    output_path=f"s3://{bucket}/{prefix}/models", sagemaker_session=session)
    est.set_hyperparameters(objective="binary:logistic", max_depth=max_depth, num_round=num_round, eta=0.2)
    return est

est_a = make_estimator(max_depth=4, num_round=100)
est_b = make_estimator(max_depth=2, num_round=30)
est_a.fit({"train": TrainingInput(train_uri, content_type="text/csv")}, wait=False)
est_b.fit({"train": TrainingInput(train_uri, content_type="text/csv")}, wait=False)
est_a.latest_training_job.wait(logs="None")
est_b.latest_training_job.wait(logs="None")
print(est_a.model_data, est_b.model_data, sep="\n")
```

> **Using Project 3's model instead:** if you kept the approved model in the registry, you can use it as Model A. Studio → **Models** → `breast-cancer-models` → version → **Deploy**, or copy its model artifact S3 URI into `Model(model_data=...)`.

### 2. Deploy Model A to a real-time endpoint with `Model.deploy()`

```python
endpoint_name = f"ab-endpoint-{ts}"

model_a = Model(image_uri=xgb_image, model_data=est_a.model_data, role=role,
                name=f"xgb-model-a-{ts}", sagemaker_session=session)
predictor = model_a.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name,
    serializer=CSVSerializer(),
)  # creates Model + EndpointConfig + Endpoint, about 5-8 minutes

sample = test.iloc[0, 1:].values
print("Prediction:", predictor.predict(sample))  # probability of class 1 (benign)
```

`deploy()` created three resources. See them in the **SageMaker AI console → Inference**: **Models**, **Endpoint configurations** and **Endpoints**.

### 3. Add Model B as a second production variant (80/20 A/B test)

Endpoint configs are immutable, so create a **new config** with both variants and **update** the live endpoint to it. The endpoint keeps serving traffic during the update.

```python
model_b_name = f"xgb-model-b-{ts}"
sm.create_model(
    ModelName=model_b_name,
    ExecutionRoleArn=role,
    PrimaryContainer={"Image": xgb_image, "ModelDataUrl": est_b.model_data},
)

ab_config_name = f"{endpoint_name}-ab-config"
sm.create_endpoint_config(
    EndpointConfigName=ab_config_name,
    ProductionVariants=[
        {"VariantName": "VariantA", "ModelName": model_a.name, "InstanceType": "ml.m5.large",
         "InitialInstanceCount": 1, "InitialVariantWeight": 0.8},
        {"VariantName": "VariantB", "ModelName": model_b_name, "InstanceType": "ml.m5.large",
         "InitialInstanceCount": 1, "InitialVariantWeight": 0.2},
    ],
)
sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=ab_config_name)
sm.get_waiter("endpoint_in_service").wait(EndpointName=endpoint_name)  # about 5-10 minutes
print([(v["VariantName"], v["CurrentWeight"]) for v in sm.describe_endpoint(EndpointName=endpoint_name)["ProductionVariants"]])
```

**Console alternative:** SageMaker AI console → **Inference → Endpoint configurations → Create** (add both variants with their weights), then **Endpoints** → your endpoint → **Change** → choose the new config.

### 4. Send traffic and watch the split

```python
from collections import Counter

def invoke(row, target_variant=None):
    kwargs = {"EndpointName": endpoint_name, "ContentType": "text/csv",
              "Body": ",".join(map(str, row))}
    if target_variant:
        kwargs["TargetVariant"] = target_variant
    resp = runtime.invoke_endpoint(**kwargs)
    return resp["InvokedProductionVariant"], float(resp["Body"].read())

counts = Counter(invoke(row)[0] for row in test.iloc[:, 1:].values[:100] for _ in range(2))
print(counts)  # roughly 160 VariantA / 40 VariantB

# Force a specific variant (e.g. for QA of the challenger)
print(invoke(sample, target_variant="VariantB"))
```

Per-variant metrics: **CloudWatch → Metrics → AWS/SageMaker → EndpointName, VariantName** → `Invocations`, `ModelLatency`, `Invocation5XXErrors`. Comparing these per variant (plus business metrics) is how you'd judge an A/B test.

### 5. Shift traffic: promote Model B

Once B wins, move traffic gradually **without redeploying**, using weights only:

```python
def set_weights(a, b):
    sm.update_endpoint_weights_and_capacities(
        EndpointName=endpoint_name,
        DesiredWeightsAndCapacities=[
            {"VariantName": "VariantA", "DesiredWeight": a},
            {"VariantName": "VariantB", "DesiredWeight": b},
        ],
    )
    sm.get_waiter("endpoint_in_service").wait(EndpointName=endpoint_name)

set_weights(0.5, 0.5)
print(Counter(invoke(row)[0] for row in test.iloc[:, 1:].values[:100]))
set_weights(0.0, 1.0)  # all traffic to B; A still running, so instant rollback = set_weights(1, 0)
print(Counter(invoke(row)[0] for row in test.iloc[:, 1:].values[:50]))
```

**Blue/green with deployment guardrails** (read and understand it; running it is optional): instead of manual weights, `update_endpoint` can take a `DeploymentConfig` that shifts traffic to a new fleet automatically and rolls back on a CloudWatch alarm:
```python
# sm.update_endpoint(
#     EndpointName=endpoint_name,
#     EndpointConfigName="<new-config>",
#     DeploymentConfig={
#         "BlueGreenUpdatePolicy": {
#             "TrafficRoutingConfiguration": {
#                 "Type": "CANARY",  # or ALL_AT_ONCE / LINEAR
#                 "CanarySize": {"Type": "CAPACITY_PERCENT", "Value": 10},
#                 "WaitIntervalInSeconds": 300,
#             },
#             "TerminationWaitInSeconds": 120,
#         },
#         "AutoRollbackConfiguration": {"Alarms": [{"AlarmName": "<5xx-alarm>"}]},
#     },
# )
```

> **A/B variants vs. blue/green:** production variants run **both models side by side** for as long as you like and are used to *compare* models. Blue/green **replaces** the old fleet with the new one, using canary or linear steps, and is used to *safely roll out* a model you've already chosen. Shadow variants (shadow testing) are a third option: the new model receives a copy of live traffic, but its responses are never returned to callers.

### 6. Serverless Inference (hands-on)

Same model B, no instances to manage. It scales to zero and you pay per request and compute time.
```python
from sagemaker.serverless import ServerlessInferenceConfig

model_b_serverless = Model(image_uri=xgb_image, model_data=est_b.model_data, role=role,
                           name=f"xgb-model-b-sls-{ts}", sagemaker_session=session)
serverless_predictor = model_b_serverless.deploy(
    serverless_inference_config=ServerlessInferenceConfig(memory_size_in_mb=2048, max_concurrency=2),
    endpoint_name=f"serverless-endpoint-{ts}",
    serializer=CSVSerializer(),
)

for i in range(3):
    start = time.time()
    serverless_predictor.predict(sample)
    print(f"call {i}: {time.time() - start:.2f}s")  # first call = cold start
```

### 7. Batch Transform (hands-on)

No endpoint: SageMaker spins up instances, scores a whole S3 dataset, writes the results to S3, and shuts down.
```python
transformer = est_a.transformer(
    instance_count=1, instance_type="ml.m5.large",
    output_path=f"s3://{bucket}/{prefix}/batch-output",
)
transformer.transform(test_features_uri, content_type="text/csv", split_type="Line")
transformer.wait()  # about 5 minutes

out = session.read_s3_file(bucket, f"{prefix}/batch-output/test_features.csv.out")
print(out.splitlines()[:5])
```

### 8. Compare the inference options

| | **Real-time** | **Serverless** | **Asynchronous** | **Batch Transform** |
|---|---|---|---|---|
| **Use when** | Steady traffic, low latency | Intermittent/unpredictable traffic, tolerates cold starts | Large payloads or long processing, near-real-time OK | Offline scoring of a whole dataset |
| **Payload / time limits** | 6 MB, 60 s | 4 MB, 60 s | Up to **1 GB**, up to **1 hour** | Unlimited dataset size (per-record max set by `MaxPayloadInMB`, up to 100 MB) |
| **Scale to zero?** | No (min 1 instance) | Yes, automatically | Yes (autoscaling min 0) | Nothing persistent |
| **Billing** | Per instance-hour while running | Per ms of compute + data processed | Per instance-hour while running | Per instance-hour for the job duration |
| **GPU** | Yes | No | Yes | Yes |
| **Request flow** | Synchronous response | Synchronous response | Request queued, input/output in S3, SNS notification | Reads S3, writes S3 |

> Your root README says async processing is up to 15 minutes. The current limit is **up to 1 hour** per request (`InvocationTimeoutSeconds`).

**Exam Concepts Tested:** Model / endpoint config / endpoint, production variants and traffic weights, `UpdateEndpoint` vs. `UpdateEndpointWeightsAndCapacities`, blue/green deployment guardrails (all-at-once, canary, linear, auto-rollback), shadow testing, Real-Time vs. Serverless vs. Async vs. Batch Transform.

### Troubleshooting
- **`ResourceLimitExceeded`:** new accounts often have a quota of 0 or 1 for `ml.m5.large` **endpoint usage**. The A/B step needs 2. Request an increase in **Service Quotas → Amazon SageMaker** → *ml.m5.large for endpoint usage*, or use `ml.t2.medium` for both variants.
- **Endpoint stuck `Creating` then `Failed`:** endpoint → **Failure reason**, plus CloudWatch → `/aws/sagemaker/Endpoints/<endpoint-name>`.
- **`ModelError` on invoke:** the body must be CSV with **30 feature values and no label**.
- **`ImportError` / `Estimator` not found:** SDK v3 is still loaded. Rerun the `%pip install` cell and restart the kernel.

### Teardown
Endpoints bill every hour they exist, so do this as soon as you're done. Run it in the notebook:

```python
for name in [endpoint_name, f"serverless-endpoint-{ts}"]:
    try:
        sm.delete_endpoint(EndpointName=name)
    except sm.exceptions.ClientError as e:
        print(name, e)

# Endpoint configs and models are free, but tidy them up
for cfg in sm.list_endpoint_configs(NameContains=ts)["EndpointConfigs"]:
    sm.delete_endpoint_config(EndpointConfigName=cfg["EndpointConfigName"])
for m in sm.list_models(NameContains=ts)["Models"]:
    sm.delete_model(ModelName=m["ModelName"])
```

Then:
1. **Verify:** SageMaker AI console → **Inference → Endpoints** is **empty** (or holds nothing from this project). This is the step that matters for cost.
2. **S3:** delete the `mla-project4/` prefix in `sagemaker-<region>-<account-id>`.
3. **CloudWatch logs:** delete `/aws/sagemaker/Endpoints/ab-endpoint-...`, `/aws/sagemaker/Endpoints/serverless-endpoint-...`, and this project's streams in `/aws/sagemaker/TrainingJobs` and `/aws/sagemaker/TransformJobs`.
4. **Stop the JupyterLab space** (Studio → JupyterLab → **Stop**). If you're not doing Project 5, delete the Studio domain: see [Project 3: Deleting the Studio domain](../Project3_SageMakerPipeline/README.md#deleting-the-studio-domain-once-youre-done-with-projects-35).

**Verify zero cost:** next day, **Billing → Bills** / **Cost Explorer** by service: *SageMaker* should show no new **Host** (endpoint) usage.

### References
- [Real-time inference](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html) · [Production variants / A/B testing](https://docs.aws.amazon.com/sagemaker/latest/dg/model-ab-testing.html) · [Deployment guardrails (blue/green)](https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails.html)
- [Serverless inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html) · [Asynchronous inference](https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html) · [Batch transform](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html)
