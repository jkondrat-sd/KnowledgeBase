## Project 3: SageMaker Pipeline with Managed Training & Feature Store
**Exam Domain:** Domain 1 (Data Prep), Domain 2 (Training), Domain 3 (Orchestration)

**Goal:** Build an end-to-end MLOps pipeline using SageMaker Pipelines: features land in Feature Store, a pipeline cleans the data, trains an XGBoost model, evaluates it, and registers it in the Model Registry only if it's good enough.

> **SDK version note:** SageMaker Python SDK **v3** (the current default) removed the classic `Estimator`, `Model` and `Predictor` classes in favour of `ModelTrainer`/`ModelBuilder`. The exam and most AWS material still use the v2 API, so this project pins **`sagemaker<3`**. See the [SDK v3 overview](https://sagemaker.readthedocs.io/).

### Building blocks

```
Raw data ──▶ Feature Store ──(Online: low-latency lookups)
              (Feature Group)  (Offline: S3 + Athena history)

S3 raw CSV ──▶ [Preprocess] ──▶ [Train] ──▶ [Evaluate] ──▶ <AUC ≥ threshold?> ──yes──▶ [Register model]
              ProcessingStep   TrainingStep  ProcessingStep   ConditionStep        └─no──▶ [Fail]
```

- **Feature Store / Feature Group:** a managed table of ML features, keyed by a **record identifier** (e.g. `patient_id`) and an **event time**.
  - **Online store:** latest value per record, single-digit-ms reads. Used at inference time.
  - **Offline store:** every version of every record, stored in S3 (Parquet) and catalogued in Glue, queryable with Athena. Used for training and point-in-time queries.
- **Processing job:** runs a script (scikit-learn, Spark or your own container) on managed instances to clean, split or evaluate data. Inputs and outputs are copied between S3 and `/opt/ml/processing/...`.
- **Training job:** runs an algorithm container (here the built-in XGBoost) on managed instances. Reads channels (`train`, `validation`) from S3 and writes `model.tar.gz` back to S3.
- **SageMaker Pipelines:** a DAG of steps (Processing, Training, Condition, Model/Register, Fail, ...) with **parameters**. Steps pass data via **properties** (e.g. the training step's S3 model path). Pipelines are free; you pay only for the jobs the steps run.
- **Model Registry:** versioned **model packages** inside a **model package group**, each with metrics and an **approval status** (`PendingManualApproval` / `Approved` / `Rejected`). Approval is the hand-off point to deployment (Project 4).

**Steps to Execute:**

**Prerequisites:** All steps happen in the same region (e.g. `us-east-1`). Most of the work is in a notebook, because Pipelines and Feature Store are code-first. Console/Studio UI steps are called out where they exist.

> **Cost:** under **$1** if you follow the teardown. A JupyterLab space on `ml.t3.medium` costs about $0.05/hour while running. Each processing/training job runs for a few minutes on `ml.m5.large` (about $0.12/hour).

### 1. Set up SageMaker Studio

1. **SageMaker AI console** → **Domains** → **Create domain** → **Set up for single user (Quick setup)** → **Set up**. This takes a few minutes. It creates the domain, a user profile and an execution role with `AmazonSageMakerFullAccess`.
2. When the domain is **InService**: open it → **User profiles** → your profile → **Launch** → **Studio**.
3. In Studio: **JupyterLab** → **Create JupyterLab space** → name it `mla-projects` → instance `ml.t3.medium` → **Run space** → **Open JupyterLab**.
4. Create a new notebook (**Python 3** kernel). Create it in this order so the cells below run top to bottom.

> Reuse this domain and space for Projects 4 and 5. **Stop the space** whenever you're not using it (Studio → JupyterLab → **Stop**). A running space bills even when idle.

### 2. Install the v2 SDK and set up the session

```python
%pip install -q "sagemaker>=2.200,<3"
```
Restart the kernel (**Kernel → Restart Kernel**), then:

```python
import time
import boto3
import numpy as np
import pandas as pd
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession

print("SDK version:", sagemaker.__version__)  # must start with 2.

session = sagemaker.Session()
pipeline_session = PipelineSession()
role = sagemaker.get_execution_role()
region = session.boto_region_name
bucket = session.default_bucket()  # sagemaker-<region>-<account-id>
prefix = "mla-project3"
```

### 3. Prepare the dataset

The scikit-learn breast cancer dataset is a small binary classification problem (569 rows, 30 numeric features) with nothing to download. A record ID and event time are added for Feature Store, plus some missing values so the processing step has something to clean.

```python
from sklearn.datasets import load_breast_cancer

df = load_breast_cancer(as_frame=True).frame
df.columns = [c.replace(" ", "_") for c in df.columns]
df = df.rename(columns={"target": "label"})

df.insert(0, "patient_id", pd.Series([f"P{i:04d}" for i in range(len(df))], dtype="string"))
df["event_time"] = float(round(time.time()))

# Make the data messy: knock out some values for the processing step to fix
rng = np.random.default_rng(42)
for col in ["mean_radius", "mean_texture"]:
    df.loc[rng.choice(df.index, 15, replace=False), col] = np.nan

df.head()
```

### 4. Create a Feature Group (Online + Offline store) and ingest

```python
from sagemaker.feature_store.feature_group import FeatureGroup

fg_name = f"breast-cancer-features-{time.strftime('%Y%m%d-%H%M%S')}"
fg = FeatureGroup(name=fg_name, sagemaker_session=session)
fg.load_feature_definitions(data_frame=df)  # infers String / Fractional / Integral from dtypes

fg.create(
    s3_uri=f"s3://{bucket}/{prefix}/feature-store",  # offline store location
    record_identifier_name="patient_id",
    event_time_feature_name="event_time",
    role_arn=role,
    enable_online_store=True,
)

while fg.describe()["FeatureGroupStatus"] == "Creating":
    time.sleep(10)
print(fg.describe()["FeatureGroupStatus"])

fg.ingest(data_frame=df, max_workers=3, wait=True)
```

**Online store:** read the latest record for one patient. This is what an inference service would do.
```python
fs_runtime = boto3.client("sagemaker-featurestore-runtime")
record = fs_runtime.get_record(FeatureGroupName=fg_name, RecordIdentifierValueAsString="P0001")
record["Record"][:5]
```

**Offline store:** query the history with Athena. Data takes about **15 minutes** to appear after ingestion, so carry on with step 5 and come back.
```python
query = fg.athena_query()
query.run(
    query_string=f'SELECT label, COUNT(*) AS n FROM "{query.table_name}" GROUP BY label',
    output_location=f"s3://{bucket}/{prefix}/athena-results",
)
query.wait()
query.as_dataframe()
```

In Studio, the left nav → **Data** → **Feature Store** shows the feature group, its schema and both stores.

### 5. Upload the raw data and write the step scripts

```python
df.to_csv("raw.csv", index=False)
raw_uri = session.upload_data("raw.csv", bucket=bucket, key_prefix=f"{prefix}/raw")
raw_uri
```

**Processing script:** fill missing values, drop non-feature columns, and split 70/15/15. XGBoost's CSV format wants the **label first** and **no header**.
```python
%%writefile preprocess.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

base = "/opt/ml/processing"
df = pd.read_csv(f"{base}/input/raw.csv")

df = df.drop(columns=["patient_id", "event_time"])
df = df.fillna(df.median(numeric_only=True))
df = df[["label"] + [c for c in df.columns if c != "label"]]

train, rest = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
validation, test = train_test_split(rest, test_size=0.5, random_state=42, stratify=rest["label"])

for name, split in [("train", train), ("validation", validation), ("test", test)]:
    os.makedirs(f"{base}/{name}", exist_ok=True)
    split.to_csv(f"{base}/{name}/{name}.csv", header=False, index=False)
```

**Evaluation script:** score the trained model on the test split and write `evaluation.json`. The condition step reads this file.
```python
%%writefile evaluate.py
import json
import os
import pickle
import tarfile

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

with tarfile.open("/opt/ml/processing/model/model.tar.gz") as tar:
    tar.extractall(".")

try:
    model = pickle.load(open("xgboost-model", "rb"))
except Exception:
    model = xgb.Booster()
    model.load_model("xgboost-model")

test = pd.read_csv("/opt/ml/processing/test/test.csv", header=None)
y_true = test.iloc[:, 0]
preds = model.predict(xgb.DMatrix(test.iloc[:, 1:].values))

report = {
    "binary_classification_metrics": {
        "auc": {"value": float(roc_auc_score(y_true, preds))},
        "accuracy": {"value": float(accuracy_score(y_true, preds > 0.5))},
    }
}
os.makedirs("/opt/ml/processing/evaluation", exist_ok=True)
with open("/opt/ml/processing/evaluation/evaluation.json", "w") as f:
    json.dump(report, f)
print(report)
```

### 6. Define the pipeline

```python
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

# Pipeline parameters: can be overridden per execution
input_data = ParameterString("InputData", default_value=raw_uri)
instance_type = ParameterString("InstanceType", default_value="ml.m5.large")
auc_threshold = ParameterFloat("AucThreshold", default_value=0.95)

model_package_group = "breast-cancer-models"
xgb_image = sagemaker.image_uris.retrieve("xgboost", region, version="1.7-1")

# 1. ProcessingStep: clean + split
sklearn_processor = SKLearnProcessor(
    framework_version="1.2-1", role=role, instance_type=instance_type,
    instance_count=1, sagemaker_session=pipeline_session,
)
step_process = ProcessingStep(
    name="Preprocess",
    step_args=sklearn_processor.run(
        code="preprocess.py",
        inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(output_name=name, source=f"/opt/ml/processing/{name}")
            for name in ["train", "validation", "test"]
        ],
    ),
)

# 2. TrainingStep: built-in XGBoost container
xgb_estimator = Estimator(
    image_uri=xgb_image, role=role, instance_type=instance_type, instance_count=1,
    output_path=f"s3://{bucket}/{prefix}/models", sagemaker_session=pipeline_session,
)
xgb_estimator.set_hyperparameters(
    objective="binary:logistic", eval_metric="auc", num_round=100, max_depth=4, eta=0.2,
)
outputs = step_process.properties.ProcessingOutputConfig.Outputs
step_train = TrainingStep(
    name="Train",
    step_args=xgb_estimator.fit({
        "train": TrainingInput(outputs["train"].S3Output.S3Uri, content_type="text/csv"),
        "validation": TrainingInput(outputs["validation"].S3Output.S3Uri, content_type="text/csv"),
    }),
)

# 3. ProcessingStep: evaluate on the test split (XGBoost image has xgboost + sklearn)
eval_processor = ScriptProcessor(
    image_uri=xgb_image, command=["python3"], role=role, instance_type=instance_type,
    instance_count=1, sagemaker_session=pipeline_session,
)
evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
step_eval = ProcessingStep(
    name="Evaluate",
    step_args=eval_processor.run(
        code="evaluate.py",
        inputs=[
            ProcessingInput(source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                            destination="/opt/ml/processing/model"),
            ProcessingInput(source=outputs["test"].S3Output.S3Uri, destination="/opt/ml/processing/test"),
        ],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
    ),
    property_files=[evaluation_report],
)

# 4. Register the model (only reached if the condition passes)
model = Model(
    image_uri=xgb_image, model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    role=role, sagemaker_session=pipeline_session,
)
model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=Join(on="/", values=[
            step_eval.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
            "evaluation.json",
        ]),
        content_type="application/json",
    )
)
step_register = ModelStep(
    name="RegisterModel",
    step_args=model.register(
        content_types=["text/csv"], response_types=["text/csv"],
        inference_instances=["ml.m5.large"], transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    ),
)

step_fail = FailStep(name="AucTooLow", error_message=Join(on=" ", values=["AUC below threshold", auc_threshold]))

# 5. ConditionStep: register only if AUC >= threshold
step_condition = ConditionStep(
    name="CheckAuc",
    conditions=[ConditionGreaterThanOrEqualTo(
        left=JsonGet(step_name=step_eval.name, property_file=evaluation_report,
                     json_path="binary_classification_metrics.auc.value"),
        right=auc_threshold,
    )],
    if_steps=[step_register],
    else_steps=[step_fail],
)

pipeline = Pipeline(
    name="breast-cancer-pipeline",
    parameters=[input_data, instance_type, auc_threshold],
    steps=[step_process, step_train, step_eval, step_condition],
    sagemaker_session=pipeline_session,
)
```

> `ModelStep` + `model.register()` is the current way to register a model in a pipeline. The older `RegisterModel` step collection does the same thing, and the exam may use either name.

### 7. Run the pipeline

```python
pipeline.upsert(role_arn=role)  # create or update the pipeline definition
execution = pipeline.start()
execution.wait(delay=30, max_attempts=120)  # about 10-15 minutes
execution.list_steps()
```

While it runs, open Studio → left nav → **Pipelines** → `breast-cancer-pipeline` → the execution. You'll see the **DAG**, each step's status, logs, inputs/outputs and the parameters used.

**Try the failure branch:** rerun with an impossible threshold and watch the condition route to `AucTooLow`:
```python
execution_fail = pipeline.start(parameters={"AucThreshold": 0.9999})
```

### 8. Review and approve the model in the Model Registry

1. Studio → left nav → **Models** → **Registered models** → `breast-cancer-models`.
2. Open version 1: see the **evaluation metrics** (AUC, accuracy), the training job lineage and the status **PendingManualApproval**.
3. **Update status → Approved**. In a real MLOps setup, an EventBridge rule on this approval triggers a deployment pipeline. You'll deploy it by hand in Project 4.

**Exam Concepts Tested:** Feature Store (Online low-latency vs. Offline S3 historical), SageMaker Pipelines (steps, parameters, properties, condition/fail steps), SageMaker Processing Jobs, built-in algorithms, Model Registry and approval workflow.

### Troubleshooting
- **`ImportError` / `Estimator` not found:** the kernel is still on SDK v3. Check `sagemaker.__version__`, rerun the `%pip install` cell and **restart the kernel**.
- **Athena query returns 0 rows:** the offline store lags by about 15 minutes. Wait and rerun.
- **AccessDenied on Athena or Glue:** attach `AmazonAthenaFullAccess` to the execution role (IAM → Roles → `AmazonSageMaker-ExecutionRole-...`), or skip the offline query.
- **`ResourceLimitExceeded` for `ml.m5.large`:** new accounts sometimes have a quota of 0 for processing/training instances. Request an increase in **Service Quotas → Amazon SageMaker** (e.g. *ml.m5.large for processing job usage*), or run with `pipeline.start(parameters={"InstanceType": "ml.m5.xlarge"})`.
- **A step failed:** in the Pipelines UI, click the step → **Logs**, or CloudWatch → Log groups → `/aws/sagemaker/ProcessingJobs` and `/aws/sagemaker/TrainingJobs`.

### Teardown
Projects 4 and 5 can reuse the Studio domain. Delete this project's resources now, then stop the space.

1. **Feature Group** (from the notebook):
   ```python
   fg.delete()
   ```
   This removes the online store. Offline data in S3 stays, so it's deleted in step 4. Also check **Glue → Data Catalog → Tables** (database `sagemaker_featurestore`) and delete the feature group's table if it's still there.
2. **Pipeline:** `pipeline.delete()` (the definition is free, but tidy).
3. **Model Registry:** Studio → **Models** → `breast-cancer-models` → delete each version, then the group. It's free, but keep it if you want to deploy it in Project 4.
4. **S3:** S3 console → `sagemaker-<region>-<account-id>` → delete the `mla-project3/` prefix (raw data, splits, models, offline store, Athena results). Also delete any `breast-cancer-pipeline/` or auto-named job folders the SDK created there.
5. **CloudWatch logs:** delete `/aws/sagemaker/ProcessingJobs` and `/aws/sagemaker/TrainingJobs` log streams for this project if you're not continuing (storage is billed and never expires by default).
6. **Stop the JupyterLab space:** Studio → **JupyterLab** → **Stop**.

#### Deleting the Studio domain (once you're done with Projects 3–5)
The domain itself is free, but its **EFS volume** and any running apps are not.
1. SageMaker AI console → **Domains** → your domain → **Space management**: stop and **delete** every space.
2. **User profiles** → each profile → delete its apps, then **Delete user profile**.
3. **Domain settings** → **Delete domain**.
4. **EFS console** → delete the file system tagged with the domain ID. It is **retained** after the domain is deleted, and it bills for storage.
5. **IAM → Roles:** delete `AmazonSageMaker-ExecutionRole-...` if nothing else uses it.

**Verify zero cost:** a day later, check **Billing → Bills** (or **Cost Explorer** by *Service*): SageMaker, S3 and CloudWatch should stop accruing. An **AWS Budgets** alert ($1) catches anything missed.

### References
- [Feature Store](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html) · [Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html) · [Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)
- [Built-in XGBoost](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html) · [Processing jobs](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
