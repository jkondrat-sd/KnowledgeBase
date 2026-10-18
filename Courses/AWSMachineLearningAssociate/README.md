# AWS Machine Learning Engineer Associate (MLA-C02)

AWS signin url - https://982534384718.signin.aws.amazon.com/console

Gemmini insturciton for quizzes: Can you generate me a 20 question questionare to test my knowledge of domain 1?
Generates an actual interactive quiz, so cool!

This part of the knowledge base repo is dedicated to completing practice projects with Amazon Bedrock, Sagemaker AI and Glue giving a practical and deep understanding to the theory necessary to pass the exam.

## Instructive projects

### [DONE] Project 1: Enterprise RAG with Bedrock Knowledge Bases 

**Exam Domain:** Domain 1 (Data Prep) & Domain 2 (FM Development)

**Goal:** Build a serverless Retrieval-Augmented Generation pipeline using Amazon Bedrock, S3, and OpenSearch Serverless.

**Steps to Execute:**
1. Create an S3 bucket and upload a few PDF documents or markdown files.
2. In the AWS Bedrock Console, create a Knowledge Base.
3. Select **Amazon Titan Embeddings G1 - Text** as your embedding model and let Bedrock automatically provision an Amazon OpenSearch Serverless vector index.
4. Test different **Chunking Strategies** in the console settings (Fixed-size chunking vs. Semantic chunking).
5. Run queries in the Bedrock test playground and compare how chunk size impacts response accuracy and context retrieval.

**Exam Concepts Tested:** Chunking strategies, Vector index creation, OpenSearch Serverless vs. pgvector, Knowledge Base sync jobs.

### [DONE] Project 2: Agentic Workflows with Bedrock Agents & Lambda

**Exam Domain:** Domain 3 (Deployment & Orchestration)

**Goal:** Build an AI Agent that calls an external API or database via AWS Lambda to fulfill user requests in real time.

**Steps to Execute:**
1. Write a simple AWS Lambda function in Python that takes input parameters (e.g., fetching mock inventory or order status) and returns JSON.
2. Create an OpenAPI 3.0 schema (JSON or YAML) defining the Lambda function's parameters and endpoint structure.
3. Create an Amazon Bedrock Agent, attach the OpenAPI schema and Lambda function as an **Action Group**, and provide clear system prompts.
4. Test the agent in the Bedrock console by asking natural language questions like *"What is the stock status for item ID 1024?"* to observe how the agent parses parameters and invokes the Lambda.

**Exam Concepts Tested:** Bedrock Agents, Action Groups, OpenAPI schemas, Lambda integration, multi-step reasoning loops.

### Project 3: SageMaker Pipeline with Managed Training & Feature Store

**Exam Domain:** Domain 1 (Data Prep), Domain 2 (Training), Domain 3 (Orchestration)

**Goal:** Build an end-to-end MLOps pipeline using SageMaker Pipelines.

**Steps to Execute:**
1. Open a SageMaker Studio Jupyter Notebook.
2. Create a SageMaker Feature Store table (Online & Offline store) and register tabular features using `sagemaker.feature_store`.
3. Construct a SageMaker Pipeline (`sagemaker.workflow.pipeline.Pipeline`) with three steps:
   - **ProcessingStep:** Clean raw data from S3 using AWS Glue.
   - **TrainingStep:** Train an XGBoost model using SageMaker built-in containers.
   - **RegisterModel:** Register the trained artifact into the SageMaker Model Registry if validation metrics exceed a target threshold.

**Exam Concepts Tested:** Feature Store (Online low-latency vs. Offline S3 historical), SageMaker Pipelines, SageMaker Processing Jobs, Model Registry.

### Project 4: Production Endpoint Deployment Strategy

**Exam Domain:** Domain 3 (Deployment & Orchestration)

**Goal:** Deploy a model to a SageMaker Endpoint and configure multi-variant A/B testing.

**Steps to Execute:**
1. Deploy a pre-trained model to a SageMaker Real-Time Endpoint using `sagemaker.model.Model.deploy()`.
2. Modify the Endpoint Config to add two **Production Variants** (Variant A with 80% traffic, Variant B with 20% traffic) to simulate an A/B test or blue/green deployment.
3. Compare when to choose:
   - **Serverless Inference:** intermittent traffic, idle cost saving
   - **Asynchronous Inference:** large payload, long processing up to 15 mins
   - **Batch Transform:** offline bulk inference

**Exam Concepts Tested:** Endpoint configurations, traffic shifting strategies, Real-Time vs. Serverless vs. Async vs. Batch Transform endpoints.

### Project 5: Bedrock Guardrails & SageMaker Model Monitor

**Exam Domain:** Domain 4 (Operating, Monitoring & Securing)

**Goal:** Implement safety controls for LLMs and set up drift monitoring for predictive ML models.

**Steps to Execute:**
1. In the Amazon Bedrock console, navigate to **Guardrails**.
2. Configure PII masking (e.g., automatically anonymizing email addresses and credit cards), blocked topic categories, and word filters.
3. Test prompt injection attempts in the Bedrock playground to see the guardrail trigger and block toxic inputs.
4. For traditional ML: Review SageMaker Model Monitor concepts by setting up a baseline dataset analysis step to generate constraints for **Data Drift** (feature distribution shifts) and **Concept Drift**.

**Exam Concepts Tested:** Bedrock Guardrails, PII redaction, prompt injection defense, SageMaker Model Monitor baseline generation and drift detection.




## Test questions
Learn details of exam results https://skillbuilder.aws/assessments/en/c28143d5-df76-4ca5-a59f-c5543340bbcd/1/summary?product_id=6HKC8HDE5C%3A001.000.000&registration_id=8342171a-8b57-5d6f-8c82-c78db0ea3848&referrer=https%3A%2F%2Fskillbuilder.aws%2Flearn%2FFCCHUGZKPJ%2Fofficial-practice-question-set-aws-certified-machine-learning--associate-mlac02--english%2F6HKC8HDE5C%3FparentId%3D5FD2DC27G1&navigation=digital&parentId=5FD2DC27G1