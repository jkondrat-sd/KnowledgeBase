## [DONE] Project 2: Agentic Workflows with Bedrock AgentCore & Lambda
**Exam Domain:** Domain 3 (Deployment & Orchestration)

**Goal:** Build an AI Agent that calls an external API or database via AWS Lambda to fulfill user requests in real time.

> **Why AgentCore and not Bedrock Agents?**
> As of July 30, 2026, Bedrock Agents is renamed **Bedrock Agents Classic** and is in maintenance mode. New accounts (no Bedrock Agents usage in the past 12 months) get this error on agent creation:
> `Bedrock Agents is in Maintenance Mode. New agent creation is not available for accounts without prior service usage.`
> There is no exception process. AWS recommends **Amazon Bedrock AgentCore** instead. Knowledge Bases and Guardrails are unaffected.
> See [Agents Classic maintenance mode](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-classic-maintenance-mode.html).

### AgentCore building blocks

```
User ──▶ Harness ──(MCP)──▶ Gateway ──▶ Target ──▶ Lambda
         (the agent)        (tool hub)   (one backend + its tools)
```

- **Harness:** the agent itself. You declare the model, system prompt and tools as configuration, and AgentCore runs the orchestration loop: call the model, let it pick a tool, run the tool, feed the result back, repeat until it can answer. AgentCore also provides the infrastructure underneath. Each chat session runs in its own isolated microVM with memory, identity and tracing built in. It's the managed equivalent of a Bedrock Agents Classic agent, with no agent code to write or host.
- **Gateway:** a managed **MCP server** that sits between the agent and your tools. It turns ordinary backends (Lambda functions, REST/OpenAPI APIs, other MCP servers) into tools any MCP-compatible agent can discover and call, through a single URL. It handles **inbound auth** (who may call the gateway: IAM, JWT/Cognito) and **outbound auth** (how the gateway calls your backend: IAM role, OAuth, API key). It can also apply policies, semantic tool search and logging. One gateway can serve many agents.
- **Target:** one backend attached to a gateway, plus the schema of the tools it provides. Here, the target is the Lambda function and `tools.json` defines its two tools. A gateway can have many targets (e.g. a Lambda, a REST API and an MCP server), and the agent sees all their tools together, named `<targetName>___<toolName>` (e.g. `InventoryTarget___get_inventory`). It's the closest equivalent of a Classic **Action Group**.

### Classic vs. AgentCore mapping

| Bedrock Agents Classic | AgentCore equivalent |
|---|---|
| Agent created in the Bedrock console | **Harness**: config-based agent (model, tools, system prompt) created in the AgentCore console |
| Action Group (OpenAPI schema + Lambda) | **Gateway** with a **Lambda target**, exposed to the agent as MCP tools |
| OpenAPI 3.0 schema | JSON **tool schema** (`name`, `description`, `inputSchema`) |
| Agent instructions | System prompt on the harness |
| Trace view in the test window | AgentCore Observability traces (harness page + CloudWatch GenAI Observability) |

**Steps to Execute:**

**Prerequisites:** Do everything in the same [region where AgentCore is available](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-regions.html) (e.g. `us-east-1`). AgentCore lives in its own console: [Amazon Bedrock AgentCore console](https://console.aws.amazon.com/bedrock-agentcore/home), separate from the Bedrock console.

> AWS documents the Gateway console steps but not the harness console steps yet. Labels marked *(approx.)* may differ slightly.

1. **Write the Lambda function** in Python (Lambda console → **Create function** → Python 3.12). It returns mock inventory or order status as JSON (see [lambda.py](lambda.py)). The Gateway event format is different from Classic Action Groups:
   - `event` is just the tool arguments, e.g. `{"itemId": "1024"}`. There is no `apiPath` or `parameters` list.
   - The tool name is in `context.client_context.custom['bedrockAgentCoreToolName']`, formatted `<targetName>___<toolName>` (three underscores). Strip the prefix to route.
   - Return plain JSON. No `messageVersion`/`responseBody` wrapper.
   - The Lambda console's **Test** button doesn't set `client_context`, so add a fallback (e.g. route on which argument is present) to test it there. A test event is just `{"itemId": "1024"}`.
   - Click **Deploy**, then copy the function **ARN** from the top of the page. You need it in step 3.

2. **Write the tool schema** (save it as `tools.json` in this folder; you paste it into the console in step 3) describing each tool the Lambda implements. This replaces the OpenAPI schema. The `description` fields are what the model reads to decide which tool to call, so make them specific:
   ```json
   [
     {
       "name": "get_inventory",
       "description": "Get the current stock level and warehouse location for a product, given its item ID.",
       "inputSchema": {
         "type": "object",
         "properties": {
           "itemId": { "type": "string", "description": "The numeric product item ID, e.g. 1024" }
         },
         "required": ["itemId"]
       }
     },
     {
       "name": "get_order_status",
       "description": "Get the shipping status and estimated delivery date for a customer order, given its order ID.",
       "inputSchema": {
         "type": "object",
         "properties": {
           "orderId": { "type": "string", "description": "The order ID, e.g. A100" }
         },
         "required": ["orderId"]
       }
     }
   ]
   ```

3. **Create a Gateway with the Lambda as a target** in the [AgentCore console](https://console.aws.amazon.com/bedrock-agentcore/home):
   1. Left nav → **Gateways** → **Create gateway**.
   2. **Gateway details:** name it `InventoryGateway`. Under **Additional configurations**, tick **Exception level debug** (useful error messages while learning).
   3. **Inbound Auth configurations:** choose **IAM** if offered, because the harness calls gateways with IAM (SigV4) by default. If only Cognito/JWT options appear, pick **Quick create configurations with Cognito**. In that case the harness needs OAuth outbound auth to the gateway (see Troubleshooting).
   4. **Permissions:** **Use an IAM service role** → **Create and use a new service role**. This role is what the gateway uses to invoke your Lambda.
   5. **Target** section:
      - **Target name:** `InventoryTarget`
      - **Target type:** **Lambda** → paste your Lambda ARN
      - Tool schema: choose **inline** and paste the contents of `tools.json`
      - **Outbound Auth:** IAM role (the gateway service role)
   6. **Create gateway**. When the status is **Ready**, copy the **Gateway ARN** from the details page.

4. **Create the harness** and attach the gateway as a tool. AgentCore console → left nav → **Harness** *(approx.)* → **Create harness** *(approx.)*:
   - **Name:** `inventoryagent`
   - **Model:** a Bedrock model, e.g. Claude Sonnet (the default if you don't pick one)
   - **System prompt / instructions:** *"You are an inventory assistant. Use the tools to look up stock levels and order status. Never guess stock numbers."*
   - **Tools:** add an **AgentCore Gateway** tool → paste the Gateway ARN → outbound auth **AWS IAM**
   - **Execution role:** let the console create one. It needs `bedrock-agentcore:InvokeGateway` on your gateway ARN (see Troubleshooting).
   - Create it and wait for the status to be **Ready**.

5. **Test the agent** with natural language questions and watch how it picks tools and fills in arguments:
   - If the harness page has a test/chat panel, use it.
   - Otherwise open **CloudShell** (terminal icon in the console top bar) and run this, replacing the ARN and region:
     ```bash
     python3 - <<'EOF'
     import boto3, uuid
     client = boto3.client("bedrock-agentcore", region_name="us-east-1")
     resp = client.invoke_harness(
         harnessArn="<YOUR_HARNESS_ARN>",
         runtimeSessionId=str(uuid.uuid4()),
         messages=[{"role": "user", "content": [{"text": "What is the stock status for item ID 1024?"}]}],
     )
     for event in resp["stream"]:
         delta = event.get("contentBlockDelta", {}).get("delta", {})
         if "text" in delta:
             print(delta["text"], end="", flush=True)
         elif "runtimeClientError" in event:
             print("\nError:", event["runtimeClientError"]["message"])
     print()
     EOF
     ```
     CloudShell's boto3 may be older than the harness API. If you get `has no attribute 'invoke_harness'`, run `pip3 install -U boto3` first.
   - **Traces:** the harness page in the AgentCore console has an observability widget, and **CloudWatch → GenAI Observability → Harnesses** shows each step: model call, tool choice, arguments, Lambda result.

   Good test prompts:
   - *"Is item 1024 in stock?"* (single tool call)
   - *"What's the status of order A102?"* (single tool call)
   - *"Order A101 is for which item, and how many are left?"* (two chained calls: multi-step reasoning)

**Exam Concepts Tested:** Agent orchestration loops (Bedrock Agents Classic & AgentCore harness), Action Groups vs. Gateway targets, OpenAPI vs. MCP tool schemas, Lambda integration, multi-step reasoning.

### Troubleshooting
- **Agent can't call the gateway (AccessDenied):** in IAM → Roles, open the harness execution role and add a policy allowing `bedrock-agentcore:InvokeGateway` on the gateway ARN.
- **Gateway created with Cognito instead of IAM:** the harness gateway tool must use **OAuth** outbound auth with a credential provider (AgentCore console → **Identity**) holding the Cognito client ID/secret. Recreating the gateway with IAM inbound auth is simpler, if the console offers it.
- **Gateway can't invoke the Lambda:** the gateway service role needs `lambda:InvokeFunction` on the function.
- **Tool calls fail with no useful message:** check that *Exception level debug* is on, then look at the gateway's logs in CloudWatch → Log groups → `/aws/bedrock-agentcore/gateways/<GATEWAY_ID>`. Also check the Lambda's own logs (Lambda → Monitor → View CloudWatch logs).
- **Tool names:** the agent sees tools as `InventoryTarget___get_inventory`. That's why the Lambda strips the prefix.
- **Clean up:** see [Teardown](#teardown) below.

### Teardown
Delete things in this order, so nothing is left calling something that's already gone. Do it all in the **same region** you built in. Charges you've already incurred (mostly model tokens) still appear on the bill. Once this is done, nothing should keep billing.

1. **Harness** (AgentCore console → **Harness** *(approx.)*): select `inventoryagent` → **Delete**. If you created any harness **endpoints** or extra versions, delete those first if the console asks.
2. **Memory** (AgentCore console → **Memory**): the harness creates a memory resource by default, and stored events and records are billed monthly. Delete any memory belonging to the harness if it's still listed after step 1.
3. **Gateway** (AgentCore console → **Gateways**): select `InventoryGateway` → delete its target `InventoryTarget` if required, then delete the gateway. This stops the tool-indexing charge.
4. **Identity** (AgentCore console → **Identity**): delete any OAuth/API-key credential providers you created. You only have these if you went the Cognito route.
5. **Cognito** (only if you used *Quick create configurations with Cognito*): Cognito console → **User pools** → delete the pool the gateway created. Remove its domain first if asked.
6. **Lambda** (Lambda console → **Functions**): select the inventory function → **Actions → Delete**.
7. **CloudWatch logs** (CloudWatch → **Log groups**): delete the groups for this project. Log storage is billed per GB-month and never expires by default:
   - `/aws/lambda/<your-function-name>`
   - `/aws/bedrock-agentcore/...` (gateway and harness/runtime logs)
   - `aws/spans` *(only if you enabled CloudWatch Transaction Search for traces and use nothing else that writes to it)*
8. **IAM roles** (IAM → **Roles**): delete the roles created for this project, i.e. the gateway service role, the harness execution role and the Lambda execution role (`<function-name>-role-...`). IAM is free, but stale roles with permissions are a security risk.
9. **S3** (only if you uploaded `tools.json` to a bucket instead of pasting it inline): empty and delete the bucket.

**Nothing to delete:**
- **Bedrock model subscription:** the AWS Marketplace subscription created on first use has no standing fee. You only pay per token when the model is called.
- **CloudShell:** free. Its home directory is deleted automatically after 120 days of inactivity.
- **Gateway workload identity:** created automatically with the gateway and free.

**Verify zero cost:**
- AgentCore console: the Harness, Gateways and Memory lists are empty.
- **Billing → Bills** (or **Cost Explorer**, grouped by *Service*), checked 1–2 days later: Bedrock, Bedrock AgentCore, Lambda and CloudWatch should stop accruing new charges.
- Set an **AWS Budgets** alert (e.g. $1 monthly cost budget with an email alert) so you hear about anything you missed.

### Notes
- [test_data.json](test_data.json) and [openai_schema.yml](openai_schema.yml) are from the original **Classic** Action Group version: the `apiPath`-style event Bedrock Agents sends to the Lambda, and its OpenAPI schema. Keep them for exam revision. The exam may still ask about Action Groups.
- [lambda.py](lambda.py) uses the Gateway tool format. It reads the tool name from the client context and falls back to inferring it from the arguments when run from the Lambda console's Test button.

### References
- [AgentCore harness](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness.html) · [Get started](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness-get-started.html) · [Tools](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness-tools.html)
- [Create a gateway in the console](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-create-console.html) · [Lambda targets](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-add-target-lambda.html)
