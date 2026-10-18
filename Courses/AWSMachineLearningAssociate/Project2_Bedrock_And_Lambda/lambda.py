import json

# Mock data
INVENTORY = {
    "1024": {"itemId": "1024", "name": "Wireless Mouse", "stock": 42, "warehouse": "Dublin"},
    "2048": {"itemId": "2048", "name": "Mechanical Keyboard", "stock": 0, "warehouse": "Frankfurt"},
    "4096": {"itemId": "4096", "name": "USB-C Hub", "stock": 7, "warehouse": "Dublin"},
}

ORDERS = {
    "A100": {"orderId": "A100", "status": "SHIPPED", "itemId": "1024", "eta": "2026-10-18"},
    "A101": {"orderId": "A101", "status": "PROCESSING", "itemId": "4096", "eta": "2026-10-22"},
    "A102": {"orderId": "A102", "status": "BACKORDERED", "itemId": "2048", "eta": None},
}

# Gateway prefixes tool names with the target name: "InventoryTarget___get_inventory"
DELIMITER = "___"


def get_tool_name(event, context):
    """Gateway passes the tool name in the client context, not the event."""
    client_context = getattr(context, "client_context", None)
    if client_context and client_context.custom:
        full_name = client_context.custom.get("bedrockAgentCoreToolName", "")
        if DELIMITER in full_name:
            return full_name.split(DELIMITER, 1)[1]
        return full_name

    # Fallback for the Lambda console's Test button, which has no client context:
    # infer the tool from which argument is present.
    if "itemId" in event:
        return "get_inventory"
    if "orderId" in event:
        return "get_order_status"
    return None


def get_inventory(event):
    item_id = str(event.get("itemId", ""))
    item = INVENTORY.get(item_id)
    if item:
        return item
    return {"error": f"Item {item_id} not found"}


def get_order_status(event):
    order_id = str(event.get("orderId", "")).upper()
    order = ORDERS.get(order_id)
    if order:
        return order
    return {"error": f"Order {order_id} not found"}


TOOLS = {
    "get_inventory": get_inventory,
    "get_order_status": get_order_status,
}


def lambda_handler(event, context):
    # The event is just the tool arguments, e.g. {"itemId": "1024"}
    tool_name = get_tool_name(event, context)
    print("Tool:", tool_name, "Event:", json.dumps(event))  # Shows up in CloudWatch, handy for debugging

    handler = TOOLS.get(tool_name)
    if handler is None:
        return {"error": f"Unknown tool {tool_name}"}

    # Gateway expects plain JSON back, no wrapper
    return handler(event)
