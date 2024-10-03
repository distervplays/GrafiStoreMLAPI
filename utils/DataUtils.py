import pandas as pd
import numpy as np

def flatten_orders(orders):
    # Initialize an empty list to store the flattened data
    flattened_data = []

    # Iterate through each order
    for order in orders:
        order_id = order.get('order_id')
        delivery_date = order.get('delivery_date')
        order_created_at = order.get('order_created_at')

        # Iterate through each product in the order
        for product in order.get('products', []):
            material = product.get('material')
            color = product.get('color')

            # Get product operations and task cards
            product_operations = product.get('product_operations', [])
            task_cards = product.get('task_cards', [])

            # Create a mapping from sort_order to task card for quick lookup
            task_card_map = {tc.get('sort_order'): tc for tc in task_cards}

            # Iterate through each product operation
            for operation in product_operations:
                sort_order = operation.get('sort_order')
                task_title = operation.get('task_title')
                task_duration = operation.get('task_duration')
                workspace = operation.get('workspace')

                # Get the corresponding task card using sort_order
                task_card = task_card_map.get(sort_order, {})
                start_at = task_card.get('start_at')
                end_at = task_card.get('end_at')

                # Append the flattened row to the data list
                flattened_data.append({
                    'order_id': order_id,
                    'delivery_date': delivery_date,
                    'order_created_at': order_created_at,
                    'material': material,
                    'color': color,
                    'task_title': task_title,
                    'task_duration': task_duration,
                    'sort_order': sort_order,
                    'workspace': workspace,
                    'start_at': start_at,
                    'end_at': end_at
                })

    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(flattened_data)

    # Convert date strings to datetime objects
    date_columns = ['delivery_date', 'order_created_at', 'start_at', 'end_at']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    return df