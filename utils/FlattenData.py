from datetime import datetime, timedelta

def flatten_output(data: list[dict]) -> list[dict]:
    """
    Function to flatten nested order data into a list of dictionaries.
        This function processes a list of orders, each containing nested product and operation data,
        and flattens it into a single list of dictionaries. Each dictionary in the returned list 
        represents a combination of order, product, and operation data, with additional task card 
        information if available.
        
        The function performs the following steps:
            - Iterates over each order in the input data.
            - Extracts non-list key-value pairs from the order as root data.
            - Iterates over each product in the order's 'products' list.
            - Determines the key for operations data ('product_operations' or 'operations').
            - Creates a list of product data dictionaries, duplicating non-list key-value pairs for each operation.
            - Iterates over each operation in the product's operations list.
            - Extracts non-list key-value pairs from the operation as operation data.
            - Matches task cards to operations based on 'task_title' and 'sort_order'.
            - Updates operation data with task card information, including 'start_at', 'end_at', and 'workspace'.
            - Merges root data, product data, and operation data into a single dictionary.
            - Appends the merged dictionary to the final order data list.
        
        Parameters:
            data (list[dict]): A list of orders, where each order is a dictionary containing nested product and operation data.
            
        Returns:
            list[dict]: A flattened list of dictionaries, each representing a combination of order, product, and operation data.
    """
    
    order_data: list[dict] = []
    for order in data:
        root_data: dict = {k: v for k,v in order.items() if not isinstance(v, list)}
        for i, product in enumerate(order['products']):
            operations_key = 'product_operations' if product.get('product_operations') else 'operations'
            product_data: list[dict] = [{k: v for k,v in product.items() if not isinstance(v, list)} for _ in range(len(product[operations_key]))]
            taskcards: list[dict] = product['task_cards']
            for j, operation in enumerate(product[operations_key]):
                operation_data: dict = {k: v for k,v in operation.items() if not isinstance(v, list)}
                matching_taskcards = [tc for tc in taskcards if tc['task_title'] == operation_data['task_title'] and tc['sort_order'] == operation_data['sort_order']]
                if matching_taskcards:
                    operation_data['start_at'] = min(tc['start_at'] for tc in matching_taskcards)
                    operation_data['end_at'] = (datetime.strptime(operation_data['start_at'], '%Y-%m-%dT%H:%M:%S.%fZ') + timedelta(seconds=operation_data['task_duration'])).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                    operation_data['workspace'] = matching_taskcards[0]['workspace']
                product_data[j] = product_data[j] | operation_data
                product_data[j] = root_data | product_data[j]
            for p in product_data:
                order_data.append(p)
    return order_data

def flatten_input(data: list[dict]) -> list[dict]:
    """
    Function to flatten nested order data into a list of dictionaries.
        This function processes a list of order dictionaries, each containing nested product and operation data,
        and flattens it into a single list of dictionaries. Each dictionary in the returned list represents a 
        combination of order, product, and operation data, with certain fields adjusted or excluded as specified.
        
        - Extracts root-level order data that is not a list.
        - Iterates through each product in the order and processes its operations.
        - Combines product and operation data, adjusting 'task_duration' if 'task_duration_done' is present.
        - Merges root order data with each product-operation combination.
        - Appends the flattened data to the result list.
        
        Parameters:
            data (list[dict]): A list of order dictionaries, where each order contains nested product and operation data.
            
        Returns:
            list[dict]: A flattened list of dictionaries, each representing a combination of order, product, and operation data.
    """
    
    
    order_data: list[dict] = []
    for order in data:
        root_data: dict = {k: v for k,v in order.items() if not isinstance(v, list)}
        for i, product in enumerate(order['products']):
            operations_key = 'product_operations' if product.get('product_operations') else 'operations'
            product_data: list[dict] = [{k: v for k,v in product.items() if not isinstance(v, list)} for _ in range(len(product[operations_key]))]
            for j, operation in enumerate(product[operations_key]):
                operation_data: dict = {k: v for k,v in operation.items() if k != 'task_duration_done' and not isinstance(v, list)}
                if operation.get('task_duration_done'):
                    operation_data['task_duration'] -= operation['task_duration_done']
                product_data[j] = product_data[j] | operation_data
                product_data[j] = root_data | product_data[j]
                
            for p in product_data:
                order_data.append(p)
    return order_data