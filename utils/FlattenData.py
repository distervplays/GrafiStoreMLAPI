from datetime import datetime, timedelta

def flatten_output(data: list[dict]) -> list[dict]:
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