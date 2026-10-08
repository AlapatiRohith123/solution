import os

with open('tests/checks.py', 'r') as f:
    content = f.read()

content = content.replace('    metrics = {}', '    metrics = {"mcc": 0.0, "auc_pr": 0.0, "worst_channel_f1": 0.0}')

return_line = "    return {'criteria': results, 'metrics': metrics}"
new_return = """
    criteria_list = [
        {"id": "inference_exists", "passed": results.get("inference_exists", False)},
        {"id": "metrics_json_exists", "passed": results.get("metrics_json_exists", False)},
        {"id": "inference_runs", "passed": results.get("inference_runs", False)},
        {"id": "performance_measured", "passed": results.get("performance_measured", False)}
    ]
    return {'criteria': criteria_list, 'metrics': metrics}
"""
content = content.replace(return_line, new_return)

with open('tests/checks.py', 'w') as f:
    f.write(content)
