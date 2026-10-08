
for fname in ['solution/train.py', 'tests/checks.py']:
    with open(fname, 'r') as f:
        content = f.read()
    content = content.replace('/app/task_inputs', '{app_dir}/task_inputs')
    content = content.replace('/app/artifacts', '{app_dir}/artifacts')
    content = content.replace('/app/tests', '{app_dir}/tests')
    
    if fname == 'solution/train.py':
        header = '    app_dir = os.environ.get("APP_DIR", "/app")\n'
        content = content.replace('    # Load dataset', header + '    # Load dataset')
        content = content.replace('df = pd.read_csv(args.input)', 'app_dir = os.environ.get("APP_DIR", "/app")\n    df = pd.read_csv(args.input)')
        
        # fix strings
        content = content.replace("'{app_dir}/task_inputs/dataset.csv'", 'f"{app_dir}/task_inputs/dataset.csv"')
        content = content.replace("'{app_dir}/artifacts'", 'f"{app_dir}/artifacts"')
        content = content.replace("'{app_dir}/artifacts/model.pth'", 'f"{app_dir}/artifacts/model.pth"')
        content = content.replace("'{app_dir}/artifacts/inference.py'", 'f"{app_dir}/artifacts/inference.py"')
        content = content.replace("'{app_dir}/artifacts/metrics.json'", 'f"{app_dir}/artifacts/metrics.json"')
        
    if fname == 'tests/checks.py':
        header = '    app_dir = os.environ.get("APP_DIR", "/app")\n'
        content = content.replace('    results = {}', '    results = {}\n' + header)
        content = content.replace("'{app_dir}/artifacts/inference.py'", 'f"{app_dir}/artifacts/inference.py"')
        content = content.replace("'{app_dir}/artifacts/metrics.json'", 'f"{app_dir}/artifacts/metrics.json"')
        content = content.replace("'{app_dir}/tests/data/hidden_test.csv'", 'f"{app_dir}/tests/data/hidden_test.csv"')
        content = content.replace("'{app_dir}/artifacts/hidden_preds.csv'", 'f"{app_dir}/artifacts/hidden_preds.csv"')

    with open(fname, 'w') as f:
        f.write(content)
