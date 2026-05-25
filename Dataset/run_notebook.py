import sys
from pathlib import Path
nb_path = Path(r"C:\Users\rodri\OneDrive\Documentos\8vo\Dataset\main.ipynb")
print('Notebook path:', nb_path)
try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
except Exception as e:
    print('ImportError:', e)
    sys.exit(2)

try:
    nb = nbformat.read(str(nb_path), as_version=4)
    ep = ExecutePreprocessor(timeout=600)
    ep.preprocess(nb, {'metadata': {'path': str(nb_path.parent)}})
    print('Execution completed successfully')
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
