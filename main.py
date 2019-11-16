from models import model
from models import dataset

dataset = dataset.load_data()
results = model.run_svc(dataset)

model.eval_results(results)