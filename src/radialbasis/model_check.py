import sys
from tensorflow.keras.models import load_model
print("Attempting to load %s" % sys.argv[1])
model = load_model(sys.argv[1])

print(model.summary())

# Get accuracy score for whole dataset ?
