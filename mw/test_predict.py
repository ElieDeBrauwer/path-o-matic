import sys
import predict as predict

if len(sys.argv) < 5:
    print('Usage: python ' +  sys.argv[0] + ' <model_dir> <model_name> <prediction_version> <pngfile>')
    sys.exit(1)

model_dir = sys.argv[1]
model_name = sys.argv[2]
prediction_version_val = sys.argv[3]
filename = sys.argv[4]
predict.restore(model_dir, model_name, prediction_version_val)
predict_val  = predict.predict(filename)
print("predict:val", predict_val)
