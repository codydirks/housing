kubectl delete -f k8s --ignore-not-found

docker build -t housing-app:latest -f ./Dockerfile .

docker build -t model-watcher:latest -f ./model-watcher.Dockerfile .

kubectl apply -f k8s

kubectl wait --for=condition=ready pod -l app=housing-app --timeout=60s

sleep 5

curl http://localhost:8000/health