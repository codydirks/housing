import hashlib
import time
from kubernetes import client, config

MODEL_PATH = "/models/production_model.pkl"  # Use a shared volume path


def get_checksum(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def restart_deployment():
    config.load_incluster_config()
    api = client.AppsV1Api()
    # Patch deployment with an annotation to force restart
    api.patch_namespaced_deployment(
        name="housing-app-deployment",
        namespace="default",
        body={
            "spec": {"template": {"metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": str(time.time())}}}}
        },
    )


def main():
    last_checksum = get_checksum(MODEL_PATH)
    while True:
        time.sleep(10)
        current_checksum = get_checksum(MODEL_PATH)
        if current_checksum != last_checksum:
            restart_deployment()
            last_checksum = current_checksum


if __name__ == "__main__":
    main()
