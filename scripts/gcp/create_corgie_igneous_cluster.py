# pylint: disable=missing-docstring,line-too-long
import argparse
import subprocess

CREATE_SERVICE_ACCOUNT_TMPL = (
    "gcloud iam service-accounts create {CLUSTER_NAME}-worker --project={PROJECT_NAME}"
)

CREATE_COMMAND_TMPL = 'gcloud beta container --project "{PROJECT_NAME}" clusters create "{CLUSTER_NAME}" --region "{REGION}" --no-enable-basic-auth --release-channel "regular" --machine-type "e2-medium" --image-type "COS_CONTAINERD" --disk-type "pd-standard" --disk-size "10" --metadata disable-legacy-endpoints=true --service-account "{CLUSTER_NAME}-worker@{PROJECT_NAME}.iam.gserviceaccount.com" --spot --enable-autoscaling --num-nodes 1 --total-min-nodes=1 --total-max-nodes=10 --logging=SYSTEM,WORKLOAD --monitoring=SYSTEM --enable-ip-alias --network "projects/{PROJECT_NAME}/global/networks/default" --subnetwork "projects/{PROJECT_NAME}/regions/{REGION}/subnetworks/default" --no-enable-intra-node-visibility --enable-dataplane-v2 --no-enable-master-authorized-networks --addons HorizontalPodAutoscaling,GcePersistentDiskCsiDriver --enable-autoupgrade --enable-autorepair --max-surge-upgrade 0 --max-unavailable-upgrade 1 --maintenance-window-start "2022-01-19T05:00:00Z" --maintenance-window-end "2022-01-20T05:00:00Z" --maintenance-window-recurrence "FREQ=WEEKLY;BYDAY=SA,SU" --labels owner={USERNAME} --workload-pool "{PROJECT_NAME}.svc.id.goog" --enable-shielded-nodes --node-locations {NODE_LOCATIONS} --enable-image-streaming'

ADD_CPU_COMMAND_TMPL = 'gcloud beta container --project "{PROJECT_NAME}" node-pools create "cpu-t2d-standard-16" --cluster "{CLUSTER_NAME}" --region "{REGION}" --machine-type "t2d-standard-16" --image-type "COS_CONTAINERD" --disk-type "pd-standard" --disk-size "64" --metadata disable-legacy-endpoints=true --service-account "{CLUSTER_NAME}-worker@{PROJECT_NAME}.iam.gserviceaccount.com" --spot --enable-autoupgrade --enable-autorepair --max-surge-upgrade 1 --max-unavailable-upgrade 0 --max-pods-per-node "16" --node-locations {NODE_LOCATIONS} --enable-autoscaling --num-nodes 0 --total-min-nodes=0 --total-max-nodes=10 --node-taints=worker-pool=true:NoSchedule --enable-image-streaming'
ADD_GPU_COMMAND_TMPL = 'gcloud beta container --project "{PROJECT_NAME}" node-pools create "gpu-g2-standard-8" --cluster "{CLUSTER_NAME}" --region "{REGION}" --machine-type "g2-standard-8" --accelerator "type=nvidia-l4,count=1" --image-type "COS_CONTAINERD" --disk-type "pd-balanced" --disk-size "64" --metadata disable-legacy-endpoints=true --service-account "{CLUSTER_NAME}-worker@{PROJECT_NAME}.iam.gserviceaccount.com" --spot --enable-autoupgrade --enable-autorepair --max-surge-upgrade 1 --max-unavailable-upgrade 0 --max-pods-per-node "16" --node-locations {NODE_LOCATIONS} --enable-autoscaling --num-nodes 0 --total-min-nodes=0 --total-max-nodes=10 --node-taints=worker-pool=true:NoSchedule --node-labels=gpu-count=1 --enable-image-streaming'

ADD_WORKLOAD_IDENTITY_TMPL = 'gcloud container clusters get-credentials {CLUSTER_NAME} --region {REGION} --project {PROJECT_NAME} \
    && gcloud projects add-iam-policy-binding {PROJECT_NAME} --member "serviceAccount:{CLUSTER_NAME}-worker@{PROJECT_NAME}.iam.gserviceaccount.com" --role "roles/storage.objectUser" \
    && gcloud projects add-iam-policy-binding {PROJECT_NAME} --member "serviceAccount:{CLUSTER_NAME}-worker@{PROJECT_NAME}.iam.gserviceaccount.com" --role "roles/artifactregistry.reader" \
    && gcloud projects add-iam-policy-binding {PROJECT_NAME} --member "serviceAccount:{CLUSTER_NAME}-worker@{PROJECT_NAME}.iam.gserviceaccount.com" --role "roles/serviceusage.serviceUsageConsumer" \
    && gcloud projects add-iam-policy-binding {PROJECT_NAME} --member "serviceAccount:zutils-worker-x0@zetta-research.iam.gserviceaccount.com" --role "roles/container.developer" \
    && gcloud projects add-iam-policy-binding zetta-research --member "serviceAccount:{CLUSTER_NAME}-worker@{PROJECT_NAME}.iam.gserviceaccount.com" --role "roles/storage.objectUser" \
    && gcloud projects add-iam-policy-binding zetta-research --member "serviceAccount:{CLUSTER_NAME}-worker@{PROJECT_NAME}.iam.gserviceaccount.com" --role "roles/datastore.user" \
    && gcloud iam service-accounts add-iam-policy-binding {CLUSTER_NAME}-worker@{PROJECT_NAME}.iam.gserviceaccount.com --role roles/iam.workloadIdentityUser --member "serviceAccount:{PROJECT_NAME}.svc.id.goog[default/default]" --project {PROJECT_NAME} \
    && kubectl annotate serviceaccount default --namespace default iam.gke.io/gcp-service-account={CLUSTER_NAME}-worker@{PROJECT_NAME}.iam.gserviceaccount.com'

CREATE_ARTIFACT_REGISTRY_REPO_TMPL = "gcloud artifacts repositories create zutils --repository-format=docker --location={REGION} --project={PROJECT_NAME}"

CONFIGURE_DRIVERS_COMMAND_TMPL = "gcloud container clusters get-credentials {CLUSTER_NAME} --region {REGION} --project {PROJECT_NAME}; kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml"

CONFIGURE_KEDA_COMMAND_TMPL = 'helm repo add kedacore https://kedacore.github.io/charts && helm repo update kedacore \
    && gcloud container clusters get-credentials {CLUSTER_NAME} --region {REGION} --project {PROJECT_NAME} \
    && helm install keda kedacore/keda --namespace keda --create-namespace'


def main():  # pylint: disable=too-many-statements
    parser = argparse.ArgumentParser(description="Creates a corgie/igneous cluster on GCP.")
    parser.add_argument(
        "--cluster_name", "-n", type=str, required=True, help="Name of the new cluster."
    )
    parser.add_argument(
        "--username",
        "-u",
        type=str,
        required=True,
        help="Name of the GCP user (for the `owner` tag).",
    )
    parser.add_argument(
        "--project_name",
        "-p",
        type=str,
        required=True,
        help="Name of the GCP project (e.g. zetta-research).",
    )
    parser.add_argument(
        "--region",
        "-r",
        type=str,
        default="us-east1",
        help="GCP region (e.g. us-east1).",
    )
    parser.add_argument(
        "--cluster_version",
        "-v",
        type=str,
        default=None,
        help='Version of the GKE (e.g. "1.23.6-gke.1700").',
    )
    parser.add_argument(
        "--cpu_zones",
        type=str,
        default="b,c,d",
        help='GCP Zones for CPU node locations (e.g. "b,c,d" for us-east1-b, us-east1-c, and us-east1-d).',
    )
    parser.add_argument(
        "--gpu_zones",
        type=str,
        default="b,c,d",
        help='GCP Zones for GPU node locations (e.g. "b,c,d" for us-east1-b, us-east1-c, and us-east1-d).',
    )
    parser.add_argument("--add_cpu", action="store_true", help="Add CPU node pool.")
    parser.add_argument("--add_gpu", action="store_true", help="Add GPU node pool.")
    parser.add_argument(
        "--add_repo", action="store_true", help="Add Artifact Registry Repository."
    )

    args = parser.parse_args()
    # cluster_version = args.cluster_version
    # if cluster_version is None:

    if args.add_cpu:
        system_zone = f"{args.region}-{args.cpu_zones.split(',')[0]}"
    elif args.add_gpu:
        system_zone = f"{args.region}-{args.gpu_zones.split(',')[0]}"
    else:
        print("Neither CPU nor GPU cluster requested.")
        return

    create_iam_command = CREATE_SERVICE_ACCOUNT_TMPL
    create_iam_command = create_iam_command.replace("{PROJECT_NAME}", args.project_name)
    create_iam_command = create_iam_command.replace("{CLUSTER_NAME}", args.cluster_name)
    print(f"Running: \n{create_iam_command}")
    subprocess.call(create_iam_command, shell=True)

    create_command = CREATE_COMMAND_TMPL
    create_command = create_command.replace("{REGION}", args.region)
    create_command = create_command.replace("{USERNAME}", args.username)
    create_command = create_command.replace("{PROJECT_NAME}", args.project_name)
    create_command = create_command.replace("{CLUSTER_NAME}", args.cluster_name)
    create_command = create_command.replace("{NODE_LOCATIONS}", system_zone)
    print(f"Running: \n{create_command}")
    subprocess.call(create_command, shell=True)

    if args.add_cpu:
        cpu_zones = [f"{args.region}-{zone_letter}" for zone_letter in args.cpu_zones.split(",")]
        add_cpu_command = ADD_CPU_COMMAND_TMPL
        add_cpu_command = add_cpu_command.replace("{REGION}", args.region)
        add_cpu_command = add_cpu_command.replace("{PROJECT_NAME}", args.project_name)
        add_cpu_command = add_cpu_command.replace("{CLUSTER_NAME}", args.cluster_name)
        add_cpu_command = add_cpu_command.replace("{NODE_LOCATIONS}", ",".join(cpu_zones))
        print(f"Running: \n{add_cpu_command}")
        subprocess.call(add_cpu_command, shell=True)

    if args.add_gpu:
        gpu_zones = [f"{args.region}-{zone_letter}" for zone_letter in args.gpu_zones.split(",")]
        add_gpu_command = ADD_GPU_COMMAND_TMPL
        add_gpu_command = add_gpu_command.replace("{REGION}", args.region)
        add_gpu_command = add_gpu_command.replace("{PROJECT_NAME}", args.project_name)
        add_gpu_command = add_gpu_command.replace("{CLUSTER_NAME}", args.cluster_name)
        add_gpu_command = add_gpu_command.replace("{NODE_LOCATIONS}", ",".join(gpu_zones))
        print(f"Running: \n{add_gpu_command}")
        subprocess.call(add_gpu_command, shell=True)

        configure_drivers_command = CONFIGURE_DRIVERS_COMMAND_TMPL
        configure_drivers_command = configure_drivers_command.replace("{REGION}", args.region)
        configure_drivers_command = configure_drivers_command.replace(
            "{PROJECT_NAME}", args.project_name
        )
        configure_drivers_command = configure_drivers_command.replace(
            "{CLUSTER_NAME}", args.cluster_name
        )
        print(f"Running: \n{configure_drivers_command}")
        subprocess.call(configure_drivers_command, shell=True)

    # add workload identity; also grants "roles/storage.objectUser" and "roles/artifactregistry.reader"
    add_wi_command = ADD_WORKLOAD_IDENTITY_TMPL
    add_wi_command = add_wi_command.replace("{REGION}", args.region)
    add_wi_command = add_wi_command.replace("{PROJECT_NAME}", args.project_name)
    add_wi_command = add_wi_command.replace("{CLUSTER_NAME}", args.cluster_name)
    print(f"Running: \n{add_wi_command}")
    subprocess.call(add_wi_command, shell=True)

    if args.add_repo:
        create_repo_command = CREATE_ARTIFACT_REGISTRY_REPO_TMPL
        create_repo_command = create_repo_command.replace("{REGION}", args.region)
        create_repo_command = create_repo_command.replace("{PROJECT_NAME}", args.project_name)
        print(f"Running: \n{create_repo_command}")
        subprocess.call(create_repo_command, shell=True)

    create_keda_command = CONFIGURE_KEDA_COMMAND_TMPL
    create_keda_command = create_keda_command.replace("{REGION}", args.region)
    create_keda_command = create_keda_command.replace("{PROJECT_NAME}", args.project_name)
    create_keda_command = create_keda_command.replace("{CLUSTER_NAME}", args.cluster_name)
    print(f"Running: \n{create_keda_command}")
    subprocess.call(create_keda_command, shell=True)


if __name__ == "__main__":
    main()
