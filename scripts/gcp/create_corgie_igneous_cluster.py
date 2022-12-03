# pylint: disable=missing-docstring,line-too-long
import argparse
import subprocess

CREATE_COMMAND_TMPL = 'gcloud beta container --project "{PROJECT_NAME}" clusters create "{CLUSTER_NAME}" --region "{REGION}" --no-enable-basic-auth --release-channel "regular" --machine-type "e2-highmem-4" --image-type "COS_CONTAINERD" --disk-type "pd-standard" --disk-size "64" --metadata disable-legacy-endpoints=true --scopes "https://www.googleapis.com/auth/devstorage.read_write","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/pubsub","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" --max-pods-per-node "16" --preemptible --num-nodes "1" --logging=SYSTEM,WORKLOAD --monitoring=SYSTEM --enable-ip-alias --network "projects/{PROJECT_NAME}/global/networks/default" --subnetwork "projects/{PROJECT_NAME}/regions/{REGION}/subnetworks/default" --no-enable-intra-node-visibility --default-max-pods-per-node "16" --enable-dataplane-v2 --no-enable-master-authorized-networks --addons HorizontalPodAutoscaling,GcePersistentDiskCsiDriver --enable-autoupgrade --enable-autorepair --max-surge-upgrade 1 --max-unavailable-upgrade 0 --maintenance-window-start "2022-01-19T05:00:00Z" --maintenance-window-end "2022-01-20T05:00:00Z" --maintenance-window-recurrence "FREQ=WEEKLY;BYDAY=SA,SU" --labels owner={USERNAME} --workload-pool "{PROJECT_NAME}.svc.id.goog" --enable-shielded-nodes --node-locations {NODE_LOCATIONS}'

ADD_GPU_COMMAND_TMPL = 'gcloud beta container --project "{PROJECT_NAME}" node-pools create "gpu-n1-highmem-4-t4" --cluster "{CLUSTER_NAME}" --region "{REGION}" --machine-type "n1-highmem-4" --accelerator "type=nvidia-tesla-t4,count=1" --image-type "COS_CONTAINERD" --disk-type "pd-standard" --disk-size "64" --metadata disable-legacy-endpoints=true --scopes "https://www.googleapis.com/auth/devstorage.read_write","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/pubsub","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" --preemptible --num-nodes "1" --enable-autoupgrade --enable-autorepair --max-surge-upgrade 1 --max-unavailable-upgrade 0 --max-pods-per-node "16" --node-locations {NODE_LOCATIONS}'

CONFIGURE_DRIVERS_COMMAND_TMPL = "gcloud container clusters get-credentials {CLUSTER_NAME} --region {REGION} --project {PROJECT_NAME}; kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml"


def main():
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
        choices=["us-east1", "us-central1"],
        help="GCP region (e.g. us-east-1).",
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
        default="c,d",
        help='GCP Zones for GPU node locations (e.g. "c,d" for us-east1-c, and us-east1-d).',
    )
    parser.add_argument("--add_gpu", action="store_true", help="Add GPU node pool.")

    args = parser.parse_args()
    # cluster_version = args.cluster_version
    # if cluster_version is None:

    gpu_zones = [f"{args.region}-{zone_letter}" for zone_letter in args.gpu_zones.split(",")]
    cpu_zones = [f"{args.region}-{zone_letter}" for zone_letter in args.cpu_zones.split(",")]

    create_command = CREATE_COMMAND_TMPL
    create_command = create_command.replace("{REGION}", args.region)
    create_command = create_command.replace("{USERNAME}", args.username)
    create_command = create_command.replace("{PROJECT_NAME}", args.project_name)
    create_command = create_command.replace("{CLUSTER_NAME}", args.cluster_name)
    create_command = create_command.replace("{NODE_LOCATIONS}", ",".join(cpu_zones))
    print(f"Running: \n{create_command}")
    subprocess.call(create_command, shell=True)

    if args.add_gpu:
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


if __name__ == "__main__":
    main()
