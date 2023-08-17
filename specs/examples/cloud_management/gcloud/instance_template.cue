"@type":            "gcloud.create_instance_template"
template_name:      "ddp-workers"
project:            "zetta-research"
disk_size_gb:       64
machine_type:       "n1-highmem-2"
source_image:       "projects/debian-cloud/global/images/family/debian-12"
provisioning_model: "SPOT"
accelerators: [
    {
        "@type":  "gcloud.AcceleratorConfig"
        accelerator_count: 1
        accelerator_type: "nvidia-tesla-t4"
    }
]
