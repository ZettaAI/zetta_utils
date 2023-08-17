"@type":            "gcloud.create_instance_template"
template_name:      "ddp-workers"
project:            "zetta-research"
machine_type:       "n1-highmem-2"
provisioning_model: "SPOT"
accelerators: [
    {
        "@type":  "gcloud.AcceleratorConfig"
        accelerator_count: 1
        accelerator_type: "nvidia-tesla-t4"
    }
]
